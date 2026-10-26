#!/usr/bin/env python3
"""Generate TG-GPU-WP03 native-CUDA inference evidence on one NVIDIA GPU."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import linecache
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from tensorgraph.benchmarks.compile_isolation import isolated_graph_dtype_pairs
from tensorgraph.codegen.native_cuda import NativeCUDAEmitter
from tensorgraph.ir import Box, Expr, Id, Seq
from tensorgraph.pipeline.verified_elementwise import (
    compile_fx_elementwise,
    load_generated_kernel,
)
from tensorgraph.runtime.native_cuda import compile_native_cuda
from tensorgraph.signature import Signature
from tensorgraph.types import Obj

GRAPH_OPERATIONS: dict[str, tuple[str, ...]] = {
    "relu": ("ReLU",),
    "neg": ("Neg",),
    "sigmoid": ("Sigmoid",),
    "tanh": ("Tanh",),
    "exp": ("Exp",),
    "log": ("Log",),
    "relu_neg_sigmoid": ("ReLU", "Neg", "Sigmoid"),
    "tanh_neg_exp": ("Tanh", "Neg", "Exp"),
}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
LAUNCH_MODES = ("ordinary", "graph_replay", "copy_plus_graph_replay")
REGIMES = ("moderate", "near_zero", "saturation", "mixed_edge")


class UnaryChain(torch.nn.Module):
    def __init__(self, operations: tuple[str, ...]) -> None:
        super().__init__()
        self.operations = operations

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for operation in self.operations:
            if operation == "ReLU":
                value = torch.relu(value)
            elif operation == "Neg":
                value = torch.neg(value)
            elif operation == "Sigmoid":
                value = torch.sigmoid(value)
            elif operation == "Tanh":
                value = torch.tanh(value)
            elif operation == "Exp":
                value = torch.exp(value)
            elif operation == "Log":
                value = torch.log(value)
            else:  # pragma: no cover
                raise AssertionError(operation)
        return value


def _signature(operations: tuple[str, ...]) -> Signature:
    tensor = Obj("Tensor")
    signature = Signature()
    for operation in sorted(set(operations)):
        signature.add(operation, tensor, tensor, traits={"elementwise"})
    return signature


def _expression(operations: tuple[str, ...]) -> Expr:
    tensor = Obj("Tensor")
    if not operations:
        return Id(tensor)
    result: Expr = Box(operations[0])
    for operation in operations[1:]:
        result = Seq(result, Box(operation))
    return result


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "unknown"


def _make_input(
    graph: str,
    regime: str,
    size: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    has_log = "Log" in GRAPH_OPERATIONS[graph]
    if has_log:
        if regime == "moderate":
            value = torch.rand(size, device="cuda", dtype=dtype, generator=generator) * 4 + 0.125
        elif regime == "near_zero":
            floor = 2**-12 if dtype == torch.float16 else 2**-24
            value = torch.rand(size, device="cuda", dtype=dtype, generator=generator)
            value = value * (floor * 16) + floor
        elif regime == "saturation":
            exponents = torch.linspace(-4, 4, size, device="cuda", dtype=torch.float32)
            value = torch.pow(torch.tensor(10.0, device="cuda"), exponents).to(dtype)
        elif regime == "mixed_edge":
            base = torch.rand(size, device="cuda", dtype=dtype, generator=generator) + 0.125
            edge = torch.tensor(
                [0.125, 0.5, 1.0, 2.0, 16.0, float("inf")],
                device="cuda",
                dtype=dtype,
            )
            base[: edge.numel()] = edge
            value = base
        else:
            raise ValueError(regime)
        return value.contiguous()

    if regime == "moderate":
        value = torch.randn(size, device="cuda", dtype=dtype, generator=generator) * 3
    elif regime == "near_zero":
        value = torch.randn(size, device="cuda", dtype=dtype, generator=generator) * 1e-4
    elif regime == "saturation":
        value = torch.linspace(-20, 20, size, device="cuda", dtype=torch.float32).to(dtype)
    elif regime == "mixed_edge":
        value = torch.randn(size, device="cuda", dtype=dtype, generator=generator)
        edge = torch.tensor(
            [
                -float("inf"),
                -20.0,
                -1.0,
                -0.0,
                0.0,
                1.0,
                20.0,
                float("inf"),
                float("nan"),
            ],
            device="cuda",
            dtype=dtype,
        )
        value[: edge.numel()] = edge
    else:
        raise ValueError(regime)
    return value.contiguous()


def _timing(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def _cuda_samples(function: Callable[[], Any], warmup: int, repetitions: int) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return _timing(samples)


def _numerical(actual: torch.Tensor, expected: torch.Tensor, dtype_name: str) -> dict[str, object]:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if bool(finite.any()):
        difference = torch.abs(actual[finite].float() - expected[finite].float())
        denominator = torch.maximum(torch.abs(expected[finite].float()), torch.tensor(1e-12, device="cuda"))
        maximum_absolute = float(difference.max().item())
        maximum_relative = float((difference / denominator).max().item())
    else:
        maximum_absolute = 0.0
        maximum_relative = 0.0
    nan_match = bool(torch.equal(torch.isnan(actual), torch.isnan(expected)))
    positive_inf_match = bool(torch.equal(torch.isposinf(actual), torch.isposinf(expected)))
    negative_inf_match = bool(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
    tolerance = {
        "float16": (5e-3, 5e-4),
        "bfloat16": (1e-2, 1e-2),
        "float32": (2e-5, 2e-6),
    }[dtype_name]
    close = torch.allclose(actual, expected, rtol=tolerance[0], atol=tolerance[1], equal_nan=True)
    passed = bool(close and nan_match and positive_inf_match and negative_inf_match)
    return {
        "passed": passed,
        "rtol": tolerance[0],
        "atol": tolerance[1],
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "nan_classification_match": nan_match,
        "positive_infinity_classification_match": positive_inf_match,
        "negative_infinity_classification_match": negative_inf_match,
    }


def _load_direct_triton(operations: tuple[str, ...]) -> Any:
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def direct_triton_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):",
        "    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n",
        "    value = tl.load(x_ptr + offsets, mask=mask, other=0.0)",
    ]
    promoted = False
    for operation in operations:
        if operation in {"Sigmoid", "Tanh", "Exp", "Log"} and not promoted:
            lines.append("    value = value.to(tl.float32)")
            promoted = True
        if operation == "ReLU":
            lines.append("    value = tl.where((value > 0) | (value != value), value, 0.0)")
        elif operation == "Neg":
            lines.append("    value = -value")
        elif operation == "Sigmoid":
            lines.append("    value = 1.0 / (1.0 + tl.exp(-value))")
        elif operation == "Tanh":
            lines.append("    value = 2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0")
        elif operation == "Exp":
            lines.append("    value = tl.exp(value)")
        elif operation == "Log":
            lines.append("    value = tl.log(value)")
    lines.append("    tl.store(y_ptr + offsets, value, mask=mask)")
    source = "\n".join(lines) + "\n"
    filename = f"<wp03-direct-triton:{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["direct_triton_kernel"]


def _triton_callable(kernel: Any, input_tensor: torch.Tensor) -> Callable[[], torch.Tensor]:
    import triton

    def execute() -> torch.Tensor:
        output = torch.empty_like(input_tensor)
        grid = (triton.cdiv(input_tensor.numel(), 256),)
        kernel[grid](input_tensor, output, input_tensor.numel(), BLOCK_SIZE=256)
        return output

    return execute


def _baseline_record(
    function: Callable[[], Any] | None,
    warmup: int,
    repetitions: int,
    unsupported_reason: str | None = None,
) -> dict[str, object]:
    if function is None:
        return {"status": "unsupported", "reason": unsupported_reason or "unavailable"}
    try:
        return {"status": "passed", "timing": _cuda_samples(function, warmup, repetitions)}
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _cell_key(graph: str, dtype_name: str, regime: str, launch_mode: str) -> str:
    return f"{graph}__{dtype_name}__{regime}__{launch_mode}"


def _unsupported_cells(
    graph: str,
    operations: tuple[str, ...],
    dtype_name: str,
    regimes: list[str],
    launch_modes: list[str],
    source: str,
    source_sha: str,
    abi: dict[str, object],
    reason: str,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for regime, launch_mode in itertools.product(regimes, launch_modes):
        cells.append(
            {
                "key": _cell_key(graph, dtype_name, regime, launch_mode),
                "graph": graph,
                "operations": list(operations),
                "dtype": dtype_name,
                "regime": regime,
                "launch_mode": launch_mode,
                "disposition": "unsupported",
                "unsupported_reason": reason,
                "generated_source": source,
                "generated_source_sha256": source_sha,
                "abi": abi,
                "compiler": {"status": "unsupported", "reason": reason},
                "numerical": {},
                "timings": {},
                "baselines": {},
            }
        )
    return cells


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    commit_sha = _git_text("rev-parse", "HEAD")
    dirty = bool(_git_text("status", "--porcelain"))
    if dirty:
        raise RuntimeError("WP03 evidence requires a clean worktree")

    cells: list[dict[str, object]] = []
    for graph, dtype_name in isolated_graph_dtype_pairs(
        args.graphs,
        args.dtypes,
        torch._dynamo.reset,
    ):
        operations = GRAPH_OPERATIONS[graph]
        expression = _expression(operations)
        log_domain = "strict_positive" if operations and operations[0] == "Log" else None
        artifact = NativeCUDAEmitter(_signature(operations)).emit_artifact(
            expression,
            dtype=dtype_name,
            log_domain=log_domain,
        )
        capability = torch.cuda.get_device_capability()
        if dtype_name == "bfloat16" and capability[0] < 8:
            cells.extend(
                _unsupported_cells(
                    graph,
                    operations,
                    dtype_name,
                    args.regimes,
                    args.launch_modes,
                    artifact.generated_source,
                    artifact.source_sha256,
                    artifact.abi,
                    "bfloat16 requires Ampere-or-newer hardware",
                )
            )
            continue

        try:
            compile_start = time.perf_counter_ns()
            executable = compile_native_cuda(artifact, verbose=args.verbose_compile)
            compile_total_ns = time.perf_counter_ns() - compile_start
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for regime, launch_mode in itertools.product(args.regimes, args.launch_modes):
                cells.append(
                    {
                        "key": _cell_key(graph, dtype_name, regime, launch_mode),
                        "graph": graph,
                        "operations": list(operations),
                        "dtype": dtype_name,
                        "regime": regime,
                        "launch_mode": launch_mode,
                        "disposition": "failed",
                        "failure_reason": reason,
                        "generated_source": artifact.generated_source,
                        "generated_source_sha256": artifact.source_sha256,
                        "abi": artifact.abi,
                        "compiler": {"status": "failed", "reason": reason},
                        "numerical": {},
                        "timings": {},
                        "baselines": {},
                    }
                )
            continue

        model = UnaryChain(operations).cuda().eval()
        triton_artifact = compile_fx_elementwise(model)
        generated_triton = load_generated_kernel(triton_artifact)
        direct_triton = _load_direct_triton(operations)
        try:
            compiled_model: Callable[[torch.Tensor], torch.Tensor] | None = torch.compile(
                model,
                fullgraph=True,
            )
            inductor_reason = None
        except Exception as exc:  # pragma: no cover - stack dependent
            compiled_model = None
            inductor_reason = f"{type(exc).__name__}: {exc}"

        for regime in args.regimes:
            cell_seed = args.seed + abs(hash((graph, dtype_name, regime))) % 1_000_000
            input_tensor = _make_input(
                graph,
                regime,
                args.size,
                DTYPES[dtype_name],
                cell_seed,
            )
            expected = model(input_tensor)
            ordinary_output = torch.empty_like(input_tensor)
            executable.run_out(input_tensor, ordinary_output)
            captured = executable.capture(input_tensor, warmup=args.warmup)
            graph_output_0 = captured.replay(input_tensor, clone_output=True)
            changed_input = _make_input(
                graph,
                regime,
                args.size,
                DTYPES[dtype_name],
                cell_seed + 1,
            )
            changed_expected = model(changed_input)
            changed_actual = captured.replay(changed_input, clone_output=True)
            changed_numerical = _numerical(changed_actual, changed_expected, dtype_name)

            eager_baseline = _baseline_record(
                lambda: model(input_tensor),
                args.warmup,
                args.repetitions,
            )
            inductor_baseline = _baseline_record(
                None if compiled_model is None else lambda: compiled_model(input_tensor),
                args.warmup,
                args.repetitions,
                inductor_reason,
            )
            tensorgraph_triton_baseline = _baseline_record(
                lambda: generated_triton.run(input_tensor),
                args.warmup,
                args.repetitions,
            )
            direct_triton_baseline = _baseline_record(
                _triton_callable(direct_triton, input_tensor),
                args.warmup,
                args.repetitions,
            )

            def ordinary() -> None:
                executable.run_out(input_tensor, ordinary_output)

            def replay_only() -> None:
                captured.graph.replay()

            def copy_and_replay() -> None:
                captured.replay(input_tensor)

            mode_functions = {
                "ordinary": ordinary,
                "graph_replay": replay_only,
                "copy_plus_graph_replay": copy_and_replay,
            }
            mode_outputs = {
                "ordinary": ordinary_output,
                "graph_replay": graph_output_0,
                "copy_plus_graph_replay": captured.replay(input_tensor, clone_output=True),
            }
            for launch_mode in args.launch_modes:
                numerical = _numerical(mode_outputs[launch_mode], expected, dtype_name)
                numerical["changed_input_passed"] = changed_numerical["passed"]
                numerical["changed_input"] = changed_numerical
                timing = _cuda_samples(
                    mode_functions[launch_mode],
                    args.warmup,
                    args.repetitions,
                )
                baselines = {
                    "pytorch_eager": eager_baseline,
                    "torch_compile": inductor_baseline,
                    "tensorgraph_triton": tensorgraph_triton_baseline,
                    "direct_triton": direct_triton_baseline,
                    "tensorgraph_native_cuda": {"status": "passed", "timing": timing},
                    "direct_native_cuda": {
                        "status": "unsupported",
                        "reason": "independent direct-native baseline deferred; generated source remains exact",
                    },
                }
                passed = bool(numerical["passed"] and numerical["changed_input_passed"])
                cell: dict[str, object] = {
                    "key": _cell_key(graph, dtype_name, regime, launch_mode),
                    "graph": graph,
                    "operations": list(operations),
                    "dtype": dtype_name,
                    "regime": regime,
                    "launch_mode": launch_mode,
                    "disposition": "passed" if passed else "failed",
                    "generated_source": artifact.generated_source,
                    "generated_source_sha256": artifact.source_sha256,
                    "triton_generated_source_sha256": triton_artifact.source_sha256,
                    "abi": artifact.abi,
                    "compiler": {
                        "status": "compiled",
                        "compile_total_ns": compile_total_ns,
                        "runtime_phase_ns": executable.phase_ns,
                        "identity": executable.compiler_identity,
                        "cuda_graph_capture_ns": captured.capture_ns,
                    },
                    "numerical": numerical,
                    "timings": {launch_mode: timing},
                    "baselines": baselines,
                    "seed": cell_seed,
                    "size": args.size,
                }
                if not passed:
                    cell["failure_reason"] = "ordinary or changed-input numerical gate failed"
                cells.append(cell)

    expected_count = len(args.graphs) * len(args.dtypes) * len(args.regimes) * len(
        args.launch_modes
    )
    unsupported_keys = [cell["key"] for cell in cells if cell["disposition"] == "unsupported"]
    failed_keys = [cell["key"] for cell in cells if cell["disposition"] == "failed"]
    passed_count = sum(cell["disposition"] == "passed" for cell in cells)
    properties = torch.cuda.get_device_properties(0)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = "unavailable"
    return {
        "schema": "tensorgraph.evidence.native-cuda-inference.v1",
        "package": "TG-GPU-WP03",
        "repository": "fyremael/TENSORGRAPH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "dirty_worktree": False,
        "command": sys.argv,
        "stack_id": args.stack_id,
        "seed": args.seed,
        "size": args.size,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "request": {
            "graphs": args.graphs,
            "dtypes": args.dtypes,
            "regimes": args.regimes,
            "launch_modes": args.launch_modes,
        },
        "torch_compile_isolation": {
            "mode": "torch._dynamo.reset_per_graph_family",
            "graph_families": args.graphs,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "triton": triton_version,
            "cuda_runtime": torch.version.cuda,
            "driver": _driver_version(),
        },
        "hardware": {
            "gpu_index": 0,
            "gpu_name": properties.name,
            "gpu_compute_capability": list(torch.cuda.get_device_capability()),
            "gpu_total_memory_bytes": properties.total_memory,
            "cpu": platform.processor(),
        },
        "cells": cells,
        "matrix_summary": {
            "requested_cells": expected_count,
            "recorded_cells": len(cells),
            "passed_cells": passed_count,
            "unsupported_cells": len(unsupported_keys),
            "failed_cells": len(failed_keys),
            "unsupported_keys": unsupported_keys,
            "failed_keys": failed_keys,
        },
        "evidence_complete_for_requested_matrix": len(cells) == expected_count,
        "promotion_claim": False,
        "claim_boundary": (
            "Bounded native-CUDA unary inference evidence only. This artifact does not "
            "establish complete transformer decoding or performance portability."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-id", required=True)
    parser.add_argument("--graphs", nargs="+", choices=sorted(GRAPH_OPERATIONS), default=["relu_neg_sigmoid"])
    parser.add_argument("--dtypes", nargs="+", choices=sorted(DTYPES), default=["float16", "bfloat16", "float32"])
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--launch-modes", nargs="+", choices=LAUNCH_MODES, default=list(LAUNCH_MODES))
    parser.add_argument("--size", type=int, default=65_536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed-cells", action="store_true")
    parser.add_argument("--verbose-compile", action="store_true")
    args = parser.parse_args()
    if args.size < 1 or args.warmup < 1 or args.repetitions < 1:
        parser.error("size, warmup, and repetitions must be positive")
    return args


def main() -> int:
    args = parse_args()
    document = run_matrix(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    failed = int(document["matrix_summary"]["failed_cells"])
    print(json.dumps(document["matrix_summary"], indent=2, sort_keys=True))
    return 0 if args.allow_failed_cells or failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
