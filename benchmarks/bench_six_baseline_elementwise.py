"""Run the six-baseline GPU matrix for the bounded elementwise compiler path.

The matrix distinguishes graph simplification from backend code generation:

A. PyTorch eager on ReLU -> ReLU -> Neg.
B. PyTorch eager on ReLU -> Neg.
C. torch.compile on ReLU -> ReLU -> Neg.
D. torch.compile on ReLU -> Neg.
E. TENSORGRAPH extraction and generated Triton on ReLU -> Neg.
F. An independent direct Triton implementation of ReLU -> Neg.

The script fails closed unless all six implementations execute on CUDA, match
the source PyTorch graph numerically, and the repository worktree is clean.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import linecache
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tensorgraph.pipeline import compile_fx_elementwise, load_generated_kernel

SOURCE_EAGER = "pytorch_eager_source"
OPTIMIZED_EAGER = "pytorch_eager_optimized"
SOURCE_COMPILED = "torch_compile_source"
OPTIMIZED_COMPILED = "torch_compile_optimized"
TENSORGRAPH_GENERATED = "tensorgraph_generated"
DIRECT_TRITON = "direct_triton_reference"

BASELINE_ORDER = (
    SOURCE_EAGER,
    OPTIMIZED_EAGER,
    SOURCE_COMPILED,
    OPTIMIZED_COMPILED,
    TENSORGRAPH_GENERATED,
    DIRECT_TRITON,
)

_DIRECT_KERNEL_NAME = "direct_relu_neg_kernel"
DIRECT_TRITON_SOURCE = """import triton
import triton.language as tl

@triton.jit
def direct_relu_neg_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    value = tl.where(value > 0.0, value, 0.0)
    value = -value
    tl.store(y_ptr + offsets, value, mask=mask)
"""


class SourceGraph(torch.nn.Module):
    """Unoptimized source graph with a provably redundant ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.relu(torch.relu(x))


class OptimizedGraph(torch.nn.Module):
    """Manually normalized graph after ReLU idempotence."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.relu(x)


@dataclass
class DirectTritonKernel:
    kernel: Any
    source_sha256: str

    def run(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        import triton

        if not x.is_cuda:
            raise RuntimeError("direct Triton reference requires a CUDA tensor")
        if not x.is_contiguous():
            raise ValueError("direct Triton reference requires contiguous input")
        if not x.dtype.is_floating_point:
            raise TypeError("direct Triton reference requires a floating dtype")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")

        output = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements == 0:
            return output
        grid = (triton.cdiv(n_elements, block_size),)
        self.kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
        return output


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_identity() -> tuple[str, bool]:
    commit = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))
    if len(commit) != 40:
        raise RuntimeError("could not resolve an exact 40-character Git commit SHA")
    return commit, dirty


def _load_direct_triton() -> DirectTritonKernel:
    source_sha256 = hashlib.sha256(DIRECT_TRITON_SOURCE.encode("utf-8")).hexdigest()
    filename = f"<tensorgraph-direct-reference:{source_sha256}>"
    linecache.cache[filename] = (
        len(DIRECT_TRITON_SOURCE),
        None,
        DIRECT_TRITON_SOURCE.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, Any] = {}
    exec(compile(DIRECT_TRITON_SOURCE, filename, "exec"), namespace)
    kernel = namespace.get(_DIRECT_KERNEL_NAME)
    if kernel is None:
        raise RuntimeError("direct Triton source did not define the expected kernel")
    return DirectTritonKernel(kernel=kernel, source_sha256=source_sha256)


def _cuda_sample_ms(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _first_execution_ns(fn: Callable[[], Any]) -> tuple[Any, int]:
    start = time.perf_counter_ns()
    output = fn()
    torch.cuda.synchronize()
    return output, time.perf_counter_ns() - start


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("cannot summarize an empty timing sample")
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p05_ms": _percentile(samples, 0.05),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _versions() -> dict[str, Any]:
    import triton

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "pid": os.getpid(),
    }


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _tolerances(name: str) -> tuple[float, float]:
    if name == "float32":
        return 1e-5, 1e-6
    if name == "float16":
        return 3e-3, 3e-3
    if name == "bfloat16":
        return 2e-2, 2e-2
    raise ValueError(name)


def _check_close(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} returned shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise RuntimeError(f"{name} returned dtype {actual.dtype}, expected {expected.dtype}")

    if expected.numel():
        absolute = (actual - expected).abs()
        max_abs_error = float(absolute.max().item())
        max_rel_error = float(
            (absolute / expected.abs().clamp_min(atol)).max().item()
        )
    else:
        max_abs_error = 0.0
        max_rel_error = 0.0

    passed = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
    if not passed:
        raise RuntimeError(
            f"numerical gate failed for {name}: "
            f"max_abs_error={max_abs_error}, max_rel_error={max_rel_error}"
        )
    return {
        "pass": True,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }


def _write_summary_csv(evidence: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "dtype",
        "size",
        "block_size",
        "baseline",
        "median_ms",
        "mean_ms",
        "p05_ms",
        "p95_ms",
        "speedup_vs_source_eager",
        "speedup_vs_optimized_eager",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for workload in evidence["workloads"]:
            for baseline_name in BASELINE_ORDER:
                baseline = workload["baselines"][baseline_name]
                writer.writerow(
                    {
                        "dtype": workload["dtype"],
                        "size": workload["size"],
                        "block_size": workload["block_size"],
                        "baseline": baseline_name,
                        "median_ms": baseline["summary"]["median_ms"],
                        "mean_ms": baseline["summary"]["mean_ms"],
                        "p05_ms": baseline["summary"]["p05_ms"],
                        "p95_ms": baseline["summary"]["p95_ms"],
                        "speedup_vs_source_eager": baseline[
                            "speedup_vs_source_eager"
                        ],
                        "speedup_vs_optimized_eager": baseline[
                            "speedup_vs_optimized_eager"
                        ],
                    }
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no substitute or estimated result is permitted")
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Triton is required for the six-baseline matrix") from exc
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is required for the six-baseline matrix")

    commit, dirty = _git_identity()
    if dirty:
        raise RuntimeError("benchmark evidence requires a clean Git worktree")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    source_model = SourceGraph().cuda().eval()
    optimized_model = OptimizedGraph().cuda().eval()

    compiler_start = time.perf_counter_ns()
    artifact = compile_fx_elementwise(source_model)
    tensorgraph_compile_total_ns = time.perf_counter_ns() - compiler_start
    if artifact.source_pretty != "((ReLU ; ReLU) ; Neg)":
        raise RuntimeError(f"unexpected TENSORGRAPH source expression: {artifact.source_pretty}")
    if artifact.optimized_pretty != "(ReLU ; Neg)":
        raise RuntimeError(
            f"unexpected TENSORGRAPH optimized expression: {artifact.optimized_pretty}"
        )

    source_load_start = time.perf_counter_ns()
    tensorgraph_kernel = load_generated_kernel(artifact)
    tensorgraph_source_load_ns = time.perf_counter_ns() - source_load_start
    direct_kernel = _load_direct_triton()

    source_compiled = torch.compile(
        source_model,
        backend=args.torch_compile_backend,
        mode=args.torch_compile_mode,
        fullgraph=True,
        dynamic=True,
    )
    optimized_compiled = torch.compile(
        optimized_model,
        backend=args.torch_compile_backend,
        mode=args.torch_compile_mode,
        fullgraph=True,
        dynamic=True,
    )

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    workloads: list[dict[str, Any]] = []

    for dtype_index, dtype_name in enumerate(args.dtypes):
        dtype = _dtype(dtype_name)
        rtol, atol = _tolerances(dtype_name)

        for size_index, size in enumerate(args.sizes):
            x = torch.randn(size, device="cuda", dtype=dtype)
            block_size = args.block_size

            functions: dict[str, Callable[[], torch.Tensor]] = {
                SOURCE_EAGER: lambda value=x: source_model(value),
                OPTIMIZED_EAGER: lambda value=x: optimized_model(value),
                SOURCE_COMPILED: lambda value=x: source_compiled(value),
                OPTIMIZED_COMPILED: lambda value=x: optimized_compiled(value),
                TENSORGRAPH_GENERATED: lambda value=x, block=block_size: tensorgraph_kernel.run(
                    value, block_size=block
                ),
                DIRECT_TRITON: lambda value=x, block=block_size: direct_kernel.run(
                    value, block_size=block
                ),
            }

            first_execution_ns: dict[str, int] = {}
            first_outputs: dict[str, torch.Tensor] = {}
            for baseline_name in BASELINE_ORDER:
                output, elapsed_ns = _first_execution_ns(functions[baseline_name])
                first_outputs[baseline_name] = output
                first_execution_ns[baseline_name] = elapsed_ns

            reference = first_outputs[SOURCE_EAGER]
            numerical = {
                baseline_name: _check_close(
                    name=baseline_name,
                    actual=first_outputs[baseline_name],
                    expected=reference,
                    rtol=rtol,
                    atol=atol,
                )
                for baseline_name in BASELINE_ORDER
            }

            for _ in range(args.warmup):
                for baseline_name in BASELINE_ORDER:
                    functions[baseline_name]()
            torch.cuda.synchronize()

            samples = {baseline_name: [] for baseline_name in BASELINE_ORDER}
            rng = random.Random(
                args.seed
                + dtype_index * 1_000_003
                + size_index * 10_007
                + size
            )
            for _ in range(args.repetitions):
                order = list(BASELINE_ORDER)
                rng.shuffle(order)
                for baseline_name in order:
                    samples[baseline_name].append(
                        _cuda_sample_ms(functions[baseline_name])
                    )

            summaries = {
                baseline_name: _summary(samples[baseline_name])
                for baseline_name in BASELINE_ORDER
            }
            source_median = float(summaries[SOURCE_EAGER]["median_ms"])
            optimized_median = float(summaries[OPTIMIZED_EAGER]["median_ms"])

            baselines: dict[str, Any] = {}
            for baseline_name in BASELINE_ORDER:
                median = float(summaries[baseline_name]["median_ms"])
                baselines[baseline_name] = {
                    "first_execution_ns": first_execution_ns[baseline_name],
                    "first_execution_may_include_compilation": baseline_name
                    in {
                        SOURCE_COMPILED,
                        OPTIMIZED_COMPILED,
                        TENSORGRAPH_GENERATED,
                        DIRECT_TRITON,
                    },
                    "numerical": numerical[baseline_name],
                    "raw_ms": samples[baseline_name],
                    "summary": summaries[baseline_name],
                    "speedup_vs_source_eager": source_median / median,
                    "speedup_vs_optimized_eager": optimized_median / median,
                }

            tensorgraph_median = float(
                summaries[TENSORGRAPH_GENERATED]["median_ms"]
            )
            direct_median = float(summaries[DIRECT_TRITON]["median_ms"])
            workloads.append(
                {
                    "dtype": dtype_name,
                    "size": size,
                    "shape": [size],
                    "block_size": block_size,
                    "baselines": baselines,
                    "decomposition": {
                        "source_to_optimized_eager_speedup": (
                            source_median / optimized_median
                        ),
                        "tensorgraph_vs_direct_triton_ratio": (
                            tensorgraph_median / direct_median
                        ),
                        "tensorgraph_speedup_vs_source_eager": (
                            source_median / tensorgraph_median
                        ),
                        "tensorgraph_speedup_vs_optimized_eager": (
                            optimized_median / tensorgraph_median
                        ),
                    },
                }
            )

            print(
                f"{dtype_name:8s} N={size:9d} | "
                f"A={source_median:.6f} ms "
                f"B={optimized_median:.6f} ms "
                f"C={float(summaries[SOURCE_COMPILED]['median_ms']):.6f} ms "
                f"D={float(summaries[OPTIMIZED_COMPILED]['median_ms']):.6f} ms "
                f"E={tensorgraph_median:.6f} ms "
                f"F={direct_median:.6f} ms"
            )

    return {
        "schema": "tensorgraph.evidence.six-baseline-elementwise.v1",
        "admissible": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "fyremael/TENSORGRAPH",
        "commit_sha": commit,
        "dirty_worktree": dirty,
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "baseline_order": list(BASELINE_ORDER),
        "versions": _versions(),
        "hardware": {
            "gpu_index": device_index,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
            "cpu": platform.processor(),
        },
        "torch_compile": {
            "backend": args.torch_compile_backend,
            "mode": args.torch_compile_mode,
            "fullgraph": True,
            "dynamic": True,
        },
        "graphs": {
            "source": "ReLU -> ReLU -> Neg",
            "optimized": "ReLU -> Neg",
        },
        "tensorgraph_compiler": {
            "source_expr": artifact.source_pretty,
            "optimized_expr": artifact.optimized_pretty,
            "rewrite_summary": artifact.rewrite_summary,
            "generated_source_sha256": artifact.source_sha256,
            "generated_source": artifact.generated_source,
            "phase_ns": artifact.phase_ns,
            "compile_total_ns": tensorgraph_compile_total_ns,
            "source_load_ns": tensorgraph_source_load_ns,
        },
        "direct_triton_reference": {
            "generated_source_sha256": direct_kernel.source_sha256,
            "generated_source": DIRECT_TRITON_SOURCE,
        },
        "claim_boundary": {
            "validated": (
                "Forward-only CUDA execution and timing of the six bounded "
                "ReLU-ReLU-Neg comparison lanes on recorded hardware."
            ),
            "not_validated": [
                "general FX compilation",
                "backward or training correctness",
                "production readiness",
                "performance on unrecorded hardware",
            ],
        },
        "workloads": workloads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1_024, 65_536, 1_048_576, 4_194_304],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float16", "bfloat16"],
        default=["float32", "float16"],
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument(
        "--torch-compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="default",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if any(size < 0 for size in args.sizes):
        parser.error("sizes must be non-negative")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be non-negative and repetitions must be positive")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        parser.error("block-size must be a positive power of two")
    return args


def main() -> int:
    args = parse_args()
    try:
        evidence = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_output = args.summary_output or args.output.with_suffix(".csv")
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_csv(evidence, summary_output)
    except Exception as exc:
        print(f"EVIDENCE REJECTED: {exc}", file=sys.stderr)
        return 1

    print(f"Evidence written: {args.output}")
    print(f"Summary written: {summary_output}")
    print(f"Commit: {evidence['commit_sha']}")
    print(
        "TENSORGRAPH source SHA-256: "
        f"{evidence['tensorgraph_compiler']['generated_source_sha256']}"
    )
    print(
        "Direct Triton source SHA-256: "
        f"{evidence['direct_triton_reference']['generated_source_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
