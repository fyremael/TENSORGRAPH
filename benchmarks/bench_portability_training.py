"""Produce TG-GPU-WP02 portability and training-semantics evidence.

The runner executes exact generated forward source and generated input-gradient
source. It writes a disposition for every requested matrix cell, including
unsupported dtype/hardware combinations and failures. It exits nonzero on a
host-wide evidence failure or when a requested supported cell is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tensorgraph.pipeline import (
    compile_fx_elementwise_training,
    load_generated_backward_kernel,
    load_generated_kernel,
)

OPERATIONS = ("sigmoid", "tanh")
DTYPES = ("float16", "bfloat16", "float32")
REGIMES = (
    "moderate",
    "positive_saturation",
    "negative_saturation",
    "near_zero",
    "mixed_edge",
)
DIRECTIONS = ("forward", "forward_backward")
DISPOSITIONS = ("passed", "unsupported", "failed")

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_TOLERANCES = {
    "float16": {
        "forward": {"rtol": 5e-3, "atol": 5e-4},
        "gradient": {"rtol": 8e-3, "atol": 8e-4},
    },
    "bfloat16": {
        "forward": {"rtol": 2e-2, "atol": 2e-3},
        "gradient": {"rtol": 3e-2, "atol": 3e-3},
    },
    "float32": {
        "forward": {"rtol": 2e-5, "atol": 2e-6},
        "gradient": {"rtol": 5e-5, "atol": 5e-6},
    },
}


class _Sigmoid(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class _Tanh(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


def _model(operation: str) -> torch.nn.Module:
    mapping: dict[str, torch.nn.Module] = {
        "sigmoid": _Sigmoid(),
        "tanh": _Tanh(),
    }
    return mapping[operation]


def matrix_key(
    *,
    dtype: str,
    operation: str,
    regime: str,
    direction: str,
) -> str:
    return "/".join((dtype, operation, regime, direction))


def expected_matrix_keys(
    dtypes: Iterable[str],
    operations: Iterable[str],
    regimes: Iterable[str],
    directions: Iterable[str],
) -> set[str]:
    return {
        matrix_key(
            dtype=dtype,
            operation=operation,
            regime=regime,
            direction=direction,
        )
        for dtype in dtypes
        for operation in operations
        for regime in regimes
        for direction in directions
    }


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


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _cuda_sample_ms(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0]


def _versions() -> dict[str, Any]:
    import triton

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": _driver_version(),
        "pid": os.getpid(),
    }


def _dtype_support_reason(dtype_name: str, compute_capability: tuple[int, int]) -> str | None:
    if dtype_name == "bfloat16" and compute_capability[0] < 8:
        return "bfloat16 requires Ampere-or-newer hardware for this evidence contract"
    return None


def _regime_values(
    regime: str,
    *,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if regime == "moderate":
        return torch.randn(size, generator=generator, device=device, dtype=torch.float32).to(
            dtype
        ) * 2.0
    if regime == "positive_saturation":
        base = 12.0 if dtype != torch.float32 else 30.0
        values = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
        return (base + 4.0 * values).to(dtype)
    if regime == "negative_saturation":
        base = 12.0 if dtype != torch.float32 else 30.0
        values = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
        return (-base - 4.0 * values).to(dtype)
    if regime == "near_zero":
        scale = 32.0 * torch.finfo(dtype).eps
        values = torch.randn(size, generator=generator, device=device, dtype=torch.float32)
        return (values * scale).to(dtype)
    if regime == "mixed_edge":
        finfo = torch.finfo(dtype)
        finite_limit = min(float(finfo.max), 1.0e4)
        tiny = float(finfo.tiny)
        pattern = torch.tensor(
            [
                -float("inf"),
                -finite_limit,
                -30.0,
                -12.0,
                -1.0,
                -tiny,
                -0.0,
                0.0,
                tiny,
                1.0,
                12.0,
                30.0,
                finite_limit,
                float("inf"),
                float("nan"),
            ],
            device=device,
            dtype=dtype,
        )
        repeats = (size + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(repeats)[:size].contiguous()
    raise ValueError(f"unknown input regime: {regime}")


def _grad_values(
    *,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(size, generator=generator, device=device, dtype=torch.float32).to(dtype)


def _numerical_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    actual_nan = torch.isnan(actual_flat)
    expected_nan = torch.isnan(expected_flat)
    actual_posinf = torch.isposinf(actual_flat)
    expected_posinf = torch.isposinf(expected_flat)
    actual_neginf = torch.isneginf(actual_flat)
    expected_neginf = torch.isneginf(expected_flat)
    special_match = bool(
        torch.equal(actual_nan, expected_nan)
        and torch.equal(actual_posinf, expected_posinf)
        and torch.equal(actual_neginf, expected_neginf)
    )

    finite_mask = torch.isfinite(actual_flat) & torch.isfinite(expected_flat)
    if bool(finite_mask.any()):
        actual_finite = actual_flat[finite_mask].float()
        expected_finite = expected_flat[finite_mask].float()
        absolute = (actual_finite - expected_finite).abs()
        relative = absolute / expected_finite.abs().clamp_min(atol)
        max_abs_error = float(absolute.max().item())
        max_rel_error = float(relative.max().item())
    else:
        max_abs_error = 0.0
        max_rel_error = 0.0

    passed = special_match and bool(
        torch.allclose(actual_flat, expected_flat, rtol=rtol, atol=atol, equal_nan=True)
    )
    return {
        "pass": passed,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "finite_count": int(finite_mask.sum().item()),
        "actual_nan_count": int(actual_nan.sum().item()),
        "expected_nan_count": int(expected_nan.sum().item()),
        "actual_posinf_count": int(actual_posinf.sum().item()),
        "expected_posinf_count": int(expected_posinf.sum().item()),
        "actual_neginf_count": int(actual_neginf.sum().item()),
        "expected_neginf_count": int(expected_neginf.sum().item()),
        "special_values_match": special_match,
    }


def _failure_samples(
    x: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    limit: int = 16,
) -> list[dict[str, float | str]]:
    difference = (actual.float() - expected.float()).abs()
    mismatch = ~torch.isclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    indices = torch.nonzero(mismatch.reshape(-1), as_tuple=False).reshape(-1)[:limit]
    samples: list[dict[str, float | str]] = []
    for index in indices.tolist():
        values: dict[str, float | str] = {"index": float(index)}
        for name, tensor in (("input", x), ("actual", actual), ("expected", expected)):
            value = float(tensor.reshape(-1)[index].float().item())
            values[name] = repr(value)
        values["absolute_error"] = repr(float(difference.reshape(-1)[index].item()))
        samples.append(values)
    return samples


def _reference_forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_x = x.detach().clone().requires_grad_(True)
    reference_y = model(reference_x)
    reference_y.backward(grad_output)
    if reference_x.grad is None:
        raise RuntimeError("PyTorch reference did not produce an input gradient")
    return reference_y.detach(), reference_x.grad.detach()


def _timed_reference_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def _timed_reference_forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> None:
    reference_x = x.detach().requires_grad_(True)
    y = model(reference_x)
    y.backward(grad_output)


def _cell_record(
    *,
    operation: str,
    dtype_name: str,
    regime: str,
    direction: str,
    size: int,
    seed: int,
    model: torch.nn.Module,
    generated_forward: Any,
    generated_backward: Any,
    warmup: int,
    repetitions: int,
    block_size: int,
    first_forward_includes_jit: bool,
    first_backward_includes_jit: bool,
) -> dict[str, Any]:
    dtype = _DTYPE_MAP[dtype_name]
    device = torch.device("cuda", torch.cuda.current_device())
    x = _regime_values(regime, size=size, dtype=dtype, device=device, seed=seed)
    grad_output = _grad_values(
        size=size,
        dtype=dtype,
        device=device,
        seed=seed + 1_000_003,
    )

    first_forward_start = time.perf_counter_ns()
    candidate_y = generated_forward.run(x, block_size=block_size)
    torch.cuda.synchronize()
    first_forward_execution_ns = time.perf_counter_ns() - first_forward_start

    candidate_grad: torch.Tensor | None = None
    first_backward_execution_ns: int | None = None
    if direction == "forward_backward":
        first_backward_start = time.perf_counter_ns()
        candidate_grad = generated_backward.run(
            x,
            candidate_y,
            grad_output,
            block_size=block_size,
        )
        torch.cuda.synchronize()
        first_backward_execution_ns = time.perf_counter_ns() - first_backward_start

    reference_y, reference_grad = _reference_forward_backward(model, x, grad_output)
    forward_tol = _TOLERANCES[dtype_name]["forward"]
    forward_numerical = _numerical_summary(candidate_y, reference_y, **forward_tol)

    gradient_numerical: dict[str, Any] | None = None
    if direction == "forward_backward":
        if candidate_grad is None:
            raise AssertionError("candidate gradient was not produced")
        gradient_tol = _TOLERANCES[dtype_name]["gradient"]
        gradient_numerical = _numerical_summary(
            candidate_grad,
            reference_grad,
            **gradient_tol,
        )

    passed = forward_numerical["pass"] and (
        direction == "forward" or bool(gradient_numerical and gradient_numerical["pass"])
    )

    if passed:
        for _ in range(warmup):
            if direction == "forward":
                _timed_reference_forward(model, x)
                generated_forward.run(x, block_size=block_size)
            else:
                _timed_reference_forward_backward(model, x, grad_output)
                y = generated_forward.run(x, block_size=block_size)
                generated_backward.run(x, y, grad_output, block_size=block_size)
        torch.cuda.synchronize()

        if direction == "forward":
            reference_fn = lambda: _timed_reference_forward(model, x)
            generated_fn = lambda: generated_forward.run(x, block_size=block_size)
        else:
            reference_fn = lambda: _timed_reference_forward_backward(model, x, grad_output)

            def generated_fn() -> None:
                y = generated_forward.run(x, block_size=block_size)
                generated_backward.run(x, y, grad_output, block_size=block_size)

        reference_samples = [_cuda_sample_ms(reference_fn) for _ in range(repetitions)]
        generated_samples = [_cuda_sample_ms(generated_fn) for _ in range(repetitions)]
    else:
        reference_samples = []
        generated_samples = []

    record: dict[str, Any] = {
        "key": matrix_key(
            dtype=dtype_name,
            operation=operation,
            regime=regime,
            direction=direction,
        ),
        "operation": operation,
        "dtype": dtype_name,
        "regime": regime,
        "direction": direction,
        "size": size,
        "shape": [size],
        "seed": seed,
        "disposition": "passed" if passed else "failed",
        "first_forward_execution_ns": first_forward_execution_ns,
        "first_forward_includes_jit": first_forward_includes_jit,
        "first_backward_execution_ns": first_backward_execution_ns,
        "first_backward_includes_jit": (
            direction == "forward_backward" and first_backward_includes_jit
        ),
        "forward_numerical": forward_numerical,
        "gradient_numerical": gradient_numerical,
        "reference_timing": {
            "raw_ms": reference_samples,
            "summary": _summary(reference_samples) if reference_samples else None,
        },
        "tensorgraph_timing": {
            "raw_ms": generated_samples,
            "summary": _summary(generated_samples) if generated_samples else None,
        },
    }
    if not passed:
        record["failure_stage"] = "numerical"
        record["failure_reason"] = "forward or input-gradient differential gate failed"
        record["failing_forward_samples"] = _failure_samples(x, candidate_y, reference_y)
        if candidate_grad is not None and gradient_numerical and not gradient_numerical["pass"]:
            record["failing_gradient_samples"] = _failure_samples(
                x,
                candidate_grad,
                reference_grad,
            )
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no substitute or estimated result is permitted")
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Triton is required; source generation alone is not evidence") from exc

    commit, dirty = _git_identity()
    if dirty:
        raise RuntimeError("benchmark evidence requires a clean Git worktree")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    compute_capability = (properties.major, properties.minor)

    compiler_records: dict[str, Any] = {}
    loaded: dict[str, tuple[torch.nn.Module, Any, Any] | None] = {}
    cells: list[dict[str, Any]] = []

    for operation in args.operations:
        try:
            model = _model(operation).cuda().eval()
            compile_start = time.perf_counter_ns()
            artifact = compile_fx_elementwise_training(model)
            compile_total_ns = time.perf_counter_ns() - compile_start

            forward_load_start = time.perf_counter_ns()
            generated_forward = load_generated_kernel(artifact.forward)
            forward_source_load_ns = time.perf_counter_ns() - forward_load_start

            backward_load_start = time.perf_counter_ns()
            generated_backward = load_generated_backward_kernel(artifact)
            backward_source_load_ns = time.perf_counter_ns() - backward_load_start
        except Exception as exc:
            compiler_records[operation] = {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
            loaded[operation] = None
            continue

        compiler_records[operation] = {
            "status": "compiled",
            "source_expr": artifact.forward.source_pretty,
            "optimized_expr": artifact.forward.optimized_pretty,
            "optimized_ops": list(artifact.optimized_ops),
            "rewrite_summary": artifact.forward.rewrite_summary,
            "forward_generated_source": artifact.forward.generated_source,
            "forward_generated_source_sha256": artifact.forward.source_sha256,
            "backward_generated_source": artifact.generated_backward_source,
            "backward_generated_source_sha256": artifact.backward_source_sha256,
            "forward_phase_ns": artifact.forward.phase_ns,
            "backward_source_generation_ns": artifact.backward_generation_ns,
            "compile_total_ns": compile_total_ns,
            "forward_source_load_ns": forward_source_load_ns,
            "backward_source_load_ns": backward_source_load_ns,
        }
        loaded[operation] = (model, generated_forward, generated_backward)

    seen_forward_specializations: set[tuple[str, str]] = set()
    seen_backward_specializations: set[tuple[str, str]] = set()
    cell_index = 0
    for dtype_name in args.dtypes:
        unsupported_reason = _dtype_support_reason(dtype_name, compute_capability)
        for operation in args.operations:
            loaded_operation = loaded[operation]
            for regime in args.regimes:
                for direction in args.directions:
                    key = matrix_key(
                        dtype=dtype_name,
                        operation=operation,
                        regime=regime,
                        direction=direction,
                    )
                    if loaded_operation is None:
                        cells.append(
                            {
                                "key": key,
                                "operation": operation,
                                "dtype": dtype_name,
                                "regime": regime,
                                "direction": direction,
                                "size": args.size,
                                "shape": [args.size],
                                "seed": args.seed + cell_index,
                                "disposition": "failed",
                                "failure_stage": "compilation",
                                "failure_reason": compiler_records[operation]["failure_reason"],
                                "forward_numerical": None,
                                "gradient_numerical": None,
                                "reference_timing": {"raw_ms": [], "summary": None},
                                "tensorgraph_timing": {"raw_ms": [], "summary": None},
                            }
                        )
                        cell_index += 1
                        continue
                    model, generated_forward, generated_backward = loaded_operation
                    if unsupported_reason is not None:
                        cells.append(
                            {
                                "key": key,
                                "operation": operation,
                                "dtype": dtype_name,
                                "regime": regime,
                                "direction": direction,
                                "size": args.size,
                                "shape": [args.size],
                                "seed": args.seed + cell_index,
                                "disposition": "unsupported",
                                "unsupported_reason": unsupported_reason,
                                "forward_numerical": None,
                                "gradient_numerical": None,
                                "reference_timing": {"raw_ms": [], "summary": None},
                                "tensorgraph_timing": {"raw_ms": [], "summary": None},
                            }
                        )
                        cell_index += 1
                        continue
                    try:
                        cells.append(
                            _cell_record(
                                operation=operation,
                                dtype_name=dtype_name,
                                regime=regime,
                                direction=direction,
                                size=args.size,
                                seed=args.seed + cell_index,
                                model=model,
                                generated_forward=generated_forward,
                                generated_backward=generated_backward,
                                warmup=args.warmup,
                                repetitions=args.repetitions,
                                block_size=args.block_size,
                                first_forward_includes_jit=(
                                    (operation, dtype_name) not in seen_forward_specializations
                                ),
                                first_backward_includes_jit=(
                                    direction == "forward_backward"
                                    and (operation, dtype_name)
                                    not in seen_backward_specializations
                                ),
                            )
                        )
                        seen_forward_specializations.add((operation, dtype_name))
                        if direction == "forward_backward":
                            seen_backward_specializations.add((operation, dtype_name))
                    except Exception as exc:  # retain a complete matrix disposition
                        cells.append(
                            {
                                "key": key,
                                "operation": operation,
                                "dtype": dtype_name,
                                "regime": regime,
                                "direction": direction,
                                "size": args.size,
                                "shape": [args.size],
                                "seed": args.seed + cell_index,
                                "disposition": "failed",
                                "failure_stage": "execution",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "forward_numerical": None,
                                "gradient_numerical": None,
                                "reference_timing": {"raw_ms": [], "summary": None},
                                "tensorgraph_timing": {"raw_ms": [], "summary": None},
                            }
                        )
                        seen_forward_specializations.add((operation, dtype_name))
                        if direction == "forward_backward":
                            seen_backward_specializations.add((operation, dtype_name))
                    cell_index += 1

    requested_keys = expected_matrix_keys(
        args.dtypes,
        args.operations,
        args.regimes,
        args.directions,
    )
    actual_keys = {cell["key"] for cell in cells}
    if actual_keys != requested_keys or len(cells) != len(requested_keys):
        raise RuntimeError("matrix construction omitted or duplicated requested cells")

    failed_cells = [cell["key"] for cell in cells if cell["disposition"] == "failed"]
    unsupported_cells = [
        cell["key"] for cell in cells if cell["disposition"] == "unsupported"
    ]

    return {
        "schema": "tensorgraph.evidence.portability-training.v1",
        "package": "TG-GPU-WP02",
        "repository": "fyremael/TENSORGRAPH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit,
        "dirty_worktree": dirty,
        "command": [sys.executable, *sys.argv],
        "stack_id": args.stack_id,
        "seed": args.seed,
        "size": args.size,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "block_size": args.block_size,
        "request": {
            "operations": list(args.operations),
            "dtypes": list(args.dtypes),
            "regimes": list(args.regimes),
            "directions": list(args.directions),
        },
        "versions": _versions(),
        "hardware": {
            "gpu_index": device_index,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(compute_capability),
            "cpu": platform.processor(),
        },
        "tolerances": _TOLERANCES,
        "compiler": compiler_records,
        "cells": cells,
        "matrix_summary": {
            "requested_cells": len(requested_keys),
            "recorded_cells": len(cells),
            "passed_cells": sum(cell["disposition"] == "passed" for cell in cells),
            "unsupported_cells": len(unsupported_cells),
            "failed_cells": len(failed_cells),
            "unsupported_keys": unsupported_cells,
            "failed_keys": failed_cells,
        },
        "evidence_complete_for_requested_matrix": not failed_cells,
        "promotion_claim": False,
        "claim_boundary": (
            "This artifact covers only the exact generated bounded Sigmoid and Tanh "
            "forward/input-gradient paths on the recorded hardware and software stack."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument("--stack-id", required=True)
    parser.add_argument("--size", type=int, default=65_536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed-cells", action="store_true")
    args = parser.parse_args()

    for name in ("operations", "dtypes", "regimes", "directions"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"{name} must not contain duplicates")
    if args.size <= 0:
        parser.error("size must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("warmup must be non-negative and repetitions must be positive")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        parser.error("block-size must be a positive power of two")
    if not args.stack_id.strip():
        parser.error("stack-id must not be empty")
    return args


def main() -> int:
    args = parse_args()
    try:
        evidence = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        args.output.write_text(raw, encoding="utf-8")
    except Exception as exc:
        print(f"EVIDENCE REJECTED: {exc}", file=sys.stderr)
        return 1

    artifact_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"Evidence written: {args.output}")
    print(f"Artifact SHA-256: {artifact_sha256}")
    print(f"Commit: {evidence['commit_sha']}")
    print(f"Recorded cells: {evidence['matrix_summary']['recorded_cells']}")
    print(f"Failed cells: {evidence['matrix_summary']['failed_cells']}")
    if evidence["matrix_summary"]["failed_cells"] and not args.allow_failed_cells:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
