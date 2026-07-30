"""Produce admissible evidence for the verified elementwise compiler path.

The script exits nonzero unless it executes the exact generated Triton source on
CUDA, observes numerical agreement with PyTorch, resolves an exact clean Git
commit, and writes raw timing samples with environment metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tensorgraph.pipeline import compile_fx_elementwise, load_generated_kernel


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


def _cuda_sample_ms(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": p50,
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
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
    torch.set_grad_enabled(False)

    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.ReLU(),
        torch.nn.Sigmoid(),
    ).cuda()
    model.eval()

    compile_start = time.perf_counter_ns()
    artifact = compile_fx_elementwise(model)
    compile_total_ns = time.perf_counter_ns() - compile_start

    load_start = time.perf_counter_ns()
    generated = load_generated_kernel(artifact)
    source_load_ns = time.perf_counter_ns() - load_start

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    records: list[dict[str, Any]] = []

    for size_index, size in enumerate(args.sizes):
        x = torch.randn(size, device="cuda", dtype=torch.float32)

        first_start = time.perf_counter_ns()
        candidate = generated.run(x, block_size=args.block_size)
        torch.cuda.synchronize()
        first_execution_ns = time.perf_counter_ns() - first_start

        reference = model(x)
        torch.cuda.synchronize()
        max_abs_error = float((candidate - reference).abs().max().item()) if size else 0.0
        max_rel_error = (
            float(
                ((candidate - reference).abs() / reference.abs().clamp_min(args.atol))
                .max()
                .item()
            )
            if size
            else 0.0
        )
        numerical_pass = bool(
            torch.allclose(candidate, reference, rtol=args.rtol, atol=args.atol)
        )
        if not numerical_pass:
            raise RuntimeError(
                f"numerical gate failed for size={size}: "
                f"max_abs_error={max_abs_error}, max_rel_error={max_rel_error}"
            )

        for _ in range(args.warmup):
            model(x)
            generated.run(x, block_size=args.block_size)
        torch.cuda.synchronize()

        eager_samples = [
            _cuda_sample_ms(lambda value=x: model(value)) for _ in range(args.repetitions)
        ]
        generated_samples = [
            _cuda_sample_ms(
                lambda value=x, block_size=args.block_size: generated.run(
                    value, block_size=block_size
                )
            )
            for _ in range(args.repetitions)
        ]

        eager_summary = _summary(eager_samples)
        generated_summary = _summary(generated_samples)
        records.append(
            {
                "size": size,
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "contiguous": x.is_contiguous(),
                "first_execution_ns": first_execution_ns,
                "first_execution_includes_jit": size_index == 0,
                "numerical": {
                    "pass": numerical_pass,
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "max_abs_error": max_abs_error,
                    "max_rel_error": max_rel_error,
                },
                "pytorch_eager": {
                    "raw_ms": eager_samples,
                    "summary": eager_summary,
                },
                "tensorgraph_generated": {
                    "raw_ms": generated_samples,
                    "summary": generated_summary,
                },
                "median_ratio_eager_over_generated": (
                    eager_summary["median_ms"] / generated_summary["median_ms"]
                ),
            }
        )

    return {
        "schema": "tensorgraph.evidence.verified-elementwise.v1",
        "admissible": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "fyremael/TENSORGRAPH",
        "commit_sha": commit,
        "dirty_worktree": dirty,
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "block_size": args.block_size,
        "versions": _versions(),
        "hardware": {
            "gpu_index": device_index,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
            "cpu": platform.processor(),
        },
        "compiler": {
            "source_expr": artifact.source_pretty,
            "optimized_expr": artifact.optimized_pretty,
            "rewrite_summary": artifact.rewrite_summary,
            "generated_source_sha256": artifact.source_sha256,
            "generated_source": artifact.generated_source,
            "phase_ns": artifact.phase_ns,
            "compile_total_ns": compile_total_ns,
            "source_load_ns": source_load_ns,
        },
        "workloads": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 65_536, 1_048_576])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
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
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        print(f"EVIDENCE REJECTED: {exc}", file=sys.stderr)
        return 1

    print(f"Evidence written: {args.output}")
    print(f"Commit: {evidence['commit_sha']}")
    print(f"Generated source SHA-256: {evidence['compiler']['generated_source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
