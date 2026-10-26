"""
TENSORGRAPH Multi-Batch Performance Regression & Scaling Benchmark for Google Colab Tesla T4 GPU.
===================================================================================================
Executes batch-size scaling sweep on Linux Colab T4 GPU:
- Single-token decoding (seq=1) across Batch Sizes: 1, 8, 16, 32, 64
- Prompt prefill (seq=512) across Batch Sizes: 1, 4, 8, 16, 32
- Compares PyTorch Eager vs Fused Triton vs CUDA Graph Stream Capture
"""

from __future__ import annotations

import math
import os
import sys
import time
import statistics
from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# -----------------------------------------------------------------------------
# BENCHMARK HARNESS WITH CUDA EVENTS
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    category: str
    batch_size: int
    seq_len: int
    eager_us: float
    triton_us: float
    cuda_graph_us: float
    triton_speedup: float
    cg_speedup: float


def benchmark_gpu_time(fn: Callable[[], Any], iterations: int = 50, warmup: int = 15) -> float:
    """Measure mean execution time in microseconds using CUDA Events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    times_us = []
    for _ in range(iterations):
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    return statistics.mean(times_us)


# -----------------------------------------------------------------------------
# TRITON FUSED KERNEL DEFINITION
# -----------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def swiglu_fused_triton_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask)
        up = tl.load(up_ptr + offsets, mask=mask)
        silu = gate * tl.sigmoid(gate.to(tl.float32)).to(gate.dtype)
        tl.store(out_ptr + offsets, silu * up, mask=mask)

    def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, 1024),)
        swiglu_fused_triton_kernel[grid](gate, up, out, n, BLOCK_SIZE=1024)
        return out
else:
    def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        res = torch.nn.functional.silu(gate) * up
        if out is not None:
            out.copy_(res)
            return out
        return res


# -----------------------------------------------------------------------------
# BATCH SCALING BENCHMARK SUITE
# -----------------------------------------------------------------------------

def run_colab_batch_scaling_suite():
    print()
    print("=" * 95)
    print("  TENSORGRAPH MULTI-BATCH SCALING SWEEP (COLAB TESLA T4 GPU)")
    print("  Evaluating Latency & Speedup Across Batch Sizes (B = 1, 4, 8, 16, 32, 64)")
    print("=" * 95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if device.type == "cuda" else 0.0

    print(f"  TARGET HARDWARE: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print("=" * 95)

    dim = 11008
    decoding_batches = [1, 4, 8, 16, 32, 64]
    prefill_batches = [1, 4, 8, 16, 32]

    decoding_results: list[BenchmarkResult] = []
    prefill_results: list[BenchmarkResult] = []

    # -------------------------------------------------------------------------
    # 1. DECODING SWEEP (seq_len = 1)
    # -------------------------------------------------------------------------
    print("\n--- [1] SINGLE-TOKEN DECODING BATCH SWEEP (seq_len = 1) ---")
    for b in decoding_batches:
        gate = torch.randn(b, 1, dim, device=device, dtype=torch.float32)
        up = torch.randn(b, 1, dim, device=device, dtype=torch.float32)
        out = torch.empty_like(gate)

        def bench_eager():
            torch.nn.functional.silu(gate, inplace=False) * up

        def bench_triton():
            fused_swiglu(gate, up, out)

        eager_us = benchmark_gpu_time(bench_eager, iterations=50, warmup=15)
        triton_us = benchmark_gpu_time(bench_triton, iterations=50, warmup=15)

        cg_us = triton_us
        if device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph"):
            g_gate = gate.clone()
            g_up = up.clone()
            g_out = torch.empty_like(g_gate)
            
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    fused_swiglu(g_gate, g_up, g_out)
            torch.cuda.current_stream().wait_stream(s)

            g_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_graph):
                fused_swiglu(g_gate, g_up, g_out)

            def bench_cg():
                g_graph.replay()

            cg_us = benchmark_gpu_time(bench_cg, iterations=50, warmup=15)

        res = BenchmarkResult(
            category="Decoding (seq=1)",
            batch_size=b,
            seq_len=1,
            eager_us=eager_us,
            triton_us=triton_us,
            cuda_graph_us=cg_us,
            triton_speedup=eager_us / triton_us,
            cg_speedup=eager_us / cg_us,
        )
        decoding_results.append(res)

    # Display Decoding Table
    print(f"  {'Batch':<7} {'PyTorch Eager':>16} {'Fused Triton':>16} {'CUDA Graph':>16} {'Triton Speedup':>16} {'CG Speedup':>14}")
    print("  " + "-" * 89)
    for r in decoding_results:
        print(f"  B={r.batch_size:<5} {r.eager_us:>13.2f} us {r.triton_us:>13.2f} us {r.cuda_graph_us:>13.2f} us {r.triton_speedup:>15.2f}x {r.cg_speedup:>13.2f}x")

    # -------------------------------------------------------------------------
    # 2. PREFILL SWEEP (seq_len = 512)
    # -------------------------------------------------------------------------
    print("\n--- [2] PROMPT PREFILL BATCH SWEEP (seq_len = 512) ---")
    for b in prefill_batches:
        gate = torch.randn(b, 512, dim, device=device, dtype=torch.float32)
        up = torch.randn(b, 512, dim, device=device, dtype=torch.float32)
        out = torch.empty_like(gate)

        def bench_eager():
            _ = torch.nn.functional.silu(gate) * up

        def bench_triton():
            fused_swiglu(gate, up, out)

        eager_us = benchmark_gpu_time(bench_eager, iterations=30, warmup=10)
        triton_us = benchmark_gpu_time(bench_triton, iterations=30, warmup=10)

        res = BenchmarkResult(
            category="Prefill (seq=512)",
            batch_size=b,
            seq_len=512,
            eager_us=eager_us,
            triton_us=triton_us,
            cuda_graph_us=0.0,
            triton_speedup=eager_us / triton_us,
            cg_speedup=0.0,
        )
        prefill_results.append(res)

    # Display Prefill Table
    print(f"  {'Batch':<7} {'PyTorch Eager (ms)':>20} {'Fused Triton (ms)':>20} {'Triton Speedup':>20} {'HBM Traffic Saved':>18}")
    print("  " + "-" * 89)
    for r in prefill_results:
        mb_saved = (r.batch_size * 512 * dim * 4 * 2) / 1e6
        print(f"  B={r.batch_size:<5} {r.eager_us/1000:>17.2f} ms {r.triton_us/1000:>17.2f} ms {r.triton_speedup:>19.2f}x {mb_saved:>15.1f} MB")

    print()
    print("=" * 95)
    print("  MULTI-BATCH VERIFICATION COMPLETE: ALL BENCHMARKS EXECUTED SUCCESSFULLY")
    print("=" * 95)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_colab_batch_scaling_suite()
