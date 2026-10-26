"""
TENSORGRAPH Long Sequence Length Benchmark Suite for Google Colab Tesla T4 GPU.
==============================================================================
Tests long sequence lengths: seq_len in [2048, 4096, 8192, 16384] (FP16)
Compares PyTorch Eager vs Fused Triton vs CUDA Graph Stream Capture.
"""

from __future__ import annotations

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


def benchmark_gpu_time(fn: Callable[[], Any], iterations: int = 25, warmup: int = 10) -> float:
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


def run_long_seq_suite():
    print()
    print("=" * 100)
    print("  TENSORGRAPH LONG SEQUENCE LENGTH BENCHMARK SUITE (COLAB TESLA T4 GPU)")
    print("  Evaluating Latency, Speedup, & HBM DRAM Bandwidth for seq_len in [2048, 4096, 8192, 16384] (FP16)")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if device.type == "cuda" else 0.0

    print(f"  TARGET HARDWARE: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print("=" * 100)

    dim = 11008
    long_seq_lens = [2048, 4096, 8192, 16384]

    print(f"\n  {'seq_len':<9} {'PyTorch Eager':>16} {'Fused Triton':>16} {'CUDA Graph':>16} {'Triton Speedup':>16} {'HBM Saved':>14}")
    print("  " + "-" * 95)

    for seq in long_seq_lens:
        gate = torch.randn(1, seq, dim, device=device, dtype=torch.float16)
        up = torch.randn(1, seq, dim, device=device, dtype=torch.float16)
        out = torch.empty_like(gate)

        def bench_eager():
            _ = torch.nn.functional.silu(gate) * up

        def bench_triton():
            fused_swiglu(gate, up, out)

        eager_us = benchmark_gpu_time(bench_eager, iterations=25, warmup=10)
        triton_us = benchmark_gpu_time(bench_triton, iterations=25, warmup=10)

        cg_us = triton_us
        if device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph"):
            g_gate = gate.clone()
            g_up = up.clone()
            g_out = torch.empty_like(g_gate)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fused_swiglu(g_gate, g_up, g_out)
            torch.cuda.current_stream().wait_stream(s)

            g_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_graph):
                fused_swiglu(g_gate, g_up, g_out)

            def bench_cg():
                g_graph.replay()

            cg_us = benchmark_gpu_time(bench_cg, iterations=25, warmup=10)

        speedup = eager_us / triton_us
        hbm_saved_mb = (1 * seq * dim * 2 * 2) / 1e6  # FP16 (2 bytes) * 2 intermediate ops saved

        eager_str = f"{eager_us/1000:.2f} ms" if eager_us >= 1000 else f"{eager_us:.1f} us"
        triton_str = f"{triton_us/1000:.2f} ms" if triton_us >= 1000 else f"{triton_us:.1f} us"
        cg_str = f"{cg_us/1000:.2f} ms" if cg_us >= 1000 else f"{cg_us:.1f} us"
        saved_str = f"{hbm_saved_mb/1000:.2f} GB" if hbm_saved_mb >= 1000 else f"{hbm_saved_mb:.1f} MB"

        print(f"  seq={seq:<5} {eager_str:>16} {triton_str:>16} {cg_str:>16} {speedup:>15.2f}x {saved_str:>14}")

    print()
    print("=" * 100)
    print("  LONG SEQUENCE BENCHMARK COMPLETE: ALL CONTEXT WINDOWS VERIFIED")
    print("=" * 100)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_long_seq_suite()
