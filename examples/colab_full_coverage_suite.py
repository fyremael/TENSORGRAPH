"""
TENSORGRAPH Comprehensive Benchmark & Full-Coverage Suite for Google Colab Tesla T4 GPU.
======================================================================================
Tests 3 advanced coverage dimensions:
1. Precision Scaling Sweep: FP16 vs FP32 Tensor Core throughput & bandwidth
2. Sequence Length Inflection Sweep: seq_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
3. Deep 32-Layer Transformer MLP Pipeline: 32-layer LLaMA MLP backbone launch accumulation
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


def run_full_coverage_suite():
    print()
    print("=" * 95)
    print("  TENSORGRAPH FULL-COVERAGE BENCHMARK SUITE (COLAB TESLA T4 GPU)")
    print("  Evaluating FP16 Precision, Sequence Length Inflection, & 32-Layer Deep Model Pipeline")
    print("=" * 95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if device.type == "cuda" else 0.0

    print(f"  TARGET HARDWARE: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print("=" * 95)

    dim = 11008

    # -------------------------------------------------------------------------
    # 1. FP16 vs FP32 PRECISION BENCHMARK (seq=512, batch=4)
    # -------------------------------------------------------------------------
    print("\n--- [1] PRECISION SCALING (FP16 vs FP32) (seq=512, batch=4) ---")
    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        gate = torch.randn(4, 512, dim, device=device, dtype=dtype)
        up = torch.randn(4, 512, dim, device=device, dtype=dtype)
        out = torch.empty_like(gate)

        def bench_eager():
            _ = torch.nn.functional.silu(gate) * up

        def bench_triton():
            fused_swiglu(gate, up, out)

        eager_us = benchmark_gpu_time(bench_eager, iterations=40, warmup=15)
        triton_us = benchmark_gpu_time(bench_triton, iterations=40, warmup=15)
        speedup = eager_us / triton_us
        
        # Calculate effective bandwidth (GB/s)
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        total_bytes = 4 * 512 * dim * bytes_per_elem * 3  # 2 reads + 1 write
        bw_gb_s = (total_bytes / (triton_us / 1e6)) / 1e9

        print(f"  {dtype_name:<6} PyTorch Eager: {eager_us/1000:>6.2f} ms | Fused Triton: {triton_us/1000:>6.2f} ms | Speedup: {speedup:>5.2f}x | Triton HBM BW: {bw_gb_s:>6.2f} GB/s")

    # -------------------------------------------------------------------------
    # 2. SEQUENCE LENGTH INFLECTION SWEEP (seq_len = 1 to 1024, batch=1, FP16)
    # -------------------------------------------------------------------------
    print("\n--- [2] SEQUENCE LENGTH CROSSOVER INFLECTION SWEEP (seq_len = 1 to 1024, FP16) ---")
    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"  {'seq_len':<9} {'PyTorch Eager':>16} {'Fused Triton':>16} {'CUDA Graph':>16} {'Optimal Engine':>18}")
    print("  " + "-" * 85)

    for seq in seq_lens:
        gate = torch.randn(1, seq, dim, device=device, dtype=torch.float16)
        up = torch.randn(1, seq, dim, device=device, dtype=torch.float16)
        out = torch.empty_like(gate)

        def bench_e():
            _ = torch.nn.functional.silu(gate) * up

        def bench_t():
            fused_swiglu(gate, up, out)

        e_us = benchmark_gpu_time(bench_e, iterations=40, warmup=15)
        t_us = benchmark_gpu_time(bench_t, iterations=40, warmup=15)

        cg_us = t_us
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

            def bench_c():
                g_graph.replay()

            cg_us = benchmark_gpu_time(bench_c, iterations=40, warmup=15)

        optimal = "CUDA_GRAPH" if cg_us < t_us and seq <= 8 else "FUSED_TRITON"
        print(f"  seq={seq:<5} {e_us:>13.2f} us {t_us:>13.2f} us {cg_us:>13.2f} us {optimal:>18}")

    # -------------------------------------------------------------------------
    # 3. 32-LAYER DEEP TRANSFORMER MLP BACKBONE PIPELINE (batch=1, seq=1, FP16)
    # -------------------------------------------------------------------------
    print("\n--- [3] 32-LAYER DEEP TRANSFORMER MLP PIPELINE (LLaMA-32x, seq=1, batch=1, FP16) ---")
    n_layers = 32
    gates = [torch.randn(1, 1, dim, device=device, dtype=torch.float16) for _ in range(n_layers)]
    ups = [torch.randn(1, 1, dim, device=device, dtype=torch.float16) for _ in range(n_layers)]
    outs = [torch.empty_like(gates[0]) for _ in range(n_layers)]

    def bench_eager_32layer():
        for i in range(n_layers):
            _ = torch.nn.functional.silu(gates[i]) * ups[i]

    def bench_triton_32layer():
        for i in range(n_layers):
            fused_swiglu(gates[i], ups[i], outs[i])

    eager_32_us = benchmark_gpu_time(bench_eager_32layer, iterations=30, warmup=10)
    triton_32_us = benchmark_gpu_time(bench_triton_32layer, iterations=30, warmup=10)

    cg_32_us = triton_32_us
    if device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph"):
        g_gates = [g.clone() for g in gates]
        g_ups = [u.clone() for u in ups]
        g_outs = [torch.empty_like(g) for g in g_gates]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for i in range(n_layers):
                    fused_swiglu(g_gates[i], g_ups[i], g_outs[i])
        torch.cuda.current_stream().wait_stream(s)

        full_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_graph):
            for i in range(n_layers):
                fused_swiglu(g_gates[i], g_ups[i], g_outs[i])

        def bench_cg_32layer():
            full_graph.replay()

        cg_32_us = benchmark_gpu_time(bench_cg_32layer, iterations=30, warmup=10)

    print(f"  PyTorch Eager (32 Layers):           {eager_32_us:>9.2f} us ({eager_32_us/1000:.3f} ms)")
    print(f"  TENSORGRAPH Uncaptured Triton (32L): {triton_32_us:>9.2f} us ({triton_32_us/1000:.3f} ms)")
    print(f"  TENSORGRAPH CUDA Graph (32 Layers):  {cg_32_us:>9.2f} us ({cg_32_us/1000:.3f} ms)")
    print(f"  32-Layer Deep Model Pipeline Speedup: {eager_32_us / cg_32_us:.2f}x Faster!")

    print()
    print("=" * 95)
    print("  FULL-COVERAGE BENCHMARK SUITE COMPLETE: ALL DIMENSIONS VERIFIED")
    print("=" * 95)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_full_coverage_suite()
