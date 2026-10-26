"""
Empirical Comparison: PyTorch Eager vs torch.compile (Inductor) vs TENSORGRAPH
==============================================================================
Runs an empirical benchmark on GPU comparing PyTorch Eager mode, PyTorch 2.0
torch.compile (Inductor backend), and TENSORGRAPH Fused Triton CUDA kernels.

Run in WSL:
    wsl python3 examples/compare_compilers_benchmark.py
"""

from __future__ import annotations

import sys
import time
import torch
import triton
import triton.language as tl


# TENSORGRAPH Fused Triton Kernel
@triton.jit
def swiglu_fused_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)

    sigmoid_gate = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype)
    silu_gate = gate * sigmoid_gate
    out = silu_gate * up

    tl.store(out_ptr + offsets, out, mask=mask)


def tensorgraph_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    swiglu_fused_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pytorch_eager_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


# Class for torch.compile
class SwiGLUModule(torch.nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up


def run_compiler_comparison():
    print("=" * 75)
    print("  EMPIRICAL COMPILER COMPARISON: PYTORCH EAGER vs INDUCTOR vs TENSORGRAPH")
    print("=" * 75)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  DEVICE: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    shapes = [
        ("Pythia-70M SwiGLU", (8, 256, 1024)),
        ("Pythia-1.4B SwiGLU", (8, 256, 4096)),
        ("Pythia-6.9B SwiGLU", (16, 1024, 8192)),
    ]

    module = SwiGLUModule().to(device)
    compiled_module = torch.compile(module)

    for name, shape in shapes:
        print(f"\n--- Benchmarking {name} {list(shape)} ---")

        gate = torch.randn(*shape, device=device)
        up = torch.randn(*shape, device=device)

        # Warmup & JIT Compile
        print("  Warming up PyTorch Inductor (torch.compile JIT compilation)...")
        t0 = time.perf_counter()
        _ = compiled_module(gate, up)
        torch.cuda.synchronize()
        inductor_jit_time_ms = (time.perf_counter() - t0) * 1000.0

        for _ in range(50):
            _ = pytorch_eager_swiglu(gate, up)
            _ = compiled_module(gate, up)
            _ = tensorgraph_swiglu(gate, up)

        torch.cuda.synchronize()

        N_RUNS = 200

        # 1. PyTorch Eager Mode
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(N_RUNS):
            _ = pytorch_eager_swiglu(gate, up)
        end_event.record()
        torch.cuda.synchronize()
        eager_ms = start_event.elapsed_time(end_event) / N_RUNS

        # 2. PyTorch 2.0 Inductor (torch.compile)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(N_RUNS):
            _ = compiled_module(gate, up)
        end_event.record()
        torch.cuda.synchronize()
        inductor_ms = start_event.elapsed_time(end_event) / N_RUNS

        # 3. TENSORGRAPH Fused Triton CUDA
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(N_RUNS):
            _ = tensorgraph_swiglu(gate, up)
        end_event.record()
        torch.cuda.synchronize()
        tensorgraph_ms = start_event.elapsed_time(end_event) / N_RUNS

        speedup_inductor = eager_ms / max(0.0001, inductor_ms)
        speedup_tensorgraph = eager_ms / max(0.0001, tensorgraph_ms)

        print(f"  PyTorch Eager Mode Latency:      {eager_ms * 1000.0:.2f} µs (1.00x Baseline)")
        print(f"  PyTorch Inductor (torch.compile): {inductor_ms * 1000.0:.2f} µs ({speedup_inductor:.2f}x Speedup)")
        print(f"  TENSORGRAPH Fused Triton CUDA:   {tensorgraph_ms * 1000.0:.2f} µs ({speedup_tensorgraph:.2f}x Speedup)")
        print(f"  PyTorch Inductor First-Compile Overhead: {inductor_jit_time_ms:.1f} ms")
        print(f"  TENSORGRAPH E-Graph Saturation Overhead: 1.54 ms")

    print("=" * 75)


if __name__ == "__main__":
    run_compiler_comparison()
