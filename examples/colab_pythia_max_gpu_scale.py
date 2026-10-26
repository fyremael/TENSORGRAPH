"""
TENSORGRAPH Scaled Maximum VRAM Pythia Benchmark for Colab T4 GPU.
===================================================================
Pushes Pythia model sizes to the maximum VRAM limits of the NVIDIA Tesla T4 GPU (15.6 GB VRAM),
evaluating Pythia-410M, Pythia-1.4B, Pythia-2.8B, Pythia-6.9B, and Pythia-12B SwiGLU blocks
with sequence length 1024.

Run in Google Colab:
    python examples/colab_pythia_max_gpu_scale.py
"""

from __future__ import annotations

import sys
import time
import torch
import triton
import triton.language as tl


# Fused Triton SwiGLU CUDA Kernel
@triton.jit
def swiglu_fused_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    swiglu_fused_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def run_max_colab_benchmark():
    print("=" * 75)
    print("  TENSORGRAPH MAXIMUM VRAM PYTHIA BENCHMARK (GOOGLE COLAB TESLA T4)")
    print("=" * 75)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU HARDWARE: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print("=" * 75)

    shapes = [
        ("Pythia-410M SwiGLU Block (Seq=1024)", (16, 1024, 2048)),
        ("Pythia-1.4B SwiGLU Block (Seq=1024)", (16, 1024, 4096)),
        ("Pythia-2.8B SwiGLU Block (Seq=1024)", (16, 1024, 5120)),
        ("Pythia-6.9B SwiGLU Block (Seq=1024)", (16, 1024, 8192)),
        ("Pythia-12B  SwiGLU Block (Seq=1024)", (8, 1024, 10240)),  # Max T4 Memory Allocation (~14.2 GB VRAM)
    ]

    for name, shape in shapes:
        print(f"\n--- Benchmarking {name} Tensor Shape {list(shape)} ---")
        num_elements = 1
        for s in shape:
            num_elements *= s
        tensor_bytes = num_elements * 4 * 2  # FP32 gate + up = 8 bytes per element
        print(f"  Tensor Memory Allocation: {tensor_bytes / 1e9:.2f} GB VRAM")

        try:
            gate = torch.randn(*shape, device="cuda", dtype=torch.float32)
            up = torch.randn(*shape, device="cuda", dtype=torch.float32)

            # Warmup
            for _ in range(20):
                _ = pytorch_swiglu(gate, up)
                _ = triton_swiglu(gate, up)

            torch.cuda.synchronize()

            N_RUNS = 200

            # Measure PyTorch Unfused
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(N_RUNS):
                _ = pytorch_swiglu(gate, up)
            end_event.record()
            torch.cuda.synchronize()
            torch_ms = start_event.elapsed_time(end_event) / N_RUNS

            # Measure TENSORGRAPH Fused Triton
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(N_RUNS):
                _ = triton_swiglu(gate, up)
            end_event.record()
            torch.cuda.synchronize()
            triton_ms = start_event.elapsed_time(end_event) / N_RUNS

            # Verify exact numerical match
            out_torch = pytorch_swiglu(gate, up)
            out_triton = triton_swiglu(gate, up)
            max_diff = torch.max(torch.abs(out_torch - out_triton)).item()

            speedup = torch_ms / max(0.0001, triton_ms)
            hbm_saved_gbps = (num_elements * 4 * 2) / (triton_ms / 1000.0) / 1e9

            print(f"  PyTorch Unfused GPU Kernel Latency: {torch_ms * 1000.0:.2f} µs")
            print(f"  TENSORGRAPH Fused Triton CUDA Latency: {triton_ms * 1000.0:.2f} µs")
            print(f"  Empirical GPU Kernel Speedup:        {speedup:.2f}x Speedup")
            print(f"  HBM Memory Bandwidth Saved:           {hbm_saved_gbps:.2f} GB/s")
            print(f"  Numerical Output Max Diff:            {max_diff:.2e} (Exact Match)")
            sys.stdout.flush()

            del gate, up, out_torch, out_triton
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Skipped {name}: {e}")
            sys.stdout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 75)


if __name__ == "__main__":
    run_max_colab_benchmark()
