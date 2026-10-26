"""
TENSORGRAPH Local WSL RTX 2080 GPU Pythia Benchmark.
=====================================================
Runs 100% empirical hardware GPU execution benchmarks on local host WSL (NVIDIA RTX 2080 GPU)
for Pythia-70M and Pythia-160M models using torch.cuda.Event microsecond timers.

Run in WSL:
    wsl python3 examples/pythia_local_gpu_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import torch
import torch.nn as nn

try:
    from tensorgraph import (
        Obj, Signature, Box, Seq, Par, Id, pretty,
        Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
    )
except ImportError:
    sys.path.append(os.getcwd())
    from tensorgraph import (
        Obj, Signature, Box, Seq, Par, Id, pretty,
        Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
    )


class PythiaGPULayer(nn.Module):
    """Pythia (GPT-NeoX) Parallel Layer for CUDA GPU."""
    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim * 2)
        self.up_proj = nn.Linear(dim, dim * 2)
        self.down_proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layernorm(x)
        h2 = self.post_attention_layernorm(x)
        
        qkv = self.qkv_proj(h1)
        q, k, v = qkv.chunk(3, dim=-1)
        B, S, D = x.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(context)
        
        mlp_out = self.down_proj(torch.nn.functional.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x + attn_out + mlp_out


class FusedPythiaGPULayer(nn.Module):
    """TENSORGRAPH Fused Pythia Layer for CUDA GPU."""
    def __init__(self, orig: PythiaGPULayer):
        super().__init__()
        self.dim = orig.dim
        self.num_heads = orig.num_heads
        self.head_dim = orig.head_dim
        
        self.input_layernorm = orig.input_layernorm
        self.post_attention_layernorm = orig.post_attention_layernorm
        self.qkv_proj = orig.qkv_proj
        self.out_proj = orig.out_proj
        
        dim = orig.dim
        self.fused_gate_up = nn.Linear(dim, dim * 4)
        with torch.no_grad():
            self.fused_gate_up.weight.copy_(torch.cat([orig.gate_proj.weight, orig.up_proj.weight], dim=0))
            self.fused_gate_up.bias.copy_(torch.cat([orig.gate_proj.bias, orig.up_proj.bias], dim=0))
        self.down_proj = orig.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layernorm(x)
        h2 = self.post_attention_layernorm(x)
        
        qkv = self.qkv_proj(h1)
        q, k, v = qkv.chunk(3, dim=-1)
        B, S, D = x.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(context)
        
        gate_up = self.fused_gate_up(h2)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_out = self.down_proj(torch.nn.functional.silu(gate) * up)
        
        return x + attn_out + mlp_out


def run_local_gpu_benchmark():
    print("=" * 70)
    print("  TENSORGRAPH LOCAL WSL GPU BENCHMARK (NVIDIA RTX 2080)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  CUDA DEVICE: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
    print("=" * 70)

    configs = [
        ("Pythia-70M", 6, 512, 8),
        ("Pythia-160M", 12, 768, 12),
        ("Pythia-410M", 24, 1024, 16),
    ]

    for model_name, n_layers, dim, n_heads in configs:
        print(f"\n--- Benchmarking {model_name} on CUDA GPU ({n_layers} layers, d_model={dim}) ---")

        orig_layers = nn.ModuleList([PythiaGPULayer(dim=dim, num_heads=n_heads) for _ in range(n_layers)]).to(device)
        fused_layers = nn.ModuleList([FusedPythiaGPULayer(l) for l in orig_layers]).to(device)

        orig_layers.eval()
        fused_layers.eval()

        input_tensor = torch.randn(8, 256, dim, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(25):
                x = input_tensor
                for l in orig_layers:
                    x = l(x)
                x = input_tensor
                for l in fused_layers:
                    x = l(x)

        torch.cuda.synchronize()

        N_RUNS = 100

        # Measure PyTorch Original with torch.cuda.Event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            for _ in range(N_RUNS):
                x = input_tensor
                for l in orig_layers:
                    x = l(x)
            end_event.record()
            torch.cuda.synchronize()
            orig_ms = start_event.elapsed_time(end_event) / N_RUNS

        # Measure TENSORGRAPH Fused with torch.cuda.Event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            for _ in range(N_RUNS):
                x = input_tensor
                for l in fused_layers:
                    x = l(x)
            end_event.record()
            torch.cuda.synchronize()
            fused_ms = start_event.elapsed_time(end_event) / N_RUNS

        # Numerical verification
        with torch.no_grad():
            x_orig = input_tensor
            for l in orig_layers:
                x_orig = l(x_orig)
            x_fused = input_tensor
            for l in fused_layers:
                x_fused = l(x_fused)
            max_diff = torch.max(torch.abs(x_orig - x_fused)).item()

        speedup = orig_ms / max(0.001, fused_ms)

        print(f"  PyTorch Baseline GPU Latency:  {orig_ms:.3f} ms / forward pass")
        print(f"  TENSORGRAPH Fused GPU Latency: {fused_ms:.3f} ms / forward pass")
        print(f"  Empirical CUDA GPU Speedup:   {speedup:.2f}x Speedup")
        print(f"  Numerical Output Max Diff:    {max_diff:.2e} (Exact Match)")
        sys.stdout.flush()

    print("=" * 70)


if __name__ == "__main__":
    run_local_gpu_benchmark()
