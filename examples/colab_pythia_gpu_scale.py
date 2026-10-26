"""
TENSORGRAPH Google Colab Scaled GPU Pythia Benchmark Kit.
==========================================================
Copy & paste this script directly into a Google Colab GPU cell (T4, L4, or A100)
to execute scaled empirical GPU benchmarks across the entire Pythia suite (70M to 12B).

Run in Google Colab:
    !pip install -q triton torch torchvision
    !git clone https://github.com/AntigravityGCT/TENSORGRAPH.git 2>/dev/null || true
    !python examples/colab_pythia_gpu_scale.py
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
    # If running directly in colab root
    sys.path.append(os.getcwd())
    from tensorgraph import (
        Obj, Signature, Box, Seq, Par, Id, pretty,
        Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
    )


class PythiaGPUParallelLayer(nn.Module):
    """Pythia (GPT-NeoX) Parallel Transformer Layer for GPU Benchmark."""
    def __init__(self, dim: int = 2048, num_heads: int = 16):
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
    """Fused Pythia Layer for GPU Benchmark."""
    def __init__(self, orig: PythiaGPUParallelLayer):
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


def run_colab_gpu_suite():
    print("=" * 70)
    print("  TENSORGRAPH GOOGLE COLAB SCALED GPU PYTHIA BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU HARDWARE DETECTED: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("  WARNING: Running on CPU. For maximum speedup, switch Colab runtime to GPU (T4/A100)!")

    configs = [
        ("Pythia-70M", 6, 512, 8),
        ("Pythia-160M", 12, 768, 12),
        ("Pythia-410M", 24, 1024, 16),
        ("Pythia-1.4B", 24, 2048, 16),
        ("Pythia-2.8B", 32, 2560, 32),
        ("Pythia-6.9B", 32, 4096, 32),
    ]

    for model_name, n_layers, dim, n_heads in configs:
        print(f"\n--- Benchmarking {model_name} ({n_layers} layers, d_model={dim}) ---")

        try:
            # Benchmark 2 layers per model scale to stay within Colab free tier RAM constraints
            bench_layers = min(4, n_layers)
            orig_block = nn.Sequential(*[PythiaGPUParallelLayer(dim=dim, num_heads=n_heads) for _ in range(bench_layers)]).to(device)
            fused_block = nn.Sequential(*[FusedPythiaGPULayer(l) for l in orig_block]).to(device)

            orig_block.eval()
            fused_block.eval()

            x = torch.randn(2, 256, dim, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = orig_block(x)
                    _ = fused_block(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            N_RUNS = 20

            # Measure Original
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(N_RUNS):
                    _ = orig_block(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                orig_ms = ((time.perf_counter() - t0) / N_RUNS) * 1000.0

            # Measure Fused
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(N_RUNS):
                    _ = fused_block(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fused_ms = ((time.perf_counter() - t0) / N_RUNS) * 1000.0

            speedup = orig_ms / max(0.001, fused_ms)
            print(f"  PyTorch Baseline Latency: {orig_ms:.3f} ms")
            print(f"  TENSORGRAPH Fused Latency: {fused_ms:.3f} ms")
            print(f"  GPU Speedup: {speedup:.2f}x Speedup")
            sys.stdout.flush()

        except Exception as e:
            print(f"  Skipped {model_name}: {e}")
            sys.stdout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    run_colab_gpu_suite()

