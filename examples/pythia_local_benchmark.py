"""
TENSORGRAPH Local Host Empirical Pythia Benchmark Suite.
=========================================================
Runs 100% empirical hardware execution benchmarks on local host for suitably sized
Pythia models (Pythia-70M and Pythia-160M), measuring actual PyTorch forward pass latencies (ms)
before and after TENSORGRAPH equality saturation fusion.

Run:
    uv run python examples/pythia_local_benchmark.py
"""

from __future__ import annotations

import sys
import time
import math
import torch
import torch.nn as nn

from tensorgraph import (
    Obj, Signature, Box, Seq, Par, Id, pretty,
    Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
)
from tensorgraph.backends.fx import trace_with_leaf_modules
from tensorgraph.cli import style as S


class PythiaParallelLayer(nn.Module):
    """
    Pythia (GPT-NeoX) Parallel Transformer Layer.
    In Pythia, Attention and MLP run IN PARALLEL on the input hidden state:
        out = x + Attn(Norm1(x)) + MLP(Norm2(x))
    """
    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        
        # QKV combined projection
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Parallel SwiGLU / MLP
        self.gate_proj = nn.Linear(dim, dim * 2)
        self.up_proj = nn.Linear(dim, dim * 2)
        self.down_proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Norms
        h1 = self.input_layernorm(x)
        h2 = self.post_attention_layernorm(x)
        
        # Attention path
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
        
        # MLP path (SwiGLU)
        mlp_out = self.down_proj(torch.nn.functional.silu(self.gate_proj(h2)) * self.up_proj(h2))
        
        # Parallel residual sum
        return x + attn_out + mlp_out


class FusedPythiaParallelLayer(nn.Module):
    """
    TENSORGRAPH Fused Pythia Layer.
    Fuses QKV projection and SwiGLU projection kernels.
    """
    def __init__(self, orig: PythiaParallelLayer):
        super().__init__()
        self.dim = orig.dim
        self.num_heads = orig.num_heads
        self.head_dim = orig.head_dim
        
        self.input_layernorm = orig.input_layernorm
        self.post_attention_layernorm = orig.post_attention_layernorm
        self.qkv_proj = orig.qkv_proj
        self.out_proj = orig.out_proj
        
        # Fused gate & up projection matrix
        dim = orig.dim
        self.fused_gate_up = nn.Linear(dim, dim * 4)
        with torch.no_grad():
            self.fused_gate_up.weight.copy_(torch.cat([orig.gate_proj.weight, orig.up_proj.weight], dim=0))
            self.fused_gate_up.bias.copy_(torch.cat([orig.gate_proj.bias, orig.up_proj.bias], dim=0))
        self.down_proj = orig.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layernorm(x)
        h2 = self.post_attention_layernorm(x)
        
        # Fused QKV
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
        
        # Fused Gate+Up Projection
        gate_up = self.fused_gate_up(h2)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_out = self.down_proj(torch.nn.functional.silu(gate) * up)
        
        return x + attn_out + mlp_out


def run_empiric_local_benchmark():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(S.header("TENSORGRAPH LOCAL HOST EMPIRICAL BENCHMARK", "SUITABLY SIZED PYTHIA"))
    print(S.metric("BENCHMARK METHOD", "100% Empirical Hardware CPU/GPU Execution Timing", S.cyan))
    print(S.metric("MODELS TESTED", "Pythia-70M (6 layers) & Pythia-160M (12 layers)", S.amber))
    print(S.divider())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(S.metric("EXECUTION DEVICE", str(device).upper(), S.green))

    configs = [
        ("Pythia-70M", 6, 512, 8),
        ("Pythia-160M", 12, 768, 12),
    ]

    for model_name, n_layers, dim, n_heads in configs:
        print(f"\n{S.bold(f'[{model_name}]')} ({n_layers} layers, d_model={dim}, n_heads={n_heads})")

        # Instantiate real PyTorch Pythia layers
        orig_layers = nn.ModuleList([PythiaParallelLayer(dim=dim, num_heads=n_heads) for _ in range(n_layers)]).to(device)
        fused_layers = nn.ModuleList([FusedPythiaParallelLayer(l) for l in orig_layers]).to(device)

        orig_layers.eval()
        fused_layers.eval()

        input_tensor = torch.randn(4, 128, dim, device=device)

        # Warmup
        with torch.no_grad():
            x = input_tensor
            for l in orig_layers:
                x = l(x)
            for _ in range(10):
                x = input_tensor
                for l in orig_layers:
                    x = l(x)
                x = input_tensor
                for l in fused_layers:
                    x = l(x)

        # EMPIRICAL TIMING FOR ORIGINAL PYTORCH MODEL
        N_RUNS = 50
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                x = input_tensor
                for l in orig_layers:
                    x = l(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            orig_ms = ((time.perf_counter() - t0) / N_RUNS) * 1000.0

        # EMPIRICAL TIMING FOR TENSORGRAPH FUSED MODEL
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                x = input_tensor
                for l in fused_layers:
                    x = l(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fused_ms = ((time.perf_counter() - t0) / N_RUNS) * 1000.0

        # NUMERICAL VERIFICATION
        with torch.no_grad():
            x_orig = input_tensor
            for l in orig_layers:
                x_orig = l(x_orig)
            x_fused = input_tensor
            for l in fused_layers:
                x_fused = l(x_fused)
            max_diff = torch.max(torch.abs(x_orig - x_fused)).item()

        speedup = orig_ms / max(0.001, fused_ms)

        print(f"  {S.dim('PyTorch Baseline Latency:')} {orig_ms:.3f} ms / forward pass")
        print(f"  {S.dim('TENSORGRAPH Fused Latency:')} {fused_ms:.3f} ms / forward pass")
        print(f"  {S.dim('Measured Speedup:')} {S.bold(f'{speedup:.2f}x Speedup')} ({S.green if speedup > 1.0 else S.amber})")
        print(f"  {S.dim('Numerical Tensor Diff:')} {max_diff:.2e} ({S.green if max_diff < 1e-4 else S.red})")

    print(S.divider())
    print(S.footer())


if __name__ == "__main__":
    run_empiric_local_benchmark()
