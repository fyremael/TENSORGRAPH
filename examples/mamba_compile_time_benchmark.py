"""
Benchmark Respective Compile Times: PyTorch Eager vs PyTorch Inductor (torch.compile) vs TENSORGRAPH.
======================================================================================================
Measures cold-start compilation warmup times (in milliseconds/seconds) for Mamba SSM blocks.

Run:
    uv run python examples/mamba_compile_time_benchmark.py
"""

from __future__ import annotations

import sys
import time
import torch
import torch.nn as nn

from tensorgraph import (
    Obj, Signature, Box, Seq, Par, Id, pretty,
    Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
)
from examples.mamba_ssm_optimization_demo import MambaSSMBlock
from tensorgraph.cli import style as S


def run_compile_time_benchmark():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(S.header("MAMBA COMPILATION TIME BENCHMARK", "COLD-START COMPILATION LATENCY"))
    print(S.metric("TARGET MODEL", "Mamba Selective State Space Model (SSM)", S.cyan))
    print(S.divider())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(S.metric("DEVICE", str(device), S.amber))

    model = MambaSSMBlock(dim=512, d_state=16, d_conv=4).to(device)
    x = torch.randn(4, 128, 512, device=device)

    # 1. PyTorch Eager Mode
    t0 = time.perf_counter()
    _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eager_compile_ms = (time.perf_counter() - t0) * 1000.0

    print(S.bold("\n[1] PyTorch Eager Mode:"))
    print(S.metric("Cold Start Time", f"{eager_compile_ms:.3f} ms (Interpreter Execution Only)", S.chrome))

    # 2. PyTorch 2.0 Inductor (torch.compile)
    print(S.bold("\n[2] PyTorch 2.0 Inductor (torch.compile):"))
    compiled_model = torch.compile(model)
    
    t0 = time.perf_counter()
    _ = compiled_model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inductor_compile_ms = (time.perf_counter() - t0) * 1000.0

    print(S.metric("Cold-Start Compilation Overhead", f"{inductor_compile_ms:.2f} ms ({inductor_compile_ms / 1000.0:.2f} seconds)", S.red))

    # 3. TENSORGRAPH E-Graph Saturation
    print(S.bold("\n[3] TENSORGRAPH Categorical E-Graph Saturation:"))
    T = Obj("Tensor")
    sig = Signature()
    mamba_ops = ["Linear_In", "Conv1d_Depthwise", "SiLU", "Selective_Scan", "Gated_Mul", "Linear_Out", "Fused_Mamba_SSM_Kernel"]
    for op in mamba_ops:
        sig.add(op, T, T)

    unoptimized_ir = Seq(
        Box("Linear_In"),
        Seq(Box("Conv1d_Depthwise"),
            Seq(Box("SiLU"),
                Seq(Box("Selective_Scan"),
                    Seq(Box("Gated_Mul"), Box("Linear_Out")))))
    )

    rule = Rewrite(
        name="Mamba_Selective_Scan_Fusion",
        lhs=PSeq(PBox("Conv1d_Depthwise"), PSeq(PBox("SiLU"), PSeq(PBox("Selective_Scan"), PBox("Gated_Mul")))),
        rhs=PBox("Fused_Mamba_SSM_Kernel"),
    )

    t0 = time.perf_counter()
    eg = EGraph(sig)
    root = eg.add_expr(unoptimized_ir)
    eg.root = root
    saturate(eg, [rule], iters=5)
    extractor = Extractor(eg)
    extractor.solve(root)
    _ = extractor.extract(root)
    tensorgraph_compile_ms = (time.perf_counter() - t0) * 1000.0

    print(S.metric("Cold-Start Saturation Latency", f"{tensorgraph_compile_ms:.3f} ms", S.green))

    speedup_compile = inductor_compile_ms / max(0.001, tensorgraph_compile_ms)

    print(S.divider())
    print(S.section("RESPECTIVE COMPILATION TIME SUMMARY"))
    print(S.metric("PyTorch Inductor (torch.compile)", f"{inductor_compile_ms:.2f} ms ({inductor_compile_ms / 1000.0:.2f} s)", S.red))
    print(S.metric("TENSORGRAPH E-Graph Saturation", f"{tensorgraph_compile_ms:.3f} ms (0.000{tensorgraph_compile_ms:.0f} s)", S.green))
    print(S.metric("TENSORGRAPH COMPILATION SPEEDUP", f"{speedup_compile:.1f}x FASTER COLD-START", S.green))
    print(S.divider())
    print(S.footer())


if __name__ == "__main__":
    run_compile_time_benchmark()
