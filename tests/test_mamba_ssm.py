"""
Unit tests for Mamba Selective State Space Model (SSM) optimization in TENSORGRAPH.
"""

from __future__ import annotations

import torch
import pytest

from tensorgraph import (
    Obj, Signature, Box, Seq, Par, Id, pretty,
    Rewrite, PSeq, PBox, PVar, EGraph, saturate, Extractor
)
from tensorgraph.egraph import ENode
from examples.mamba_ssm_optimization_demo import MambaSSMBlock, FusedMambaSSMBlock


def test_mamba_ssm_egraph_fusion():
    T = Obj("Tensor")
    sig = Signature()
    mamba_ops = ["Linear_In", "Conv1d_Depthwise", "SiLU", "Selective_Scan", "Gated_Mul", "Linear_Out", "Fused_Scan", "Fused_Mamba_SSM_Kernel"]
    for op in mamba_ops:
        sig.add(op, T, T)

    unoptimized_mamba_ir = Seq(
        Box("Linear_In"),
        Seq(Box("Conv1d_Depthwise"),
            Seq(Box("SiLU"),
                Seq(Box("Selective_Scan"),
                    Seq(Box("Gated_Mul"), Box("Linear_Out")))))
    )

    def fuse_scan_fn(eg, root, env, oenv):
        rest_id = env["rest"]
        fused_box_id = eg.add_expr(Box("Fused_Scan"))
        return eg.add_enode(ENode("Seq", (), (fused_box_id, rest_id)), (T, T))

    def fuse_conv_fn(eg, root, env, oenv):
        rest_id = env["rest"]
        fused_box_id = eg.add_expr(Box("Fused_Mamba_SSM_Kernel"))
        return eg.add_enode(ENode("Seq", (), (fused_box_id, rest_id)), (T, T))

    rule1 = Rewrite("SiLU_Scan_Fusion", PSeq(PBox("SiLU"), PSeq(PBox("Selective_Scan"), PSeq(PBox("Gated_Mul"), PVar("rest")))), fuse_scan_fn)
    rule2 = Rewrite("Conv_Scan_Fusion", PSeq(PBox("Conv1d_Depthwise"), PSeq(PBox("Fused_Scan"), PVar("rest"))), fuse_conv_fn)

    eg = EGraph(sig)
    root = eg.add_expr(unoptimized_mamba_ir)
    eg.root = root

    saturate(eg, [rule1, rule2], iters=10)

    extractor = Extractor(eg)
    extractor.solve(root)
    optimized_ir = extractor.extract(root)

    assert "Fused_Mamba_SSM_Kernel" in pretty(optimized_ir)


def test_mamba_ssm_numerical_precision():
    torch.manual_seed(42)
    model_orig = MambaSSMBlock(dim=128, d_state=16, d_conv=4)
    model_fused = FusedMambaSSMBlock(model_orig)

    model_orig.eval()
    model_fused.eval()

    sample_x = torch.randn(2, 32, 128)
    with torch.no_grad():
        out_orig = model_orig(sample_x)
        out_fused = model_fused(sample_x)

    max_diff = torch.max(torch.abs(out_orig - out_fused)).item()
    assert max_diff < 1e-4
    assert torch.allclose(out_orig, out_fused, atol=1e-4)
