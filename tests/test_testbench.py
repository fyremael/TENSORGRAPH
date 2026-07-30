"""
Tests for TENSORGRAPH Testbench Validation Suite — Scaled Industrial Workloads.
"""

import pytest
from pathlib import Path
import torch

from tensorgraph.testbench.workloads import (
    get_all_workloads,
    build_transformer_attention_workload,
    build_llama_decoder_workload,
    build_resnet_conv_bn_workload,
    build_convnext_block_workload,
    build_lora_chain_workload,
    build_control_flow_licm_workload,
    build_triton_reduction_workload,
    build_sharded_egraph_workload,
    build_egraph_stress_test_workload,
)
from tensorgraph.testbench.evaluator import Evaluator
from tensorgraph.testbench.runner import TestbenchRunner
from tensorgraph.testbench.fx_roundtrip import FXRoundtripOptimizer
from tensorgraph.rewrite import Rewrite, PSeq, PBox


def test_workload_instantiation():
    workloads = get_all_workloads()
    assert len(workloads) >= 9, f"Expected at least 9 testbench workloads, got {len(workloads)}"
    names = {w.name for w in workloads}
    assert "transformer_attention_qkv_fusion" in names
    assert "llama_decoder_block_swiglu_fusion" in names
    assert "resnet_conv_bn_relu_fusion" in names
    assert "convnext_block_fusion" in names
    assert "lora_adapter_chain_fusion" in names
    assert "control_flow_licm_hoist" in names
    assert "triton_reduction_codegen" in names
    assert "distributed_sharded_egraph_merge" in names
    assert "egraph_stress_test_500_nodes" in names


def test_evaluator_llama_decoder_workload():
    wl = build_llama_decoder_workload()
    evaluator = Evaluator(verify_correctness=True, iterations=5)
    res = evaluator.evaluate_workload(wl)

    assert res.workload_name == wl.name
    assert res.cost_after <= res.cost_before
    assert res.cost_reduction_pct >= 0.0
    assert res.correctness_passed is True


def test_evaluator_convnext_workload():
    wl = build_convnext_block_workload()
    evaluator = Evaluator(verify_correctness=True, iterations=5)
    res = evaluator.evaluate_workload(wl)

    assert res.workload_name == wl.name
    assert res.cost_after <= res.cost_before
    assert res.cost_reduction_pct >= 0.0
    assert res.correctness_passed is True


def test_evaluator_500_node_stress_test():
    wl = build_egraph_stress_test_workload(500)
    evaluator = Evaluator(verify_correctness=False, iterations=5)
    res = evaluator.evaluate_workload(wl)

    assert res.workload_name == wl.name
    assert res.peak_nodes > 100
    assert res.saturation_time_ms >= 0.0


def test_fx_roundtrip_optimizer():
    class ToyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(32, 32)
            self.linear2 = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    mod = ToyModule()
    x = torch.randn(2, 32)

    rule = Rewrite("LinearFusion", PSeq(PBox("Linear"), PBox("Linear")), PBox("FusedLinear"))
    optimizer = FXRoundtripOptimizer()
    report = optimizer.optimize_and_verify(mod, x, [rule], iters=3)

    assert report.model_name == "ToyModule"
    assert report.correctness_passed is True
    assert report.max_tensor_diff < 1e-4


def test_testbench_runner_execution(tmp_path: Path):
    runner = TestbenchRunner(verify_correctness=True, iterations=3, output_dir=tmp_path)
    report = runner.run()

    assert report.total_workloads >= 9
    assert report.passed_correctness_count == report.total_workloads
    assert report.avg_cost_reduction_pct >= 0.0
    assert (tmp_path / "testbench_results.json").exists()
    assert (tmp_path / "TESTBENCH_REPORT.md").exists()
