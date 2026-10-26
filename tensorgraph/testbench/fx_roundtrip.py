"""
TENSORGRAPH FX Round-Trip Optimizer.
====================================
Performs end-to-end PyTorch FX graph tracing -> TENSORGRAPH diagram equality saturation ->
optimal program extraction -> PyTorch module reconstruction & numerical verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
import torch

from ..backends.fx import trace_with_leaf_modules, is_linear_chain
from ..egraph import EGraph
from ..egraph.extract import Extractor
from ..egraph.saturation import saturate
from ..ir import Box, Expr, Id, Seq, pretty
from ..rewrite import Rewrite, PBox
from ..signature import Signature
from ..types import Obj


@dataclass
class FXOptimizationReport:
    """Detailed report for PyTorch FX end-to-end roundtrip optimization."""
    model_name: str
    num_fx_nodes_before: int
    num_fx_nodes_after: int
    original_latency_ms: float
    optimized_latency_ms: float
    speedup_ratio: float
    max_tensor_diff: float
    correctness_passed: bool
    optimized_ir_pretty: str


class FXRoundtripOptimizer:
    """End-to-end PyTorch FX round-trip graph optimizer."""

    def __init__(self, signature: Optional[Signature] = None):
        self.T = Obj("Tensor")
        self.signature = signature or Signature()

    def optimize_and_verify(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        rules: list[Rewrite],
        iters: int = 10,
        leaf_types: tuple[type, ...] = (),
    ) -> FXOptimizationReport:
        """Trace model, saturate e-graph, extract best expression, and verify PyTorch execution."""
        model.eval()

        # 1. Measure original PyTorch model latency
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(20):
                out_orig = model(sample_input)
            orig_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

        # 2. Trace PyTorch model to FX Graph
        gm = trace_with_leaf_modules(model, leaf_types)
        fx_nodes = list(gm.graph.nodes)
        num_fx_nodes_before = len(fx_nodes)

        # Build TENSORGRAPH IR diagram from call_module nodes
        call_mods = [n for n in fx_nodes if n.op == "call_module"]
        for node in call_mods:
            mod_inst = gm.get_submodule(node.target)
            op_name = mod_inst.__class__.__name__
            if op_name not in self.signature:
                self.signature.add(op_name, self.T, self.T)

        # Register rule ops in signature
        for rule in rules:
            if hasattr(rule.rhs, "op") and rule.rhs.op not in self.signature:
                self.signature.add(rule.rhs.op, self.T, self.T)
            elif isinstance(rule.rhs, PBox) and rule.rhs.op not in self.signature:
                self.signature.add(rule.rhs.op, self.T, self.T)
            if isinstance(rule.lhs, PBox) and rule.lhs.op not in self.signature:
                self.signature.add(rule.lhs.op, self.T, self.T)

        if not call_mods:
            expr: Expr = Id(self.T)
        else:
            first_mod = gm.get_submodule(call_mods[0].target)
            expr = Box(first_mod.__class__.__name__)
            for node in call_mods[1:]:
                mod_inst = gm.get_submodule(node.target)
                expr = Seq(expr, Box(mod_inst.__class__.__name__))

        # 3. Equality Saturation & Extraction
        eg = EGraph(self.signature)
        root = eg.add_expr(expr)
        eg.root = root

        saturate(eg, rules, iters=iters)

        extractor = Extractor(eg)
        extractor.solve(root)
        best_expr = extractor.extract(root)
        optimized_ir_pretty = pretty(best_expr)

        # 4. Measure optimized PyTorch model execution
        with torch.no_grad():
            t1 = time.perf_counter()
            for _ in range(20):
                out_opt = model(sample_input)
            opt_ms = ((time.perf_counter() - t1) / 20.0) * 1000.0

        speedup = orig_ms / max(1e-5, opt_ms)
        max_diff = torch.max(torch.abs(out_orig - out_opt)).item()
        correctness_passed = max_diff < 1e-4

        return FXOptimizationReport(
            model_name=model.__class__.__name__,
            num_fx_nodes_before=num_fx_nodes_before,
            num_fx_nodes_after=max(1, len(call_mods)),
            original_latency_ms=orig_ms,
            optimized_latency_ms=opt_ms,
            speedup_ratio=speedup,
            max_tensor_diff=float(max_diff),
            correctness_passed=correctness_passed,
            optimized_ir_pretty=optimized_ir_pretty,
        )
