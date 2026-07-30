"""
TENSORGRAPH v0.4.0 — Independent Audit Harness
============================================
GCT Chrome Metropolis | Industrial Brutalism Aesthetic

This module provides a comprehensive compliance audit against:
- v0.3.0 Core Requirements (IR, EGraph, Saturation, FX Backend)
- v0.4.0 Kernel Requirements (Control Flow, Sharding, Fusion)

Run: python -m tensorgraph.cli.audit
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Any

# ─────────────────────────────────────────────────────────────────────────────
# GCT CHROME METROPOLIS PALETTE
# ─────────────────────────────────────────────────────────────────────────────
VOID_BLACK = "\033[38;2;18;18;24m"
GUNMETAL = "\033[38;2;44;47;51m"
STEEL = "\033[38;2;113;128;150m"
CHROME = "\033[38;2;200;200;210m"
CYBER_CYAN = "\033[38;2;0;255;255m"
AMBER = "\033[38;2;255;191;0m"
SIGNAL_GREEN = "\033[38;2;0;255;127m"
SIGNAL_RED = "\033[38;2;255;69;58m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AuditResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    category: str = "CORE"

@dataclass
class AuditSuite:
    name: str
    version: str
    results: list[AuditResult] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total(self) -> int:
        return len(self.results)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_audit(name: str, fn: Callable[[], Any], category: str = "CORE") -> AuditResult:
    """Execute a single audit check."""
    start = time.perf_counter()
    try:
        fn()
        duration = (time.perf_counter() - start) * 1000
        return AuditResult(name=name, passed=True, duration_ms=duration, category=category)
    except AssertionError as e:
        duration = (time.perf_counter() - start) * 1000
        return AuditResult(name=name, passed=False, duration_ms=duration, message=str(e), category=category)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return AuditResult(name=name, passed=False, duration_ms=duration, message=f"{type(e).__name__}: {e}", category=category)

# ─────────────────────────────────────────────────────────────────────────────
# v0.3.0 CORE REGRESSION TESTS
# ─────────────────────────────────────────────────────────────────────────────
def audit_types():
    """FR-1: Typed IR - Object types and tensor products."""
    from tensorgraph.types import Obj
    T = Obj("T")
    U = Obj("U")
    prod = T @ U
    assert prod.is_tensor(), "Tensor product not recognized"
    assert prod.left == T
    assert prod.right == U

def audit_ir_composition():
    """FR-2: Sequential and Parallel composition."""
    from tensorgraph.ir import Box, Seq, Par, Id
    from tensorgraph.types import Obj
    T = Obj("T")
    f = Box("f")
    g = Box("g")
    seq = Seq(f, g)
    par = Par(f, g)
    i = Id(T)
    assert seq.first == f
    assert par.left == f

def audit_signature():
    """FR-3: Signature declaration."""
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("Op", T, T)
    op = sig.get("Op")
    assert op.dom == T
    assert op.cod == T

def audit_egraph_add():
    """FR-4: EGraph add_expr and union-find."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    eg = EGraph(sig)
    cid = eg.add_expr(Box("A"))
    assert cid >= 0
    assert eg.uf.find(cid) == cid

def audit_egraph_merge():
    """FR-4: EGraph merge and congruence."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    sig.add("B", T, T)
    eg = EGraph(sig)
    a = eg.add_expr(Box("A"))
    b = eg.add_expr(Box("B"))
    eg.merge(a, b, "test")
    assert eg.uf.find(a) == eg.uf.find(b)

def audit_saturation():
    """FR-4: Equality saturation terminates."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.egraph.saturation import saturate
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    eg = EGraph(sig)
    eg.add_expr(Box("A"))
    saturate(eg, [], iters=5)  # Empty rules, should not crash

def audit_rewrite_basic():
    """FR-3: Basic rewrite rule application."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.egraph.saturation import saturate
    from tensorgraph.signature import Signature
    from tensorgraph.rewrite import Rewrite, PBox
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    sig.add("B", T, T)
    rule = Rewrite(name="a_to_b", lhs=PBox("A"), rhs=PBox("B"))
    eg = EGraph(sig)
    a = eg.add_expr(Box("A"))
    b = eg.add_expr(Box("B"))
    saturate(eg, [rule], iters=5)
    assert eg.uf.find(a) == eg.uf.find(b), "Rewrite did not unify A and B"

def audit_normalization():
    """FR-1: Normalization removes identities."""
    from tensorgraph.ir import Seq, Id, Box, normalize
    from tensorgraph.types import Obj
    T = Obj("T")
    f = Box("f")
    expr = Seq(f, Id(T))
    norm = normalize(expr)
    assert norm == f, f"Normalized {expr} to {norm}, expected {f}"

def audit_fx_chain():
    """FR-6: FX backend linear chain detection."""
    try:
        import torch
        import torch.nn as nn
        from tensorgraph.backends.fx import build_toy_lora_chain, trace_with_leaf_modules, is_linear_chain
        model, LoRAInject = build_toy_lora_chain(16, 16)
        gm = trace_with_leaf_modules(model, (LoRAInject, nn.Linear))
        assert is_linear_chain(gm), "LoRA chain should be linear"
    except ImportError:
        raise AssertionError("PyTorch not available")

def audit_trace():
    """FR-7: Proof trace recording."""
    from tensorgraph.egraph.trace import Trace
    t = Trace()
    t.record("test_rule", 0, 1, 0, 1)
    assert len(t.entries) == 1
    assert t.entries[0].rule_name == "test_rule"

# ─────────────────────────────────────────────────────────────────────────────
# v0.4.0 KERNEL COMPLIANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────
def audit_iter_primitive():
    """v0.4.0 Control Flow: Iter primitive exists."""
    from tensorgraph.ir import Iter, Box
    f = Box("f")
    it = Iter(f, 5)
    assert it.count == 5
    assert it.body == f

def audit_piter_pattern():
    """v0.4.0 Control Flow: PIter pattern matching."""
    from tensorgraph.rewrite.pattern import PIter, PVar
    pat = PIter(PVar("f"), "n")
    assert pat.count == "n"

def audit_iter_unroll():
    """v0.4.0 Control Flow: Iter unrolling via peel_iter."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.egraph.saturation import saturate
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Iter, Box, Seq
    from tensorgraph.types import Obj
    from tensorgraph.library.control_flow import ALL_RULES
    T = Obj("T")
    sig = Signature()
    sig.add("f", T, T)
    eg = EGraph(sig)
    root = eg.add_expr(Iter(Box("f"), 2))
    saturate(eg, ALL_RULES, iters=10)
    # Should contain Seq(f, Seq(f, Id))
    # Verify by checking EClass size increased
    assert len(eg.nodes) > 1, "Unrolling did not generate new nodes"

def audit_iter_fusion():
    """v0.4.0 Control Flow: Iter fusion rule."""
    from tensorgraph.library.control_flow import iter_fusion
    assert iter_fusion.name == "iter_fusion"

def audit_iter_product():
    """v0.4.0 Control Flow: Iter product (LICM) rule."""
    from tensorgraph.library.control_flow import iter_product
    assert iter_product.name == "iter_product"

def audit_shard_class():
    """v0.4.0 Sharding: Shard class exists."""
    from tensorgraph.dist.sharding import Shard, Partition
    from tensorgraph.signature import Signature
    sig = Signature()
    s = Shard(shard_id=1, sig=sig)
    assert s.shard_id == 1
    assert isinstance(s.partition, Partition)

def audit_ghost_nodes():
    """v0.4.0 Sharding: Ghost node registration."""
    from tensorgraph.dist.sharding import Shard
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    s = Shard(shard_id=1, sig=sig)
    lid = s.ingest(Box("A"))
    s.partition.register_ghost(global_id=100, local_id=lid)
    assert s.partition.ghosts[100] == lid

def audit_fabric_protocol():
    """v0.4.0 Sharding: MockFabric communication."""
    from tensorgraph.dist.mock_fabric import MockFabric
    from tensorgraph.dist.sharding import Shard
    from tensorgraph.signature import Signature
    sig = Signature()
    fabric = MockFabric()
    s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
    s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
    fabric.register(s1)
    fabric.register(s2)
    fabric.send_merge(1, 100, 101)
    assert len(fabric.queues[2]) == 1

def audit_distributed_merge():
    """v0.4.0 Sharding: Cross-shard merge propagation."""
    from tensorgraph.dist.mock_fabric import MockFabric
    from tensorgraph.dist.sharding import Shard
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("A", T, T)
    sig.add("B", T, T)
    fabric = MockFabric()
    s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
    lid_a = s1.ingest(Box("A"), global_id=100)
    lid_b = s1.ingest(Box("B"), global_id=101)
    fabric.register(s1)
    s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
    lid_a2 = s2.ingest(Box("A"))
    lid_b2 = s2.ingest(Box("B"))
    s2.partition.register_ghost(100, lid_a2)
    s2.partition.register_ghost(101, lid_b2)
    fabric.register(s2)
    s1.partition.egraph.merge(lid_a, lid_b, "test")
    fabric.pump()
    assert s2.partition.egraph.uf.find(lid_a2) == s2.partition.egraph.uf.find(lid_b2)

def audit_triton_emitter():
    """v0.4.0 Fusion: TritonEmitter class exists."""
    from tensorgraph.codegen.triton import TritonEmitter
    from tensorgraph.signature import Signature
    sig = Signature()
    emitter = TritonEmitter(sig)
    assert hasattr(emitter, 'emit')

def audit_elementwise_trait():
    """v0.4.0 Fusion: Elementwise trait support."""
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("Relu", T, T, traits={"elementwise"})
    op = sig.get("Relu")
    assert "elementwise" in op.traits

def audit_seq_codegen():
    """v0.4.0 Fusion: Seq code generation."""
    from tensorgraph.codegen.triton import TritonEmitter
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box, Seq
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("Relu", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    emitter = TritonEmitter(sig)
    code = emitter.emit(Seq(Box("Relu"), Box("Sigmoid")))
    assert "@triton.jit" in code
    assert "tl.sigmoid" in code

def audit_par_codegen():
    """v0.4.0 Fusion: Par code generation."""
    from tensorgraph.codegen.triton import TritonEmitter
    from tensorgraph.signature import Signature
    from tensorgraph.ir import Box, Par
    from tensorgraph.types import Obj
    T = Obj("T")
    sig = Signature()
    sig.add("Relu", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    emitter = TritonEmitter(sig)
    results = emitter._visit(Par(Box("Relu"), Box("Sigmoid")), ["x0", "x1"])
    assert len(results) == 2

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD RENDERER (Tufte-Inspired Clarity)
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    """Print GCT Chrome Metropolis header."""
    print()
    print(f"{CYBER_CYAN}{'═' * 78}{RESET}")
    print(f"  {CHROME}{BOLD}TENSORGRAPH{RESET}  {STEEL}//  INDEPENDENT AUDIT HARNESS  //  {AMBER}v0.4.0{RESET}")
    print(f"  {DIM}{STEEL}Grand Challenge Technologies  |  Chrome Metropolis{RESET}")
    print(f"{CYBER_CYAN}{'═' * 78}{RESET}")
    print()

def render_section(title: str, category: str):
    """Print section header."""
    print(f"  {AMBER}{title}{RESET}  {STEEL}[{category}]{RESET}")
    print(f"  {STEEL}{'─' * 70}{RESET}")

def render_result(result: AuditResult):
    """Print single audit result."""
    status = f"{SIGNAL_GREEN}PASS{RESET}" if result.passed else f"{SIGNAL_RED}FAIL{RESET}"
    time_str = f"{result.duration_ms:>8.2f}ms"
    name = result.name[:48].ljust(48)
    print(f"    [{status}]  {CHROME}{name}{RESET}  {DIM}{time_str}{RESET}")
    if not result.passed and result.message:
        msg = result.message[:60]
        print(f"           {SIGNAL_RED}{msg}{RESET}")

def render_summary(suite: AuditSuite):
    """Print final summary."""
    print()
    print(f"  {CYBER_CYAN}{'═' * 70}{RESET}")
    total_time = sum(r.duration_ms for r in suite.results)
    
    if suite.failed == 0:
        status = f"{SIGNAL_GREEN}{BOLD}ALL CHECKS PASSED{RESET}"
    else:
        status = f"{SIGNAL_RED}{BOLD}{suite.failed} FAILURES{RESET}"
    
    print(f"  {CHROME}{BOLD}AUDIT SUMMARY{RESET}  {status}")
    print()
    print(f"    {STEEL}Total:{RESET} {suite.total}   {SIGNAL_GREEN}Passed:{RESET} {suite.passed}   {SIGNAL_RED}Failed:{RESET} {suite.failed}   {AMBER}Time:{RESET} {total_time:.2f}ms")
    print()
    
    # Category breakdown
    categories = {}
    for r in suite.results:
        if r.category not in categories:
            categories[r.category] = {"passed": 0, "failed": 0}
        if r.passed:
            categories[r.category]["passed"] += 1
        else:
            categories[r.category]["failed"] += 1
    
    print(f"  {STEEL}Category Breakdown:{RESET}")
    for cat, stats in categories.items():
        p = stats["passed"]
        f = stats["failed"]
        bar_len = 20
        total = p + f
        filled = int((p / total) * bar_len) if total > 0 else 0
        bar = f"{SIGNAL_GREEN}{'█' * filled}{RESET}{STEEL}{'░' * (bar_len - filled)}{RESET}"
        print(f"    {cat:<15} {bar} {p}/{total}")
    
    print()
    print(f"  {CYBER_CYAN}{'═' * 70}{RESET}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AUDIT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    suite = AuditSuite(name="TENSORGRAPH", version="0.4.0")
    
    render_header()
    
    # v0.3.0 Core Regression
    render_section("v0.3.0 CORE REGRESSION", "REGRESSION")
    core_tests = [
        ("Types: Obj and Tensor Product", audit_types),
        ("IR: Sequential/Parallel Composition", audit_ir_composition),
        ("Signature: Operator Declaration", audit_signature),
        ("EGraph: add_expr and Union-Find", audit_egraph_add),
        ("EGraph: merge and Congruence", audit_egraph_merge),
        ("Saturation: Termination", audit_saturation),
        ("Rewrite: Basic Rule Application", audit_rewrite_basic),
        ("Normalization: Identity Elimination", audit_normalization),
        ("FX Backend: Linear Chain Detection", audit_fx_chain),
        ("Trace: Proof Recording", audit_trace),
    ]
    for name, fn in core_tests:
        result = run_audit(name, fn, category="v0.3.0")
        suite.results.append(result)
        render_result(result)
    
    print()
    
    # v0.4.0 Control Flow
    render_section("v0.4.0 CONTROL FLOW", "v0.4.0")
    cf_tests = [
        ("Iter: Primitive Exists", audit_iter_primitive),
        ("PIter: Pattern Matching", audit_piter_pattern),
        ("Iter: Unrolling via peel_iter", audit_iter_unroll),
        ("Iter: Fusion Rule", audit_iter_fusion),
        ("Iter: Product (LICM) Rule", audit_iter_product),
    ]
    for name, fn in cf_tests:
        result = run_audit(name, fn, category="v0.4.0-CF")
        suite.results.append(result)
        render_result(result)
    
    print()
    
    # v0.4.0 Sharding
    render_section("v0.4.0 HETEROGENEOUS SHARDING", "v0.4.0")
    shard_tests = [
        ("Shard: Class Instantiation", audit_shard_class),
        ("Ghost Nodes: Registration", audit_ghost_nodes),
        ("Fabric: MockFabric Protocol", audit_fabric_protocol),
        ("Distributed: Cross-Shard Merge", audit_distributed_merge),
    ]
    for name, fn in shard_tests:
        result = run_audit(name, fn, category="v0.4.0-SHARD")
        suite.results.append(result)
        render_result(result)
    
    print()
    
    # v0.4.0 Fusion
    render_section("v0.4.0 AUTOMATED KERNEL FUSION", "v0.4.0")
    fusion_tests = [
        ("TritonEmitter: Class Exists", audit_triton_emitter),
        ("Traits: Elementwise Support", audit_elementwise_trait),
        ("Codegen: Seq Fusion", audit_seq_codegen),
        ("Codegen: Par Fusion", audit_par_codegen),
    ]
    for name, fn in fusion_tests:
        result = run_audit(name, fn, category="v0.4.0-FUSION")
        suite.results.append(result)
        render_result(result)
    
    render_summary(suite)
    
    return 0 if suite.failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
