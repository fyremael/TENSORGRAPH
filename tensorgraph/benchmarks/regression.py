"""
TENSORGRAPH Performance Regression Suite
======================================
Automated benchmarking with baseline comparison and CI integration.

Run: python -m tensorgraph.benchmarks.regression
     python -m tensorgraph.benchmarks.regression --save-baseline
     python -m tensorgraph.benchmarks.regression --compare
"""
from __future__ import annotations

import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Any

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_PATH = Path(__file__).parent / "baseline.json"
THRESHOLD_PCT = 10.0  # Alert if >10% regression

@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    
@dataclass
class BenchmarkSuite:
    version: str
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkSuite":
        suite = cls(version=data["version"], timestamp=data["timestamp"])
        for r in data["results"]:
            suite.results.append(BenchmarkResult(**r))
        return suite

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK HARNESS
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(name: str, fn: Callable[[], Any], iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Run a function multiple times and collect timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)
    
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms)
    )

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def bench_egraph_add():
    """Benchmark E-Graph node addition."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    from tensorgraph.ir import Box, Seq
    
    T = Obj("Tensor")
    sig = Signature()
    sig.add("f", T, T)
    sig.add("g", T, T)
    
    eg = EGraph(sig)
    f = Box("f")
    g = Box("g")
    
    # Build deep composition
    expr = f
    for _ in range(50):
        expr = Seq(expr, g)
    
    eg.add_expr(expr)

def bench_saturation_small():
    """Benchmark saturation on small expression."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.egraph.saturation import saturate
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    from tensorgraph.ir import Box, Iter
    from tensorgraph.library.control_flow import ALL_RULES
    
    T = Obj("Tensor")
    sig = Signature()
    sig.add("f", T, T)
    
    eg = EGraph(sig)
    eg.add_expr(Iter(Box("f"), 5))
    saturate(eg, ALL_RULES, iters=10)

def bench_saturation_medium():
    """Benchmark saturation on medium expression (Iter 10)."""
    from tensorgraph.egraph.egraph import EGraph
    from tensorgraph.egraph.saturation import saturate
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    from tensorgraph.ir import Box, Iter
    from tensorgraph.library.control_flow import ALL_RULES
    
    T = Obj("Tensor")
    sig = Signature()
    sig.add("f", T, T)
    
    eg = EGraph(sig)
    eg.add_expr(Iter(Box("f"), 10))
    saturate(eg, ALL_RULES, iters=15)

def bench_fx_trace():
    """Benchmark FX model tracing."""
    import torch
    import torch.nn as nn
    from tensorgraph.backends.fx import trace_with_leaf_modules
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.relu(x))
    
    model = Model()
    trace_with_leaf_modules(model, (nn.ReLU, nn.Sigmoid))

def bench_triton_codegen():
    """Benchmark Triton code generation."""
    from tensorgraph.codegen.triton import TritonEmitter
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    from tensorgraph.ir import Box, Seq
    
    T = Obj("Tensor")
    sig = Signature()
    sig.add("ReLU", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    
    emitter = TritonEmitter(sig)
    expr = Seq(Box("ReLU"), Box("Sigmoid"))
    emitter.emit(expr, kernel_name="bench_kernel")

def bench_distributed_merge():
    """Benchmark distributed merge propagation."""
    from tensorgraph.dist.sharding import Shard
    from tensorgraph.dist.mock_fabric import MockFabric
    from tensorgraph.signature import Signature
    from tensorgraph.types import Obj
    from tensorgraph.ir import Box
    
    T = Obj("Tensor")
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

# ─────────────────────────────────────────────────────────────────────────────
# SUITE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    ("EGraph: 50-deep Seq addition", bench_egraph_add, 20),
    ("Saturation: Iter(f, 5)", bench_saturation_small, 10),
    ("Saturation: Iter(f, 10)", bench_saturation_medium, 5),
    ("FX: Model trace", bench_fx_trace, 10),
    ("Triton: Codegen Seq", bench_triton_codegen, 50),
    ("Distributed: Merge propagation", bench_distributed_merge, 20),
]

def run_suite() -> BenchmarkSuite:
    """Run all benchmarks and return results."""
    from datetime import datetime
    
    suite = BenchmarkSuite(
        version="0.4.0",
        timestamp=datetime.now().isoformat()
    )
    
    for name, fn, iters in BENCHMARKS:
        result = benchmark(name, fn, iterations=iters)
        suite.results.append(result)
    
    return suite

def save_baseline(suite: BenchmarkSuite):
    """Save benchmark results as baseline."""
    with open(BASELINE_PATH, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)
    print(f"Baseline saved to {BASELINE_PATH}")

def load_baseline() -> BenchmarkSuite | None:
    """Load baseline if it exists."""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH) as f:
        return BenchmarkSuite.from_dict(json.load(f))

def compare_results(current: BenchmarkSuite, baseline: BenchmarkSuite) -> list[tuple[str, float, bool]]:
    """Compare current results to baseline. Returns (name, pct_change, is_regression)."""
    baseline_map = {r.name: r for r in baseline.results}
    comparisons = []
    
    for r in current.results:
        if r.name in baseline_map:
            base = baseline_map[r.name]
            pct_change = ((r.mean_ms - base.mean_ms) / base.mean_ms) * 100
            is_regression = pct_change > THRESHOLD_PCT
            comparisons.append((r.name, pct_change, is_regression))
    
    return comparisons

# ─────────────────────────────────────────────────────────────────────────────
# CLI OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
CYAN = "\033[38;2;0;255;255m"
AMBER = "\033[38;2;255;191;0m"
GREEN = "\033[38;2;0;255;127m"
RED = "\033[38;2;255;69;58m"
STEEL = "\033[38;2;113;128;150m"
CHROME = "\033[38;2;200;200;210m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header():
    print()
    print(f"{CYAN}{'═' * 78}{RESET}")
    print(f"  {CHROME}{BOLD}TENSORGRAPH{RESET}  {STEEL}//  PERFORMANCE REGRESSION SUITE  //  {AMBER}v0.4.0{RESET}")
    print(f"{CYAN}{'═' * 78}{RESET}")
    print()

def print_results(suite: BenchmarkSuite, comparisons: list | None = None):
    """Print benchmark results table."""
    comp_map = {c[0]: c for c in comparisons} if comparisons else {}
    
    print(f"  {STEEL}{'Benchmark':<40} {'Mean':>10} {'Std':>10} {'Δ':>10}{RESET}")
    print(f"  {STEEL}{'─' * 70}{RESET}")
    
    for r in suite.results:
        delta_str = ""
        name_color = CHROME
        
        if r.name in comp_map:
            _, pct, is_reg = comp_map[r.name]
            if is_reg:
                delta_str = f"{RED}+{pct:.1f}%{RESET}"
                name_color = RED
            elif pct < -THRESHOLD_PCT:
                delta_str = f"{GREEN}{pct:.1f}%{RESET}"
                name_color = GREEN
            else:
                delta_str = f"{STEEL}{pct:+.1f}%{RESET}"
        
        print(f"  {name_color}{r.name:<40}{RESET} {r.mean_ms:>9.2f}ms {r.std_ms:>9.2f}ms {delta_str:>10}")
    
    print()

def main():
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    
    parser = argparse.ArgumentParser(description="TENSORGRAPH Performance Regression Suite")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as new baseline")
    parser.add_argument("--compare", action="store_true", help="Compare against baseline")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 on regression")
    args = parser.parse_args()
    
    print_header()
    
    print(f"  {STEEL}Running benchmarks...{RESET}\n")
    suite = run_suite()
    
    comparisons = None
    if args.compare or args.ci:
        baseline = load_baseline()
        if baseline:
            comparisons = compare_results(suite, baseline)
            print(f"  {STEEL}Comparing against baseline from {baseline.timestamp}{RESET}\n")
        else:
            print(f"  {AMBER}No baseline found. Run with --save-baseline first.{RESET}\n")
    
    print_results(suite, comparisons)
    
    if args.save_baseline:
        save_baseline(suite)
    
    # CI mode: check for regressions
    if args.ci and comparisons:
        regressions = [c for c in comparisons if c[2]]
        if regressions:
            print(f"  {RED}{BOLD}REGRESSION DETECTED{RESET}")
            for name, pct, _ in regressions:
                print(f"    {RED}• {name}: +{pct:.1f}%{RESET}")
            print()
            return 1
        else:
            print(f"  {GREEN}{BOLD}NO REGRESSIONS{RESET}\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
