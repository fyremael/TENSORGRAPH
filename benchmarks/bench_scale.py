"""
TENSORGRAPH Benchmark Suite: Real-World Scale Stress Test
Usage: python benchmarks/bench_scale.py
"""
import time
import sys
import os
from dataclasses import dataclass

# Ensure package is in path
sys.path.insert(0, os.getcwd())

from tensorgraph.ir import Box, Seq, Par
from tensorgraph.egraph import EGraph, saturation
from tensorgraph.rewrite import Rewrite, PSeq, PVar, PBox, PPar
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.cli import style as S

# Force UTF-8 output for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class Result:
    name: str
    nodes_initial: int
    nodes_final: int
    iters: int
    duration_ms: float

def build_resnet_block(i: int) -> Seq:
    """Simulate a ResNet block structure:
       x -> (conv1 ; relu ; conv2) + x -> out
       Represented as: Dup ; ( (Conv ; Relu ; Conv) ⊗ Id ) ; Add
    """
    # Simply using sequential composition for scale depth testing 
    # as strict structural types might get complex for quick bench
    # Block = Conv ; Relu ; Conv
    return Seq(Box(f"Conv{i}"), Seq(Box("Relu"), Box(f"Conv{i}_2")))

def build_deep_network(layers: int) -> Seq:
    """Build a deep sequence of blocks."""
    current = build_resnet_block(0)
    for i in range(1, layers):
        current = Seq(current, build_resnet_block(i))
    return current

def bench_network(layers: int) -> Result:
    T = Obj("Tensor")
    sig = Signature()
    # Register minimal ops
    sig.add("Relu", T, T)
    for i in range(layers):
        sig.add(f"Conv{i}", T, T)
        sig.add(f"Conv{i}_2", T, T)
    
    # 1. Rules: Associativity allows massive exploration of the e-class
    # x ; (y ; z) <=> (x ; y) ; z
    
    # Rule 1: Expand canonical form (right-assoc) to left-assoc
    # x ; (y ; z) -> (x ; y) ; z
    assoc_r2l = Rewrite(
        "AssocR2L",
        PSeq(PVar("x"), PSeq(PVar("y"), PVar("z"))),
        PSeq(PSeq(PVar("x"), PVar("y")), PVar("z"))
    )

    # Rule 2: Contract left-assoc back to canonical (triggers merges)
    # (x ; y) ; z -> x ; (y ; z)
    assoc_l2r = Rewrite(
        "AssocL2R",
        PSeq(PSeq(PVar("x"), PVar("y")), PVar("z")),
        PSeq(PVar("x"), PSeq(PVar("y"), PVar("z")))
    )

    rules = [assoc_r2l, assoc_l2r]
    
    # Build graph
    expr = build_deep_network(layers)
    
    eg = EGraph(sig)
    eg.add_expr(expr)
    
    initial_nodes = len(eg.nodes)
    
    start = time.perf_counter()
    # Low iter count because assoc explodes quickly without worklist rebuild
    saturation.saturate(eg, rules, iters=5, max_applications=5000)
    end = time.perf_counter()
    
    return Result(
        name=f"ResNet-{layers*3}", # 3 nodes per block
        nodes_initial=initial_nodes,
        nodes_final=len(eg.nodes),
        iters=5,
        duration_ms=(end - start) * 1000
    )

def main():
    print(S.header("TENSORGRAPH BENCHMARKS", "SCALE STRESS TEST"))
    
    configs = [10, 50, 100] # Layer counts implies x3 nodes
    results = []
    
    for c in configs:
        print(f"  Benchmarking DeepNet(layers={c})... ", end="", flush=True)
        try:
            res = bench_network(c)
            results.append(res)
            print(S.success(f"{res.duration_ms:.2f}ms"))
        except Exception as e:
            print(S.error(f"FAILED: {e}"))
            import traceback
            traceback.print_exc()
            
    print(S.section("RESULTS"))
    print(f"  {'Dataset':<20} | {'Nodes':<10} | {'Time (ms)':<10} | {'Growth X':<10}")
    print("  " + "-"*60)
    for r in results:
        growth = r.nodes_final / r.nodes_initial if r.nodes_initial else 0
        print(f"  {r.name:<20} | {r.nodes_initial:<10} | {r.duration_ms:<10.2f} | {growth:<10.2f}")
        
    print(S.footer())

if __name__ == "__main__":
    main()
