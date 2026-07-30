"""
TENSORGRAPH Benchmark Suite: Saturation
Usage: python benchmarks/bench_saturation.py
"""
import time
import sys
import os
from dataclasses import dataclass
from typing import List

# Ensure package is in path
sys.path.insert(0, os.getcwd())

from tensorgraph.ir import Box, Seq, Id
from tensorgraph.egraph import EGraph, saturation
from tensorgraph.rewrite import Rewrite, PSeq, PVar
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.cli import style as S

# Force UTF-8 output for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class Result:
    name: str
    nodes: int
    iters: int
    duration_ms: float
    nodes_in_egraph: int

def build_chain(length: int) -> Seq:
    """Build a chain of length n: f ; f ; ... ; f"""
    current = Box("f")
    for _ in range(length - 1):
        current = Seq(current, Box("f"))
    return current

def bench_saturation(chain_len: int) -> Result:
    T = Obj("T")
    sig = Signature()
    sig.add("f", T, T)
    
    # Rule: f ; f -> f (idempotent)
    # This will cause massive contraction
    def fuse_rhs(eg, root, env, oenv):
        x_id = eg.uf.find(env["x"])
        x_nodes = list(eg.nodes[x_id])
        if x_nodes and x_nodes[0].data == "f":
             return x_id
        return root

    # Rule: f -> f ; f (expansion - to fight contraction)
    # We won't use this one for now as it causes infinite loops easily without cost control.
    # Instead let's use associativity which explores the graph structure.
    # (a ; b) ; c -> a ; (b ; c)
    # This is expensive on chains.
    
    rules = []
    # Associativity rule
    # PSeq(PSeq(x, y), z) -> PSeq(x, PSeq(y, z))
    # We need to define this carefully or just use the idempotence which is the main claim.
    
    # Let's benchmark the "Fuse" rule on long chains of identical items.
    # This tests pattern matching speed.
    fuse_rule = Rewrite("Fuse", PSeq(PVar("x"), PVar("y")), fuse_rhs)
    rules.append(fuse_rule)
    
    eg = EGraph(sig)
    expr = build_chain(chain_len)
    eg.root = eg.add_expr(expr)
    
    start = time.perf_counter()
    saturation.saturate(eg, rules, iters=50)
    end = time.perf_counter()
    
    return Result(
        name=f"Chain(n={chain_len})",
        nodes=chain_len,
        iters=50,
        duration_ms=(end - start) * 1000,
        nodes_in_egraph=len(eg.nodes)
    )

def main():
    print(S.header("TENSORGRAPH BENCHMARKS", "SATURATION"))
    print(S.section("CONFIGURATION"))
    print(S.metric("CPU", "System Default", S.steel))
    
    results = []
    
    lengths = [10, 50, 100]
    
    print(S.section("RUNNING"))
    
    for l in lengths:
        print(f"  Benchmarking Chain(n={l})... ", end="", flush=True)
        try:
            res = bench_saturation(l)
            results.append(res)
            print(S.success(f"{res.duration_ms:.2f}ms"))
        except Exception as e:
            print(S.error(f"FAILED: {e}"))
            
    print(S.section("RESULTS"))
    print(f"  {'Dataset':<20} | {'Nodes':<10} | {'Time (ms)':<10} | {'EGraph Size':<12}")
    print("  " + "-"*60)
    for r in results:
        color = S.green if r.duration_ms < 1000 else (S.amber if r.duration_ms < 5000 else S.red)
        print(f"  {r.name:<20} | {r.nodes:<10} | {color(f'{r.duration_ms:.2f}'):<20} | {r.nodes_in_egraph:<12}")
        
    print(S.footer())

if __name__ == "__main__":
    main()
