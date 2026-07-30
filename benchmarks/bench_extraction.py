"""
TENSORGRAPH Benchmark Suite: Extraction
Usage: python benchmarks/bench_extraction.py
"""
import time
import sys
import os
import traceback
from dataclasses import dataclass

# Ensure package is in path
sys.path.insert(0, os.getcwd())

from tensorgraph.ir import Box, Seq, Id, Par
from tensorgraph.egraph import EGraph, saturation, extract
from tensorgraph.rewrite import Rewrite, PSeq, PVar, PBox
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
    duration_ms: float
    extracted_size: int

def bench_extraction_chain(length: int) -> Result:
    T = Obj("T")
    sig = Signature()
    sig.add("f", T, T)
    
    # Create simple chain
    def make_chain(n):
        e = Box("f")
        for _ in range(n-1):
            e = Seq(e, Box("f"))
        return e
        
    expr = make_chain(length)
    
    eg = EGraph(sig)
    eg.root = eg.add_expr(expr)
    
    # Just extraction on raw graph is fast.
    # We want to benchmark extraction on a "dense" graph.
    # Let's add variations.
    
    # Add equivalent loop: f = f;f;f (fake rule for density)
    # Actually, let's just use the extractor on the raw graph first to set baseline.
    
    start = time.perf_counter()
    ex = extract.Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)
    end = time.perf_counter()
    
    # Size estimate
    size = 0 
    # dirty size count
    q = [best]
    while q:
        curr = q.pop()
        size += 1
        if isinstance(curr, Seq):
            q.append(curr.first)
            q.append(curr.second)
    
    return Result(
        name=f"Chain(n={length})",
        nodes=len(eg.nodes),
        duration_ms=(end - start) * 1000,
        extracted_size=size
    )

def main():
    print(S.header("TENSORGRAPH BENCHMARKS", "EXTRACTION"))
    print(S.section("CONFIGURATION"))
    print(S.metric("CPU", "System Default", S.steel))
    
    results = []
    
    lengths = [10, 100, 1000]
    
    print(S.section("RUNNING"))
    
    for l in lengths:
        print(f"  Benchmarking Extraction(n={l})... ", end="", flush=True)
        try:
            res = bench_extraction_chain(l)
            results.append(res)
            print(S.success(f"{res.duration_ms:.2f}ms"))
        except Exception as e:
            print(S.error(f"FAILED: {e}"))
            traceback.print_exc()
            
    print(S.section("RESULTS"))
    print(f"  {'Dataset':<20} | {'Nodes':<10} | {'Time (ms)':<10} | {'Extracted':<12}")
    print("  " + "-"*60)
    for r in results:
        color = S.green if r.duration_ms < 100 else (S.amber if r.duration_ms < 500 else S.red)
        print(f"  {r.name:<20} | {r.nodes:<10} | {color(f'{r.duration_ms:.2f}'):<20} | {r.extracted_size:<12}")
        
    print(S.footer())

if __name__ == "__main__":
    main()
