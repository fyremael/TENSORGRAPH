
import time
import sys
from tensorgraph.egraph.egraph import EGraph
from tensorgraph.signature import Signature
from tensorgraph.types import Obj

T = Obj("T")
from tensorgraph.ir import Box, Iter, Seq, Id
from tensorgraph.egraph.saturation import saturate
from tensorgraph.library.control_flow import ALL_RULES

def bench_loop_unroll():
    print("Benchmarking Loop Unrolling...")
    sig = Signature()
    sig.add("f", T, T)
    
    eg = EGraph(sig)
    f = Box("f")
    
    # Scale test: Unroll Iter(f, N)
    depths = [1, 5, 10, 20]
    
    for N in depths:
        expr = Iter(f, N)
        root = eg.add_expr(expr)
        
        start = time.time()
        # Iters needs to be sufficient: N peels + algebra.
        # Estimate N*2 iters?
        saturate(eg, ALL_RULES, iters=N*3+10)
        end = time.time()
        
        print(f"Depth {N}: {end - start:.4f}s")
        
        # Verify result is Sequence of N f's
        # Construct expected
        chain = Id(T)
        for _ in range(N):
            chain = Seq(f, chain) # right-associative to match peel
        
        expected = eg.add_expr(chain)
        if eg.uf.find(root) != eg.uf.find(expected):
            print(f"FAILED depth {N}")
        else:
            print(f"Verified depth {N}")

if __name__ == "__main__":
    bench_loop_unroll()
