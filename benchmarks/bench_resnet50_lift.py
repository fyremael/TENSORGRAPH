
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from tensorgraph.backends.fx_dag import lift_fx_graph
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.ir import pretty
import time

import sys
sys.setrecursionlimit(5000)

def bench_resnet50_lift():
    print("Loading ResNet-50...")
    model = models.resnet50(weights=None)
    model.eval()

    print("Tracing with torch.fx...")
    gm = symbolic_trace(model)

    print("Lifting to TENSORGRAPH...")
    sig = Signature()
    T = Obj("Tensor")
    
    start = time.time()
    try:
        # Debug: Print node count
        print(f"Graph nodes: {len(list(gm.graph.nodes))}")
        
        expr = lift_fx_graph(gm, sig, T)
        end = time.time()
        
        print(f"Lifting successful in {end - start:.4f}s")
        print(f"Expression depth: {depth(expr)}")
        print(f"Signature entries: {len(sig)}")
        
    except Exception as e:
        print(f"Lifting FAILED: {e}")
        import traceback
        traceback.print_exc()

def depth(expr):
    if hasattr(expr, 'children'):
        return 1 + max((depth(c) for c in expr.children), default=0)
    return 1

if __name__ == "__main__":
    bench_resnet50_lift()
