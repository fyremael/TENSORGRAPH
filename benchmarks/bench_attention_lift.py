
import time
from tensorgraph.egraph import EGraph, saturation
from tensorgraph.ir import Box, Id, Par, Seq, normalize, pretty
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.library.attention import get_attention_rules

def bench_attention_fusion():
    print("Constructing Attention Graph...")
    sig = Signature()
    T = Obj("Tensor")
    
    # Define primitives
    sig.add("transpose", T, T)
    sig.add("bmm", T @ T, T)
    sig.add("div_scalar", T, T)
    sig.add("softmax", T, T)
    sig.add("attention", T @ T @ T, T) # Fused op
    
    # Inputs: Q, K, V
    # Represented as partial computations (Boxes) to avoid Id normalization
    # collapsing the Seq structure required by the pattern.
    q = Box("input_q")
    k = Box("input_k")
    v = Box("input_v")
    
    # Add input ops to sig
    sig.add("input_q", Obj("I"), T)
    sig.add("input_k", Obj("I"), T)
    sig.add("input_v", Obj("I"), T)
    
    # Construct: bmm(softmax(div(bmm(q, transpose(k)))), v)
    
    # 1. KT = Seq(k, Box("transpose"))
    kt = Seq(k, Box("transpose"))
    
    # 2. Score = Seq(Par(q, kt), Box("bmm"))
    score = Seq(Par(q, kt), Box("bmm"))
    
    # 3. Scaled = Seq(score, Box("div_scalar"))
    scaled = Seq(score, Box("div_scalar"))
    
    # 4. Prob = Seq(scaled, Box("softmax"))
    prob = Seq(scaled, Box("softmax"))
    
    # 5. Out = Seq(Par(prob, v), Box("bmm"))
    expr = Seq(Par(prob, v), Box("bmm"))
    
    expr = normalize(expr)
    print("Initial Expression Structure:")
    print(pretty(expr))
    print("-" * 40)
    
    # Initialize EGraph
    eg = EGraph(sig)
    root = eg.add_expr(expr)
    
    print("Saturating...")
    start = time.time()
    
    rules = get_attention_rules()
    saturation.saturate(eg, rules, iters=5)
    
    end = time.time()
    print(f"Saturation done in {end - start:.4f}s")
    
    # Check if attention fusion occurred.
    # The result should be `Seq(Par(Par(Q, K), V), Box("attention"))`.
    # So root class should contain a Seq node, whose right child is `Box("attention")`.
    
    found_attention = False
    root_nodes = eg.nodes[eg.uf.find(root)]
    
    for enode in root_nodes:
        if enode.tag == "Seq":
            # Check right child
            right_id = enode.children[1]
            right_nodes = eg.nodes[eg.uf.find(right_id)]
            for rn in right_nodes:
                if rn.tag == "Box" and rn.data[0] == "attention":
                    found_attention = True
                    break
        if found_attention: break
            
    if found_attention:
        print("SUCCESS: Attention fusion detected!")
    else:
        print("FAILURE: Attention box not found.")
        print("Root nodes:", root_nodes)

def expr_size(e):
    if hasattr(e, 'children'):
        return 1 + sum(expr_size(c) for c in e.children)
    return 1

if __name__ == "__main__":
    bench_attention_fusion()
