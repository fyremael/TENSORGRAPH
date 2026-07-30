
import pytest
from tensorgraph.egraph.egraph import EGraph
from tensorgraph.signature import Signature
from tensorgraph.types import Obj

T = Obj("T")
from tensorgraph.ir import Box, Seq, Iter, Id
from tensorgraph.egraph.saturation import saturate
from tensorgraph.library.control_flow import ALL_RULES

def test_peel_iteration():
    sig = Signature()
    sig.add("f", T, T)
    
    eg = EGraph(sig)
    
    # Iter(f, 3)
    f = Box("f")
    expr = Iter(f, 3)
    root = eg.add_expr(expr)
    
    # Expected: f ; f ; f ; Id
    # or f ; (f ; (f ; Id))
    
    saturate(eg, ALL_RULES, iters=10)
    
    # Construct expected structure manually to check equivalence
    # f ; f ; f
    # Note: loop ends with Id.
    
    f_node = eg.add_expr(f)
    seq3 = eg.add_expr(Seq(f, Seq(f, Seq(f, Id(T)))))
    
    print(f"Root: {eg.uf.find(root)}")
    print(f"Seq3: {eg.uf.find(seq3)}")
    if eg.uf.find(root) != eg.uf.find(seq3):
        print("Nodes in root class:", eg.nodes[eg.uf.find(root)])
        print("Nodes in seq3 class:", eg.nodes[eg.uf.find(seq3)])

    assert eg.uf.find(root) == eg.uf.find(seq3)

def test_unroll_zero():
    sig = Signature()
    sig.add("f", T, T)
    eg = EGraph(sig)
    
    expr = Iter(Box("f"), 0)
    root = eg.add_expr(expr)
    
    saturate(eg, ALL_RULES, iters=3)
    
    saturate(eg, ALL_RULES, iters=3)
    
    assert eg.uf.find(root) == eg.uf.find(eg.add_expr(Id(T)))

def test_unroll_one():
    sig = Signature()
    sig.add("f", T, T)
    eg = EGraph(sig)
    
    f = Box("f")
    expr = Iter(f, 1)
    root = eg.add_expr(expr)
    
    from tensorgraph.egraph.trace import Trace
    t = Trace()
    saturate(eg, ALL_RULES, iters=5, trace=t)
    print("Applied rules:", [rec.rule_name for rec in t.entries])
    
    # Expected: f ; Id
    expected = eg.add_expr(Seq(f, Id(T)))
    print(f"Root: {eg.uf.find(root)}")
    print(f"Expected: {eg.uf.find(expected)}")
    print(f"Nodes Root: {eg.nodes[eg.uf.find(root)]}")
    print(f"Nodes Expected: {eg.nodes[eg.uf.find(expected)]}")
    assert eg.uf.find(root) == eg.uf.find(expected)


if __name__ == "__main__":
    test_unroll_zero()
    print("Zero passed")
    test_unroll_one()
    print("One passed")
    test_peel_iteration()
    print("Peel passed")
