import pytest
from tensorgraph import (
    Obj, Signature, Box, EGraph, Extractor, Seq, Par, normalize, pretty
)
from tensorgraph.ir.primitives import Case, Iter

def test_case_eclass_merge():
    """Verify Case branches are correctly added to e-graph."""
    T = Obj("T")
    I = Obj("I")
    sig = Signature()
    sig.add("f", T, T)
    sig.add("g", T, T)
    
    eg = EGraph(sig)
    # Case(Id(I), Box(f))
    expr = Case(Box("id_i", attrs=(("obj", I),)), Box("f"))
    # (Manual registration for id_i)
    sig.add("id_i", I, I)
    
    root = eg.add_expr(expr)
    
    # Extract
    ex = Extractor(eg)
    ex.solve(root)
    res = ex.extract(root)
    
    assert isinstance(res, Case)
    assert res.right_branch.op == "f"

def test_iter_extraction():
    """Verify Iter is correctly handled by e-graph and extraction."""
    T = Obj("T")
    sig = Signature()
    sig.add("f", T, T)
    
    eg = EGraph(sig)
    expr = Iter(Box("f"), count=5)
    root = eg.add_expr(expr)
    
    ex = Extractor(eg)
    ex.solve(root)
    res = ex.extract(root)
    
    assert isinstance(res, Iter)
    assert res.count == 5
    assert res.body.op == "f"

def test_iter_endomorphism_check():
    """Verify Iter rejects non-endomorphisms."""
    A, B = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", A, B)
    
    eg = EGraph(sig)
    with pytest.raises(TypeError):
        eg.add_expr(Iter(Box("f"), count=2))
