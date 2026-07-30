import pytest
from tensorgraph import (
    Obj, Signature, Box, EGraph, Extractor, Seq, Par, normalize, pretty
)
from tensorgraph.ir.primitives import Dup, Del, Swap

def test_del_naturality():
    """Verify f ; Del(B) == Del(A)"""
    A, B = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", A, B)
    
    eg = EGraph(sig)
    # f ; Del(B)
    expr = Seq(Box("f"), Del(B))
    root = eg.add_expr(expr)
    
    # Del(A)
    target = Del(A)
    target_id = eg.add_expr(target)
    
    eg.rebuild()
    
    assert eg.uf.find(root) == eg.uf.find(target_id)

def test_dup_naturality():
    """Verify f ; Dup(B) == Dup(A) ; (f ⊗ f)"""
    A, B = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", A, B)
    
    eg = EGraph(sig)
    # f ; Dup(B)
    expr = Seq(Box("f"), Dup(B))
    root = eg.add_expr(expr)
    
    # Dup(A) ; (f ⊗ f)
    target = Seq(Dup(A), Par(Box("f"), Box("f")))
    target_id = eg.add_expr(target)
    
    eg.rebuild()
    
    assert eg.uf.find(root) == eg.uf.find(target_id)

def test_swap_naturality():
    """Verify (f ⊗ g) ; Swap(B, D) == Swap(A, C) ; (g ⊗ f)"""
    A, B, C, D = Obj("A"), Obj("B"), Obj("C"), Obj("D")
    sig = Signature()
    sig.add("f", A, B)
    sig.add("g", C, D)
    
    eg = EGraph(sig)
    # (f ⊗ g) ; Swap(B, D)
    expr = Seq(Par(Box("f"), Box("g")), Swap(B, D))
    root = eg.add_expr(expr)
    
    # Swap(A, C) ; (g ⊗ f)
    target = Seq(Swap(A, C), Par(Box("g"), Box("f")))
    target_id = eg.add_expr(target)
    
    eg.rebuild()
    
    assert eg.uf.find(root) == eg.uf.find(target_id)
