import pytest
from tensorgraph import (
    Obj, Signature, Box, Rewrite, PBox, PSeq, PVar, EGraph, saturate, Extractor
)
from tensorgraph.adjunction import Adjunction

def test_automated_mate_synthesis_ltr():
    """Verify Pattern 1: f ; u ≡ v ; f  =>  u ≡ f ; v ; g"""
    A, B = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", A, B)
    sig.add("g", B, A)
    sig.add("u", A, A)
    sig.add("v", B, B)
    
    # Commuting square: (f ; u) ≡ (v ; f)
    commute = Rewrite(
        name="commute",
        lhs=PSeq(PBox("f"), PBox("u")),
        rhs=PSeq(PBox("v"), PBox("f"))
    )
    
    adj = Adjunction(f_lower=Box("f"), g_lift=Box("g"))
    mate = adj.transport_rule(commute)
    
    assert mate.name == "mate_ltr(commute)"
    # LHS of mate should be 'u'
    assert isinstance(mate.lhs, PBox) and mate.lhs.op == "u"
    
    # Verify optimization using the mate
    eg = EGraph(sig)
    root = eg.add_expr(Box("u"))
    
    # We want to see if 'u' can be rewritten to 'f ; v ; g'
    saturate(eg, [mate])
    
    ex = Extractor(eg)
    ex.solve(root)
    # Both 'u' and 'f ; v ; g' should be in the same e-class
    # Note: extraction depends on cost, but here they are equivalent.
    
    # Add an identity to force one to be cheaper? 
    # Or just check if the sequence is in the e-class.
    reps = eg.nodes[eg.uf.find(root)]
    tags = {en.tag for en in reps}
    assert "Seq" in tags # Indicates f ; (v ; g) was added

def test_automated_mate_synthesis_rtl():
    """Verify Pattern 2: u ; g ≡ g ; v  =>  v ≡ f ; u ; g"""
    A, B = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", A, B)
    sig.add("g", B, A)
    sig.add("u", A, A)
    sig.add("v", B, B)
    
    # Commuting square: (u ; g) ≡ (g ; v)
    commute = Rewrite(
        name="commute_rtl",
        lhs=PSeq(PBox("u"), PBox("g")),
        rhs=PSeq(PBox("g"), PBox("v"))
    )
    
    adj = Adjunction(f_lower=Box("f"), g_lift=Box("g"))
    mate = adj.transport_rule(commute)
    
    assert mate.name == "mate_rtl(commute_rtl)"
    # LHS of mate should be 'v'
    assert isinstance(mate.lhs, PBox) and mate.lhs.op == "v"
