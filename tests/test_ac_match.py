"""
TENSORGRAPH v0.2.0: AC-Matching Tests

Verifies P1-4 and P1-5: canonical form for AC-terms and multiset partition matching.
"""
import pytest

from tensorgraph.egraph import EGraph
from tensorgraph.ir import Box, Par, Seq
from tensorgraph.rewrite import PPar, PVar, ac_ematch
from tensorgraph.signature import Signature
from tensorgraph.types import Obj


@pytest.fixture
def tensor_obj():
    return Obj("T")


@pytest.fixture
def signature(tensor_obj):
    sig = Signature()
    sig.add("A", tensor_obj, tensor_obj)
    sig.add("B", tensor_obj, tensor_obj)
    sig.add("C", tensor_obj, tensor_obj)
    return sig


# -----------------------------------------------------------------------------
# P1-4: Canonical form for AC-terms
# -----------------------------------------------------------------------------


def test_ac_match_simple_swap(signature, tensor_obj):
    """Test that PPar matches both orderings."""
    eg = EGraph(signature)
    
    # Add Par(A, B)
    expr = Par(Box("A"), Box("B"))
    eg.add_expr(expr)
    
    # Pattern PPar(PVar("x"), PVar("y"))
    pat = PPar(PVar("x"), PVar("y"))
    
    # Should match with BOTH orderings
    matches = ac_ematch(eg, pat)
    
    # Each Par node should produce 2 matches (x=A,y=B and x=B,y=A)
    assert len(matches) >= 2


def test_ac_match_specific_box(signature, tensor_obj):
    """Test AC-match with specific box pattern."""
    from tensorgraph.rewrite import PBox
    
    eg = EGraph(signature)
    
    # Add Par(A, B)
    expr = Par(Box("A"), Box("B"))
    eg.add_expr(expr)
    
    # Pattern PPar(PBox("B"), PVar("x")) - note B is first
    pat = PPar(PBox("B"), PVar("x"))
    
    # Should match Par(A, B) with x=A (commutativity)
    matches = ac_ematch(eg, pat)
    
    assert len(matches) >= 1


def test_ac_match_nested_par(signature, tensor_obj):
    """Test AC-match with nested Par structure."""
    eg = EGraph(signature)
    
    # Add Par(Par(A, B), C)
    expr = Par(Par(Box("A"), Box("B")), Box("C"))
    eg.add_expr(expr)
    
    # Simple pattern at top level
    pat = PPar(PVar("left"), PVar("right"))
    
    matches = ac_ematch(eg, pat)
    
    # Should find the top-level Par
    assert len(matches) >= 2  # Both orderings


# -----------------------------------------------------------------------------
# P1-5: Multiset partition matching
# -----------------------------------------------------------------------------


def test_multiset_partition_symmetric(signature, tensor_obj):
    """Test that symmetric patterns find all matches."""
    eg = EGraph(signature)
    
    # Add Par(A, A) - symmetric
    expr = Par(Box("A"), Box("A"))
    eg.add_expr(expr)
    
    pat = PPar(PVar("x"), PVar("y"))
    
    matches = ac_ematch(eg, pat)
    
    # x and y both get A, but should have 2 orderings
    assert len(matches) >= 2


def test_ac_match_preserves_bindings(signature, tensor_obj):
    """Test that AC-match produces correct bindings."""
    from tensorgraph.rewrite import PBox
    
    eg = EGraph(signature)
    
    expr = Par(Box("A"), Box("B"))
    root = eg.add_expr(expr)
    
    # Pattern: Par(PBox("A"), PVar("other"))
    pat = PPar(PBox("A"), PVar("other"))
    
    matches = ac_ematch(eg, pat)
    
    # Should have match where "other" is bound to B's class
    assert len(matches) >= 1
    
    # Verify binding
    found_b = False
    for rep, env, oenv in matches:
        if "other" in env:
            # Check that "other" points to a class containing Box("B")
            other_class = env["other"]
            nodes = eg.nodes.get(eg.uf.find(other_class), set())
            for n in nodes:
                if n.tag == "Box" and n.data[0] == "B":
                    found_b = True
                    break
    
    assert found_b, "AC-match should bind 'other' to class containing B"
