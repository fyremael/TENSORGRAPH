import pytest

from tensorgraph import Box, EGraph, Extractor, Obj, Signature
from tensorgraph.ir.primitives import Case, Iter


def test_case_eclass_merge():
    """Verify a well-typed Case is added to and extracted from the e-graph."""
    tensor = Obj("T")
    unit = Obj("I")
    sig = Signature()
    sig.add("init", unit, tensor)
    sig.add("f", tensor, tensor)

    eg = EGraph(sig)
    expr = Case(Box("init"), Box("f"))
    root = eg.add_expr(expr)

    extractor = Extractor(eg)
    extractor.solve(root)
    result = extractor.extract(root)

    assert isinstance(result, Case)
    assert result.left_branch.op == "init"
    assert result.right_branch.op == "f"


def test_iter_extraction():
    """Verify Iter is correctly handled by e-graph and extraction."""
    tensor = Obj("T")
    sig = Signature()
    sig.add("f", tensor, tensor)

    eg = EGraph(sig)
    expr = Iter(Box("f"), count=5)
    root = eg.add_expr(expr)

    extractor = Extractor(eg)
    extractor.solve(root)
    result = extractor.extract(root)

    assert isinstance(result, Iter)
    assert result.count == 5
    assert result.body.op == "f"


def test_iter_endomorphism_check():
    """Verify Iter rejects non-endomorphisms."""
    source, target = Obj("A"), Obj("B")
    sig = Signature()
    sig.add("f", source, target)

    eg = EGraph(sig)
    with pytest.raises(TypeError):
        eg.add_expr(Iter(Box("f"), count=2))
