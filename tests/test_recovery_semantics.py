import pytest

from tensorgraph import Box, EGraph, Obj, Seq, Signature
from tensorgraph.egraph.enode import ENode
from tensorgraph.ir import Case, Del, Iter


def test_case_rejected_identically_by_expression_and_enode_paths() -> None:
    unit = Obj("I")
    a = Obj("A")
    b = Obj("B")
    c = Obj("C")
    sig = Signature()
    sig.add("bad_left", a, b)
    sig.add("right", a, b)
    sig.add("left", unit, b)
    sig.add("wrong_codomain", a, c)

    eg = EGraph(sig)
    with pytest.raises(TypeError, match="left branch must have domain I"):
        eg.add_expr(Case(Box("bad_left"), Box("right")))

    with pytest.raises(TypeError, match="same codomain"):
        eg.add_expr(Case(Box("left"), Box("wrong_codomain")))

    left = eg.add_expr(Box("left"))
    right = eg.add_expr(Box("right"))
    with pytest.raises(TypeError, match="declared sort"):
        eg.add_enode(ENode("Case", (), (left, right)), (a, b))


def test_iter_rejects_non_integer_negative_and_non_endomorphic_forms() -> None:
    a = Obj("A")
    b = Obj("B")
    sig = Signature()
    sig.add("step", a, a)
    sig.add("not_endo", a, b)

    with pytest.raises(TypeError, match="statically known integer"):
        Iter(Box("step"), True)
    with pytest.raises(ValueError, match="non-negative"):
        Iter(Box("step"), -1)

    eg = EGraph(sig)
    with pytest.raises(TypeError, match="endomorphism"):
        eg.add_expr(Iter(Box("not_endo"), 2))

    body = eg.add_expr(Box("step"))
    with pytest.raises(ValueError, match="non-negative"):
        eg.add_enode(ENode("Iter", (-1,), (body,)), (a, a))


def test_copy_delete_naturality_applies_to_pure_morphisms() -> None:
    a = Obj("A")
    b = Obj("B")
    sig = Signature()
    sig.add("pure_f", a, b)

    eg = EGraph(sig)
    composite = eg.add_expr(Seq(Box("pure_f"), Del(b)))
    eg.rebuild()
    direct = eg.add_expr(Del(a))
    eg.rebuild()

    assert eg.uf.find(composite) == eg.uf.find(direct)
    assert any(reason == "del_naturality_pure" for reason, _, _ in eg.merge_log)


def test_copy_delete_naturality_is_blocked_by_effects() -> None:
    a = Obj("A")
    b = Obj("B")
    sig = Signature()
    sig.add("random_f", a, b, traits={"effectful"})

    eg = EGraph(sig)
    composite = eg.add_expr(Seq(Box("random_f"), Del(b)))
    eg.rebuild()
    direct = eg.add_expr(Del(a))
    eg.rebuild()

    assert eg.uf.find(composite) != eg.uf.find(direct)
    assert not any(reason.startswith("del_naturality") for reason, _, _ in eg.merge_log)


def test_signature_rejects_contradictory_effect_traits() -> None:
    a = Obj("A")
    sig = Signature()
    with pytest.raises(ValueError, match="both pure and effectful"):
        sig.add("bad", a, a, traits={"pure", "effectful"})
