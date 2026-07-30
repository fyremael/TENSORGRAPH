from __future__ import annotations

"""Core TENSORGRAPH demo (no PyTorch).

1) Fuse consecutive InjectLoRA stages in a small tensor-product diagram.
2) Demonstrate adjunction mates transporting a commuting rewrite.

Run:
  python -m tensorgraph.examples.demo_core
"""


from ..adjunction import Adjunction
from ..egraph import EGraph, ENode
from ..egraph.extract import Extractor
from ..egraph.saturation import saturate
from ..ir import Box, Id, Par, Seq, infer_type, normalize, pretty
from ..rewrite import PBox, PId, PPar, PSeq, PVar, Rewrite
from ..signature import Signature
from ..types import Obj, ObjVar


def _count_boxes_expr(e) -> int:
    if isinstance(e, Box):
        return 1
    if isinstance(e, Id):
        return 0
    if isinstance(e, Seq):
        return _count_boxes_expr(e.first) + _count_boxes_expr(e.second)
    if isinstance(e, Par):
        return _count_boxes_expr(e.left) + _count_boxes_expr(e.right)
    return 0


def _get_inject_deltas(eg: EGraph, cid: int) -> tuple[str, ...]:
    cid = eg.uf.find(cid)
    for en in eg.nodes[cid]:
        if en.tag == "Box":
            op, attrs = en.data
            if op == "InjectLoRA":
                d = dict(attrs).get("deltas", ())
                if not isinstance(d, tuple):
                    raise ValueError("deltas must be tuple")
                return d
    raise ValueError("No InjectLoRA in eclass")


def make_fuse_injects(sig: Signature) -> Rewrite:
    """Fuse consecutive LoRA injections in a tensor-product diagram."""

    lhs = PSeq(
        PPar(PVar("i1"), PId(ObjVar("X"))),
        PSeq(
            PPar(PVar("i2"), PId(ObjVar("X"))),
            PVar("tail"),
        ),
    )

    def rhs_builder(eg: EGraph, root: int, env, oenv) -> int:
        i1 = env["i1"]
        i2 = env["i2"]
        tail = env["tail"]
        X = oenv["X"]

        d1 = _get_inject_deltas(eg, i1)
        d2 = _get_inject_deltas(eg, i2)

        fused = Box.with_attrs("InjectLoRA", deltas=d1 + d2)
        lhs_par = eg.add_expr(Par(fused, Id(X)))

        lhs_par = eg.uf.find(lhs_par)
        tail = eg.uf.find(tail)

        d1_, c1_ = eg.sort[lhs_par]
        d2_, c2_ = eg.sort[tail]
        if c1_ != d2_:
            raise TypeError(f"FuseInjectLoRA RHS Seq mismatch: {c1_} != {d2_}")

        return eg.add_enode(ENode("Seq", (), (lhs_par, tail)), (d1_, c2_))

    return Rewrite(name="FuseInjectLoRA", lhs=lhs, rhs=rhs_builder)


def demo_lora_fusion() -> None:
    W = Obj("W")
    X = Obj("X")
    Y = Obj("Y")

    sig = Signature()
    sig.add("InjectLoRA", W, W)
    sig.add("LinearApply", W @ X, Y)

    inj1 = Box.with_attrs("InjectLoRA", deltas=("A1B1",))
    inj2 = Box.with_attrs("InjectLoRA", deltas=("A2B2",))
    lin = Box("LinearApply")

    prog = normalize(
        Seq(
            Seq(
                Par(inj1, Id(X)),
                Par(inj2, Id(X)),
            ),
            lin,
        )
    )

    print("\n=== LoRA fusion demo (e-graph equality saturation) ===")
    print("Original:", pretty(prog))
    print("Boxes:", _count_boxes_expr(prog))
    print("Type:", infer_type(prog, sig))

    eg = EGraph(sig)
    eg.root = eg.add_expr(prog)

    saturate(eg, [make_fuse_injects(sig)], iters=6)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("Best    :", pretty(best))
    print("Boxes   :", _count_boxes_expr(best))


def demo_mates_transport() -> None:
    A = Obj("A")
    B = Obj("B")

    sig = Signature()
    sig.add("Lower", A, B)
    sig.add("Lift", B, A)
    sig.add("OptA", A, A)
    sig.add("OptB", B, B)

    alpha = Rewrite(
        name="LowerCommutesWithOpt",
        lhs=PSeq(PBox("Lower"), PBox("OptA")),
        rhs=PSeq(PBox("OptB"), PBox("Lower")),
    )

    adj = Adjunction(f_lower=Box("Lower"), g_lift=Box("Lift"))
    mate = adj.mate_left_to_right(alpha)

    print("\n=== Mates transport demo ===")
    print("alpha.lhs:", alpha.lhs)
    print("alpha.rhs:", alpha.rhs)
    print("mate.lhs :", mate.lhs)
    print("mate.rhs :", mate.rhs)

    eg = EGraph(sig)
    eg.root = eg.add_expr(Box("OptA"))

    saturate(eg, [mate], iters=3)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("Extracted best for OptA class:", pretty(best))


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    demo_lora_fusion()
    demo_mates_transport()


if __name__ == "__main__":
    main()
