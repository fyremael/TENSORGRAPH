from tensorgraph import Obj, Signature
from tensorgraph.egraph import EGraph
from tensorgraph.egraph.extract import Extractor
from tensorgraph.egraph.saturation import saturate
from tensorgraph.examples.demo_core import make_fuse_injects
from tensorgraph.ir import Box, Id, Par, Seq, infer_type, normalize


def count_boxes(e) -> int:
    from tensorgraph.ir import Box, Id, Par, Seq
    if isinstance(e, Box):
        return 1
    if isinstance(e, Id):
        return 0
    if isinstance(e, Seq):
        return count_boxes(e.first) + count_boxes(e.second)
    if isinstance(e, Par):
        return count_boxes(e.left) + count_boxes(e.right)
    return 0


def test_lora_fusion_reduces_boxes() -> None:
    W = Obj("W")
    X = Obj("X")
    Y = Obj("Y")

    sig = Signature()
    sig.add("InjectLoRA", W, W)
    sig.add("LinearApply", W @ X, Y)

    inj1 = Box.with_attrs("InjectLoRA", deltas=("A1B1",))
    inj2 = Box.with_attrs("InjectLoRA", deltas=("A2B2",))
    lin = Box("LinearApply")

    prog = normalize(Seq(Seq(Par(inj1, Id(X)), Par(inj2, Id(X))), lin))

    assert infer_type(prog, sig) == (W @ X, Y)

    eg = EGraph(sig)
    eg.root = eg.add_expr(prog)

    saturate(eg, [make_fuse_injects(sig)], iters=6)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    assert infer_type(best, sig) == (W @ X, Y)
    assert count_boxes(best) <= count_boxes(prog)


def test_trace_records_rewrites() -> None:
    """Test that the Trace infrastructure records rewrite applications (FR-7)."""
    from tensorgraph.egraph.trace import Trace

    W = Obj("W")
    X = Obj("X")
    Y = Obj("Y")

    sig = Signature()
    sig.add("InjectLoRA", W, W)
    sig.add("LinearApply", W @ X, Y)

    inj1 = Box.with_attrs("InjectLoRA", deltas=("A1B1",))
    inj2 = Box.with_attrs("InjectLoRA", deltas=("A2B2",))
    lin = Box("LinearApply")

    prog = normalize(Seq(Seq(Par(inj1, Id(X)), Par(inj2, Id(X))), lin))

    eg = EGraph(sig)
    eg.root = eg.add_expr(prog)

    # Create a trace and pass it to saturate
    trace = Trace()
    saturate(eg, [make_fuse_injects(sig)], iters=6, trace=trace)

    # Verify trace recorded entries
    assert len(trace) > 0, "Trace should have recorded at least one rewrite"

    # Check trace summary
    summary = trace.summary()
    assert "FuseInjectLoRA" in summary, "Trace should contain FuseInjectLoRA rule"

    # Check that entries have expected structure
    for entry in trace.entries:
        assert entry.rule_name == "FuseInjectLoRA"
        assert isinstance(entry.expr_env, dict)
        assert isinstance(entry.obj_env, dict)
        assert "X" in entry.obj_env, "Object variable X should be captured"

    # Verify dump works
    dump = trace.dump()
    assert "FuseInjectLoRA" in dump
    assert "entries" in dump

