from __future__ import annotations

"""TENSORGRAPH + torch.fx optimizer demo.

Run:
  python -m tensorgraph.cli.optimize_fx --model toy_lora_chain --in-dim 64 --out-dim 64 --batch 2 --iters 6

This is a strict MVP intended as a reference integration pattern.

Optimization demonstrated:
  Fuse consecutive LoRAInject stages.
"""

import argparse

from ..backends.fx import (
    build_toy_lora_chain,
    expr_to_sequential_module,
    fx_chain_to_ops,
    is_linear_chain,
    ops_to_expr,
    trace_with_leaf_modules,
)
from ..egraph import EGraph, ENode
from ..egraph.extract import Extractor
from ..egraph.saturation import saturate
from ..ir import Box, pretty
from ..rewrite import PSeq, PVar, Rewrite
from ..signature import Signature
from ..types import Obj


def _try_get_lora_deltas(eg: EGraph, cid: int, op_name: str) -> tuple[str, ...] | None:
    """Return deltas if eclass contains `Box(op_name, deltas=...)`."""

    cid = eg.uf.find(cid)
    for en in eg.nodes[cid]:
        if en.tag == "Box":
            op, attrs = en.data
            if op == op_name:
                d = dict(attrs).get("deltas", ())
                return d if isinstance(d, tuple) else None
    return None


def make_fuse_lora_chain_rule(op_name: str) -> Rewrite:
    """Fuse consecutive LoRAInject markers in a linear chain.

        LoRAInject(d1) ; (LoRAInject(d2) ; tail)
          <-> LoRAInject(d1+d2) ; tail
    """

    lhs = PSeq(PVar("i1"), PSeq(PVar("i2"), PVar("tail")))

    def rhs_builder(eg: EGraph, root: int, env: dict[str, int], oenv: dict[str, Obj]) -> int:
        i1 = env["i1"]
        i2 = env["i2"]
        tail = env["tail"]

        d1 = _try_get_lora_deltas(eg, i1, op_name)
        d2 = _try_get_lora_deltas(eg, i2, op_name)
        if d1 is None or d2 is None:
            return eg.uf.find(root)

        fused_id = eg.add_expr(Box.with_attrs(op_name, deltas=d1 + d2))

        fused_id = eg.uf.find(fused_id)
        tail = eg.uf.find(tail)
        df, cf = eg.sort[fused_id]
        dt, ct = eg.sort[tail]

        if cf != dt:
            return eg.uf.find(root)

        return eg.add_enode(ENode("Seq", (), (fused_id, tail)), (df, ct))

    return Rewrite(name="FuseLoRAInjects", lhs=lhs, rhs=rhs_builder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="toy_lora_chain", choices=["toy_lora_chain"])
    ap.add_argument("--in-dim", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    model, LoRAInject = build_toy_lora_chain(args.in_dim, args.out_dim)

    # Import torch lazily
    import torch
    import torch.nn as nn

    model.eval()

    # Trace with FX
    gm = trace_with_leaf_modules(model, leaf_types=(LoRAInject, nn.Linear))
    print("\n=== Original FX Graph ===")
    print(gm.graph)

    if not is_linear_chain(gm):
        raise RuntimeError("This MVP expects a simple linear-chain FX graph")

    ops = fx_chain_to_ops(gm, lora_inject_type=LoRAInject, linear_type=nn.Linear)

    # Build signature
    T = Obj("Tensor")
    sig = Signature()
    seen = set()
    for op in ops:
        if op.op_name in seen:
            continue
        sig.add(op.op_name, T, T)
        seen.add(op.op_name)

    expr = ops_to_expr(ops, sig, T)
    print("\n=== TENSORGRAPH Expr (before) ===")
    print(pretty(expr))

    eg = EGraph(sig)
    root = eg.add_expr(expr)
    eg.root = root

    rewrites = [make_fuse_lora_chain_rule(op_name=LoRAInject.__name__)]
    saturate(eg, rewrites, iters=args.iters)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("\n=== TENSORGRAPH Expr (after / extracted best) ===")
    print(pretty(best))

    opt_model = expr_to_sequential_module(best, model, lora_inject_type=LoRAInject, linear_type=nn.Linear)
    opt_model.eval()

    opt_gm = trace_with_leaf_modules(opt_model, leaf_types=(LoRAInject, nn.Linear))
    print("\n=== Optimized FX Graph (rebuilt) ===")
    print(opt_gm.graph)

    # Numerical sanity check
    x = torch.randn(args.batch, args.in_dim)
    with torch.no_grad():
        y0 = model(x)
        y1 = opt_model(x)

    max_err = (y0 - y1).abs().max().item()
    print(f"\nSanity: max |y0 - y1| = {max_err:.6g}")
    print("Note: LoRAInject is a no-op marker in this demo, so exact match is expected")


if __name__ == "__main__":
    main()
