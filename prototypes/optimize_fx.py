#!/usr/bin/env python3
"""
optimize_fx.py

TENSORGRAPH + torch.fx integration (first practical hook).

What this does:
1) Trace a PyTorch module with torch.fx
2) Convert a simple FX *linear chain* graph into TENSORGRAPH IR (Expr)
3) Run equality saturation (e-graphs) using 2-morphism rewrite rules
4) Extract the cheapest equivalent program by a cost model
5) Rebuild an optimized nn.Sequential + FX GraphModule for this supported subset

Why this matters:
- It turns "graph surgery" optimizations (fusion, motion, cancellation) into
  typed, reusable rewrite rules, then uses equality saturation to find the best
  equivalent form automatically.

Supported subset (MVP, intentionally strict):
- Single-input, single-output *linear chain* of call_module nodes.
- Ops are treated as Tensor -> Tensor.
- Demo optimization: fuse consecutive LoRAInject markers.

Run:
  python optimize_fx.py --model toy_lora_chain --in-dim 64 --out-dim 64 --batch 2

"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

# Import TENSORGRAPH v2 (e-graph engine)
from tensorgraph_eg import (
    Obj, Signature,
    Expr, Id, Box, Seq,
    Pattern, PVar, PBox, PSeq,
    Rewrite, EGraph, Extractor, saturate, normalize, pretty, ENode
)


# -----------------------------------------------------------------------------
# 1) Toy modules for a realistic first integration target
# -----------------------------------------------------------------------------

class LoRAInject(nn.Module):
    """
    A *marker* module representing a LoRA injection stage.

    For this demo we keep semantics trivial: forward(x) = x.
    The point is to show that TENSORGRAPH can:
      - recognize sequential injections
      - fuse them via rewrite rules
      - rebuild the FX graph accordingly

    In a real PEFT/LoRA setting, these stages would correspond to
    actual adapter effects on weights/activations, and fusion/motion
    would depend on validity constraints.
    """
    def __init__(self, deltas: Tuple[str, ...]):
        super().__init__()
        if not isinstance(deltas, tuple):
            raise TypeError("deltas must be a tuple for stability.")
        self.deltas = deltas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToyLoRAChain(nn.Module):
    """
    A simple sequential chain:
        x -> LoRAInject(d1) -> LoRAInject(d2) -> Linear -> y
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lora1 = LoRAInject(("A1B1",))
        self.lora2 = LoRAInject(("A2B2",))
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora1(x)
        x = self.lora2(x)
        x = self.proj(x)
        return x




# -----------------------------------------------------------------------------
# 1.5) FX tracing helper: keep certain modules as "leaves"
# -----------------------------------------------------------------------------

class LeafTracer(fx.Tracer):
    """
    By default, FX will trace *into* submodules.

    For identity-ish marker modules like LoRAInject, tracing into them
    makes them disappear (their forward is just `return x`).

    We mark LoRAInject (and Linear) as leaves so they appear as call_module ops.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, (LoRAInject, nn.Linear)):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def trace_with_leaf_modules(model: nn.Module) -> fx.GraphModule:
    tracer = LeafTracer()
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)


# -----------------------------------------------------------------------------
# 2) FX -> TENSORGRAPH conversion for the "linear chain" subset
# -----------------------------------------------------------------------------

@dataclass
class FXChainOp:
    op_name: str
    attrs: Dict[str, Any]
    module_ref: Optional[str]
    node: fx.Node


def is_linear_chain(gm: fx.GraphModule) -> bool:
    """
    True iff:
      placeholder -> call_module -> call_module -> ... -> output

    (MVP strictness is intentional.)
    """
    nodes = list(gm.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    outputs = [n for n in nodes if n.op == "output"]
    if len(placeholders) != 1 or len(outputs) != 1:
        return False

    cur = placeholders[0]
    seen = {cur}
    while True:
        users = list(cur.users.keys())
        if len(users) != 1:
            return False
        nxt = users[0]
        if nxt in seen:
            return False
        seen.add(nxt)

        if nxt.op == "output":
            break

        if nxt.op != "call_module":
            return False
        if len(nxt.args) != 1 or nxt.args[0] != cur:
            return False

        cur = nxt

    return True


def fx_chain_to_ops(gm: fx.GraphModule) -> List[FXChainOp]:
    assert is_linear_chain(gm), "FX graph is not a supported linear chain."

    modules = dict(gm.named_modules())
    nodes = list(gm.graph.nodes)
    placeholder = next(n for n in nodes if n.op == "placeholder")

    ops: List[FXChainOp] = []
    cur = placeholder
    while True:
        nxt = next(iter(cur.users.keys()))
        if nxt.op == "output":
            break

        target = str(nxt.target)
        mod = modules[target]
        op_name = type(mod).__name__

        attrs: Dict[str, Any] = {}
        if isinstance(mod, LoRAInject):
            attrs["deltas"] = mod.deltas
        elif isinstance(mod, nn.Linear):
            # NOTE: we don't encode weights into IR yet; we preserve them at rebuild time
            attrs["in_features"] = mod.in_features
            attrs["out_features"] = mod.out_features
            attrs["bias"] = (mod.bias is not None)

        ops.append(FXChainOp(op_name=op_name, attrs=attrs, module_ref=target, node=nxt))
        cur = nxt

    return ops


def ops_to_expr(ops: List[FXChainOp], sig: Signature, tensor_obj: Obj) -> Expr:
    """
    Convert chain ops into a right-associated Seq of Boxes.
    """
    expr: Optional[Expr] = None
    for op in ops:
        # Ensure signature contains this op
        _ = sig.get(op.op_name)
        box = Box.with_attrs(op.op_name, **op.attrs) if op.attrs else Box(op.op_name)
        expr = box if expr is None else normalize(Seq(expr, box))
    return expr if expr is not None else Id(tensor_obj)


# -----------------------------------------------------------------------------
# 3) Rewrite library (2-morphisms) for this integration
# -----------------------------------------------------------------------------

def _try_get_lora_deltas(eg: EGraph, cid: int) -> Optional[Tuple[str, ...]]:
    """
    Return LoRA deltas if cid contains a LoRAInject Box; otherwise None.
    """
    cid = eg.uf.find(cid)
    for en in eg.nodes[cid]:
        if en.tag == "Box":
            op, attrs = en.data
            if op == "LoRAInject":
                d = dict(attrs).get("deltas", ())
                return d if isinstance(d, tuple) else None
    return None


def make_fuse_lora_chain() -> Rewrite:
    """
    Fuse consecutive LoRAInject stages in a linear chain:

        LoRAInject(d1) ; (LoRAInject(d2) ; tail)
            <->  LoRAInject(d1+d2) ; tail

    This is the smallest "real" optimization you want for PEFT/adapter stacks.
    """
    lhs = PSeq(PVar("i1"), PSeq(PVar("i2"), PVar("tail")))

    def rhs_builder(eg: EGraph, root: int, env: Dict[str, int], oenv: Dict[str, Obj]) -> int:
        i1 = env["i1"]
        i2 = env["i2"]
        tail = env["tail"]

        d1 = _try_get_lora_deltas(eg, i1)
        d2 = _try_get_lora_deltas(eg, i2)
        if d1 is None or d2 is None:
            # Not applicable; keep root unchanged.
            return eg.uf.find(root)

        fused_expr = Box.with_attrs("LoRAInject", deltas=d1 + d2)
        fused_id = eg.add_expr(fused_expr)

        fused_id = eg.uf.find(fused_id)
        tail = eg.uf.find(tail)
        df, cf = eg.sort[fused_id]
        dt, ct = eg.sort[tail]
        if cf != dt:
            return eg.uf.find(root)

        sort = (df, ct)
        return eg.add_enode(ENode("Seq", (), (fused_id, tail)), sort)

    return Rewrite(name="FuseLoRAInjects", lhs=lhs, rhs=rhs_builder)


# -----------------------------------------------------------------------------
# 4) Rebuild an optimized module from extracted Expr (supported subset)
# -----------------------------------------------------------------------------

def flatten_seq_boxes(e: Expr) -> List[Box]:
    """
    Flatten (right-associated) Seq into a list of Boxes.
    """
    e = normalize(e)
    out: List[Box] = []

    def rec(x: Expr) -> None:
        x = normalize(x)
        if isinstance(x, Box):
            out.append(x)
        elif isinstance(x, Seq):
            rec(x.first)
            rec(x.second)
        elif isinstance(x, Id):
            pass
        else:
            raise ValueError(f"Unsupported Expr for rebuild: {type(x)} :: {pretty(x)}")

    rec(e)
    return out


def expr_to_sequential_module(best: Expr, original: nn.Module) -> nn.Module:
    """
    Build nn.Sequential for:
      - LoRAInject stages (with fused deltas)
      - nn.Linear (copied from original)

    This is intentionally narrow: it's a first integration step.
    """
    boxes = flatten_seq_boxes(best)

    # Grab the first Linear layer from the original for parameter transfer.
    orig_linear: Optional[nn.Linear] = None
    for m in original.modules():
        if isinstance(m, nn.Linear):
            orig_linear = m
            break

    layers: List[nn.Module] = []
    for b in boxes:
        if b.op == "LoRAInject":
            d = dict(b.attrs).get("deltas", ())
            layers.append(LoRAInject(deltas=d if isinstance(d, tuple) else ()))
        elif b.op == "Linear":
            if orig_linear is None:
                raise ValueError("Original model has no Linear to copy.")
            layers.append(copy.deepcopy(orig_linear))
        else:
            raise ValueError(f"Cannot rebuild op '{b.op}' in MVP rebuild.")

    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# 5) Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="toy_lora_chain",
                    choices=["toy_lora_chain"], help="Demo model to run")
    ap.add_argument("--in-dim", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    if args.model == "toy_lora_chain":
        model: nn.Module = ToyLoRAChain(args.in_dim, args.out_dim)
    else:
        raise ValueError("Unknown model")

    model.eval()

    # Trace with FX
    gm = trace_with_leaf_modules(model)
    print("\n=== Original FX Graph ===")
    print(gm.graph)

    if not is_linear_chain(gm):
        raise RuntimeError("This MVP expects a simple linear-chain FX graph.")

    # Extract chain ops
    ops = fx_chain_to_ops(gm)

    # Build signature for TENSORGRAPH
    T = Obj("Tensor")
    sig = Signature()
    seen_ops = set()
    for op in ops:
        # Tensor -> Tensor in this MVP
        if op.op_name in seen_ops:
            continue
        sig.add(op.op_name, T, T)
        seen_ops.add(op.op_name)

    # Convert to Expr
    expr = ops_to_expr(ops, sig, T)
    print("\n=== TENSORGRAPH Expr (before) ===")
    print(pretty(expr))

    # E-graph saturation
    eg = EGraph(sig)
    root = eg.add_expr(expr)
    eg.root = root

    rewrites = [make_fuse_lora_chain()]
    saturate(eg, rewrites, iters=args.iters)

    # Extract best equivalent program
    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("\n=== TENSORGRAPH Expr (after / extracted best) ===")
    print(pretty(best))

    # Rebuild optimized model + trace again for inspection
    opt_model = expr_to_sequential_module(best, model)
    opt_model.eval()

    opt_gm = trace_with_leaf_modules(opt_model)
    print("\n=== Optimized FX Graph (rebuilt) ===")
    print(opt_gm.graph)

    # Numerical sanity check
    x = torch.randn(args.batch, args.in_dim)
    with torch.no_grad():
        y0 = model(x)
        y1 = opt_model(x)

    max_err = (y0 - y1).abs().max().item()
    print(f"\nSanity: max |y0 - y1| = {max_err:.6g}")
    print("Note: LoRAInject is a no-op marker in this demo, so exact match is expected.")


if __name__ == "__main__":
    main()
