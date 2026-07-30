from __future__ import annotations

"""torch.fx backend (MVP).

This backend is intentionally strict: it supports a **linear chain** FX graph

    placeholder -> call_module -> call_module -> ... -> output

All modules in the chain are treated as `Tensor -> Tensor`.

Goal: end-to-end demonstration
FX -> TENSORGRAPH IR -> saturation -> extraction -> rebuild.

PyTorch is an optional dependency for the core TENSORGRAPH package.
This module raises a clear error if torch is unavailable.
"""

from dataclasses import dataclass
from typing import Any

from ..ir import Box, Expr, Id, Seq, normalize, pretty
from ..signature import Signature
from ..types import Obj


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.fx as fx  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the FX backend. Install PyTorch to use tensorgraph.backends.fx"
        ) from e


# -----------------------------------------------------------------------------
# 1) Demo modules
# -----------------------------------------------------------------------------


def build_toy_lora_chain(in_dim: int, out_dim: int):
    """Create a minimal model for the FX demo.

    Structure:
        x -> LoRAInject(d1) -> LoRAInject(d2) -> Linear -> y

    The LoRAInject blocks are markers and are identity maps.
    """

    _require_torch()
    import torch.nn as nn

    class LoRAInject(nn.Module):
        def __init__(self, deltas: tuple[str, ...]):
            super().__init__()
            if not isinstance(deltas, tuple):
                raise TypeError("deltas must be a tuple")
            self.deltas = deltas

        def forward(self, x):
            return x

    class ToyLoRAChain(nn.Module):
        def __init__(self, in_dim_: int, out_dim_: int):
            super().__init__()
            self.lora1 = LoRAInject(("A1B1",))
            self.lora2 = LoRAInject(("A2B2",))
            self.proj = nn.Linear(in_dim_, out_dim_, bias=True)

        def forward(self, x):
            x = self.lora1(x)
            x = self.lora2(x)
            x = self.proj(x)
            return x

    return ToyLoRAChain(in_dim, out_dim), LoRAInject


# -----------------------------------------------------------------------------
# 2) FX helpers
# -----------------------------------------------------------------------------


def trace_with_leaf_modules(model, leaf_types: tuple[type, ...]):
    """Trace using torch.fx, keeping `leaf_types` as call_module ops."""

    _require_torch()
    import torch.fx as fx

    class LeafTracer(fx.Tracer):
        def is_leaf_module(self, m, module_qualified_name: str) -> bool:
            if isinstance(m, leaf_types):
                return True
            return super().is_leaf_module(m, module_qualified_name)

    tracer = LeafTracer()
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)


@dataclass
class FXChainOp:
    op_name: str
    attrs: dict[str, Any]
    module_ref: str | None
    node: Any


def is_linear_chain(gm) -> bool:
    """True iff gm is placeholder -> call_module* -> output."""

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


def fx_chain_to_ops(gm, lora_inject_type, linear_type) -> list[FXChainOp]:
    """Extract a linear chain into `FXChainOp`s."""

    assert is_linear_chain(gm), "FX graph is not a supported linear chain"

    modules = dict(gm.named_modules())
    nodes = list(gm.graph.nodes)
    placeholder = next(n for n in nodes if n.op == "placeholder")

    ops: list[FXChainOp] = []
    cur = placeholder

    while True:
        nxt = next(iter(cur.users.keys()))
        if nxt.op == "output":
            break

        target = str(nxt.target)
        mod = modules[target]
        op_name = type(mod).__name__

        attrs: dict[str, Any] = {}
        if isinstance(mod, lora_inject_type):
            attrs["deltas"] = mod.deltas
        elif isinstance(mod, linear_type):
            attrs["in_features"] = mod.in_features
            attrs["out_features"] = mod.out_features
            attrs["bias"] = (mod.bias is not None)

        ops.append(FXChainOp(op_name=op_name, attrs=attrs, module_ref=target, node=nxt))
        cur = nxt

    return ops


def ops_to_expr(ops: list[FXChainOp], sig: Signature, tensor_obj: Obj) -> Expr:
    """Convert chain ops into a right-associated `Seq` of `Box`es."""

    expr: Expr | None = None

    for op in ops:
        _ = sig.get(op.op_name)
        box = Box.with_attrs(op.op_name, **op.attrs) if op.attrs else Box(op.op_name)
        expr = box if expr is None else normalize(Seq(expr, box))

    return expr if expr is not None else Id(tensor_obj)


def flatten_seq_boxes(e: Expr) -> list[Box]:
    """Flatten (right-associated) Seq into a list of Boxes."""

    e = normalize(e)
    out: list[Box] = []

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


def expr_to_sequential_module(best: Expr, original, lora_inject_type, linear_type):
    """Build nn.Sequential for the supported subset.

    - LoRAInject stages (with fused deltas)
    - Linear (copied from original)

    Parameters are transferred by deep-copying the first matching Linear.
    """

    _require_torch()
    import copy

    import torch.nn as nn

    boxes = flatten_seq_boxes(best)

    orig_linear: nn.Linear | None = None
    for m in original.modules():
        if isinstance(m, linear_type):
            orig_linear = m
            break

    layers: list[nn.Module] = []

    for b in boxes:
        if b.op == lora_inject_type.__name__:
            d = dict(b.attrs).get("deltas", ())
            layers.append(lora_inject_type(deltas=d if isinstance(d, tuple) else ()))
        elif b.op == linear_type.__name__:
            if orig_linear is None:
                raise ValueError("Original model has no Linear to copy")
            layers.append(copy.deepcopy(orig_linear))
        else:
            raise ValueError(f"Cannot rebuild op '{b.op}' in MVP backend")

    return nn.Sequential(*layers)
