from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import Obj

# -----------------------------------------------------------------------------
# Diagram terms (1-morphisms)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """Base class for diagram terms."""


@dataclass(frozen=True)
class Id(Expr):
    obj: Obj


@dataclass(frozen=True)
class Box(Expr):
    op: str
    attrs: tuple[tuple[str, Any], ...] = ()

    @staticmethod
    def with_attrs(op: str, **attrs: Any) -> Box:
        """Create a `Box` with stable, hashable attrs."""
        return Box(op=op, attrs=tuple(sorted(attrs.items())))


@dataclass(frozen=True)
class Seq(Expr):
    """Sequential composition: first ; second."""

    first: Expr
    second: Expr


@dataclass(frozen=True)
class Par(Expr):
    """Parallel composition: left ⊗ right."""

    left: Expr
    right: Expr


def pretty(e: Expr) -> str:
    """Human-friendly string representation."""
    if isinstance(e, Id):
        return f"Id[{e.obj}]"
    if isinstance(e, Box):
        if e.attrs:
            return f"{e.op}({', '.join(f'{k}={v}' for k, v in e.attrs)})"
        return e.op
    if isinstance(e, Seq):
        return f"({pretty(e.first)} ; {pretty(e.second)})"
    if isinstance(e, Par):
        return f"({pretty(e.left)} ⊗ {pretty(e.right)})"
    return repr(e)
