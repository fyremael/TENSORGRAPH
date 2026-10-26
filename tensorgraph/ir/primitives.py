"""Structural and bounded control-flow primitives for TENSORGRAPH."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Obj
from .expr import Expr


@dataclass(frozen=True)
class Dup(Expr):
    """Cartesian duplication: ``A -> A ⊗ A``."""

    obj: Obj


@dataclass(frozen=True)
class Del(Expr):
    """Cartesian deletion: ``A -> I``."""

    obj: Obj


@dataclass(frozen=True)
class Swap(Expr):
    """Symmetric braiding: ``A ⊗ B -> B ⊗ A``."""

    left: Obj
    right: Obj


@dataclass(frozen=True)
class Let(Expr):
    """Research binding marker resolved by frontends into explicit wiring."""

    binding: Expr
    name: str
    body: Expr


@dataclass(frozen=True)
class Case(Expr):
    """Conditional branching from ``I + A`` to a common codomain ``B``."""

    left_branch: Expr
    right_branch: Expr


@dataclass(frozen=True)
class Iter(Expr):
    """Statically bounded iteration of an endomorphism."""

    body: Expr
    count: int

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("Iter count must be a statically known integer")
        if self.count < 0:
            raise ValueError("Iter count must be non-negative")


def pretty_structural(e: Expr) -> str:
    """Extended pretty printer for structural primitives."""

    if isinstance(e, Dup):
        return f"Δ[{e.obj}]"
    if isinstance(e, Del):
        return f"ε[{e.obj}]"
    if isinstance(e, Swap):
        return f"σ[{e.left},{e.right}]"
    if isinstance(e, Let):
        return f"let {e.name} = ... in ..."
    if isinstance(e, Case):
        return "case(left, right)"
    if isinstance(e, Iter):
        return f"iter^{e.count}(body)"
    return repr(e)
