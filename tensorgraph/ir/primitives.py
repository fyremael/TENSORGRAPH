"""
TENSORGRAPH v0.2.0: Structural Primitives for DAG Support

These primitives extend the typed IR with structural morphisms
required for representing non-linear (DAG) computational graphs.

- `Dup`: Duplicates a signal (fan-out). Type: A -> A ⊗ A
- `Del`: Discards a signal (fan-in to void). Type: A -> I
- `Let`: Binding construct for shared subexpressions.
- `Case`: Conditional branching. Type: (I + A) -> B (via pair of morphisms)
- `Iter`: Bounded iteration. Type: A -> A (via endomorphism)

These are essential for correctly representing `torch.fx` DAGs where
a single node may have multiple consumers (requires `Dup`), or
explicit discarding of unused outputs (requires `Del`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import Obj
from .expr import Expr


# -----------------------------------------------------------------------------
# Structural Morphisms (v0.2.0)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Dup(Expr):
    """Duplication: A -> A ⊗ A.
    
    Represents fan-out where a single wire is used by multiple consumers.
    This is the categorical 'diagonal' morphism for Cartesian structure.
    """
    obj: Obj


@dataclass(frozen=True)
class Del(Expr):
    """Deletion: A -> I (unit object).
    
    Represents discarding a signal. The 'counit' for Cartesian structure.
    Used when an FX node output is not consumed by any subsequent node.
    """
    obj: Obj


@dataclass(frozen=True)
class Swap(Expr):
    """Symmetric braiding: A ⊗ B -> B ⊗ A.
    
    Represents the crossing of wires. Essential for DAG routing.
    """
    left: Obj
    right: Obj


@dataclass(frozen=True)
class Let(Expr):
    """Binding construct: let x = f in g.
    
    Type: if f : A -> B and g : (scope with B) -> C, then Let(f, g) : A -> C.
    The 'body' is a function that takes a variable and returns an Expr.
    
    Implementation Note: This is NOT a true higher-order let; the binding
    is resolved during FX import into a Dup + explicit wiring.
    """
    binding: Expr  # The expression to bind
    name: str      # Binding identifier (for debugging/tracing)
    body: Expr     # The expression using the binding (with placeholder)


# -----------------------------------------------------------------------------
# Control Flow Primitives (v0.2.0)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Case(Expr):
    """Conditional branching: (I + A) -> B.
    
    Represents a conditional selection. If the input is from the left
    injection (I), returns `left_branch`. If from right (A), applies `right_branch`.
    
    For torch.fx, this maps to `if` nodes or `where` operations.
    """
    left_branch: Expr   # Type: I -> B (constant/identity)
    right_branch: Expr  # Type: A -> B


@dataclass(frozen=True)
class Iter(Expr):
    """Bounded Iteration: A -> A.
    
    Represents a bounded loop applying `body` endomorphism `count` times.
    
    Equivalent to:
        Seq(body, Seq(body, Seq(body, ...)))  # count times
    
    Useful for representing unrolled loops or repeated application.
    """
    body: Expr   # Type: A -> A (endomorphism)
    count: int   # Number of iterations (statically known)


# -----------------------------------------------------------------------------
# Pretty Printing Extensions
# -----------------------------------------------------------------------------


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
        return f"case(left, right)"
    if isinstance(e, Iter):
        return f"iter^{e.count}(body)"
    return repr(e)
