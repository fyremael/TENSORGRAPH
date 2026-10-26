from __future__ import annotations

from ..signature import Signature
from ..types import Sort, Obj
from .expr import Box, Expr, Id, Par, Seq, pretty
from .primitives import Dup, Del, Swap, Case, Iter


def normalize(e: Expr) -> Expr:
    """Lightweight coherence discipline.

    - Eliminates identities in Seq
    - Right-associates Seq for canonical form
    - Recursively normalizes structural primitives
    """
    if isinstance(e, (Id, Box, Dup, Del, Swap)):
        return e
    if isinstance(e, Par):
        return Par(normalize(e.left), normalize(e.right))
    if isinstance(e, Seq):
        terms: list[Expr] = []
        stack: list[Expr] = [e]
        while stack:
            curr = stack.pop()
            if isinstance(curr, Seq):
                stack.append(curr.second)
                stack.append(curr.first)
            else:
                norm_curr = normalize(curr)
                if not isinstance(norm_curr, Id):
                    terms.append(norm_curr)

        if not terms:
            return normalize(e.first)

        res = terms[-1]
        for t in reversed(terms[:-1]):
            res = Seq(t, res)
        return res
    if isinstance(e, Case):
        return Case(normalize(e.left_branch), normalize(e.right_branch))
    if isinstance(e, Iter):
        return Iter(normalize(e.body), e.count)
    raise TypeError(type(e))


def infer_type(e: Expr, sig: Signature) -> Sort:
    """Infer the (dom, cod) sort of an expression under a signature."""

    if isinstance(e, Id):
        return (e.obj, e.obj)
    if isinstance(e, Box):
        op = sig.get(e.op)
        return (op.dom, op.cod)
    if isinstance(e, Par):
        d1, c1 = infer_type(e.left, sig)
        d2, c2 = infer_type(e.right, sig)
        return (d1 @ d2, c1 @ c2)
    if isinstance(e, Seq):
        d1, c1 = infer_type(e.first, sig)
        d2, c2 = infer_type(e.second, sig)
        if c1 != d2:
            raise TypeError(f"Seq type mismatch: {c1} != {d2}\n{pretty(e)}")
        return (d1, c2)
    if isinstance(e, Dup):
        return (e.obj, e.obj @ e.obj)
    if isinstance(e, Del):
        return (e.obj, Obj("I"))
    if isinstance(e, Swap):
        return (e.left @ e.right, e.right @ e.left)
    if isinstance(e, Case):
        dl, cl = infer_type(e.left_branch, sig)
        dr, cr = infer_type(e.right_branch, sig)
        # Type check: left dom must be I, codomains must match
        if dl != Obj("I"):
            raise TypeError(f"Case left branch must have domain I, got {dl}")
        if cl != cr:
            raise TypeError(f"Case branches must have same codomain, got {cl} and {cr}")
        return (Obj("I") + dr, cl)
    if isinstance(e, Iter):
        dom, cod = infer_type(e.body, sig)
        if dom != cod:
            raise TypeError(f"Iter requires endomorphism: {dom} -> {cod}")
        return (dom, cod)
    raise TypeError(type(e))
