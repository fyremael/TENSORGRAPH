from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..egraph.egraph import EGraph
from ..egraph.enode import ENode
from ..types import ObjVar
from .pattern import (
    ObjSubst,
    Pattern,
    PBox,
    PId,
    PIter,
    PPar,
    PSeq,
    PVar,
    Subst,
    DataSubst,
)


@dataclass(frozen=True)
class Rewrite:
    """A typed rewrite law.

    `lhs` is a Pattern.
    `rhs` is either:
      - a Pattern (instantiable via environment), or
      - a builder function (eg, for attribute computations).

    The builder signature is:
        (eg, root_eclass, env_expr, env_obj, env_data) -> eclass_id
    """

    name: str
    lhs: Pattern
    rhs: Pattern | Callable[[EGraph, int, Subst, ObjSubst, DataSubst], int]
    origin: str | None = None


def instantiate_pattern(
    eg: EGraph, pat: Pattern, env: Subst, oenv: ObjSubst, denv: DataSubst | None = None
) -> int:
    """Instantiate a RHS pattern into the e-graph and return its e-class id."""
    
    if denv is None:
        denv = {}

    if isinstance(pat, PVar):
        if pat.name not in env:
            raise KeyError(f"Unbound pattern var {pat.name}")
        return eg.uf.find(env[pat.name])

    if isinstance(pat, PId):
        if isinstance(pat.obj, ObjVar):
            if pat.obj.name not in oenv:
                raise KeyError(f"Unbound object var {pat.obj.name}")
            obj = oenv[pat.obj.name]
        else:
            obj = pat.obj
        return eg.add_enode(ENode("Id", (obj,), ()), (obj, obj))

    if isinstance(pat, PBox):
        opdef = eg.sig.get(pat.op)
        attrs = pat.attrs if pat.attrs is not None else ()
        return eg.add_enode(ENode("Box", (pat.op, attrs), ()), (opdef.dom, opdef.cod))

    if isinstance(pat, PPar):
        l = instantiate_pattern(eg, pat.l, env, oenv, denv)
        r = instantiate_pattern(eg, pat.r, env, oenv, denv)
        dl, cl = eg.sort[eg.uf.find(l)]
        dr, cr = eg.sort[eg.uf.find(r)]
        return eg.add_enode(ENode("Par", (), (l, r)), (dl @ dr, cl @ cr))

    if isinstance(pat, PSeq):
        a = instantiate_pattern(eg, pat.a, env, oenv, denv)
        b = instantiate_pattern(eg, pat.b, env, oenv, denv)
        da, ca = eg.sort[eg.uf.find(a)]
        db, cb = eg.sort[eg.uf.find(b)]
        if ca != db:
            raise TypeError(f"instantiate RHS Seq mismatch: {ca} != {db}")
        return eg.add_enode(ENode("Seq", (), (a, b)), (da, cb))

    if isinstance(pat, PIter):
        body = instantiate_pattern(eg, pat.body, env, oenv, denv)
        dom, cod = eg.sort[eg.uf.find(body)]
        
        count = pat.count
        if isinstance(count, str):
            if count not in denv:
                raise KeyError(f"Unbound data var {count}")
            count = denv[count] # Should be int
            
        return eg.add_enode(ENode("Iter", (count,), (body,)), (dom, cod))

    raise TypeError(type(pat))
