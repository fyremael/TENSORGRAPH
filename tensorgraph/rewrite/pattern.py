from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..types import Obj, ObjLike, ObjVar, Sort

if TYPE_CHECKING:  # pragma: no cover
    from ..egraph.egraph import EGraph


# -----------------------------------------------------------------------------
# Pattern language
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """Base class for match patterns."""


@dataclass(frozen=True)
class PVar(Pattern):
    name: str
    sort: Sort | None = None


@dataclass(frozen=True)
class PId(Pattern):
    obj: ObjLike


@dataclass(frozen=True)
class PBox(Pattern):
    op: str
    attrs: tuple[tuple[str, Any], ...] | None = None


@dataclass(frozen=True)
class PSeq(Pattern):
    a: Pattern
    b: Pattern


@dataclass(frozen=True)
class PPar(Pattern):
    l: Pattern
    r: Pattern


@dataclass(frozen=True)
class PIter(Pattern):
    body: Pattern
    count: int | str  # Exact match or capture variable name


Subst = dict[str, int]
ObjSubst = dict[str, Obj]
DataSubst = dict[str, Any]


def match_obj(pat: ObjLike, obj: Obj, oenv: ObjSubst) -> bool:
    """Match an object pattern against a concrete object, updating oenv."""

    if isinstance(pat, ObjVar):
        if pat.name in oenv:
            return oenv[pat.name] == obj
        oenv[pat.name] = obj
        return True
    return pat == obj


def ematch_at(
    eg: EGraph,
    eclass: int,
    pat: Pattern,
    env: Subst,
    oenv: ObjSubst,
    denv: DataSubst | None = None,
) -> list[tuple[Subst, ObjSubst, DataSubst]]:
    """Match `pat` at `eclass`, producing all consistent environments."""

    if denv is None:
        denv = {}

    eclass = eg.uf.find(eclass)

    if isinstance(pat, PVar):
        if pat.sort is not None and eg.sort[eclass] != pat.sort:
            return []
        if pat.name in env:
            return [(env, oenv, denv)] if eg.uf.find(env[pat.name]) == eclass else []
        new_env = dict(env)
        new_env[pat.name] = eclass
        return [(new_env, oenv, denv)]

    outs: list[tuple[Subst, ObjSubst, DataSubst]] = []

    for en in eg.nodes[eclass]:
        if isinstance(pat, PId):
            if en.tag != "Id":
                continue
            (obj,) = en.data
            new_oenv = dict(oenv)
            if match_obj(pat.obj, obj, new_oenv):
                outs.append((dict(env), new_oenv, dict(denv)))

        elif isinstance(pat, PBox):
            if en.tag != "Box":
                continue
            op, attrs = en.data
            if op != pat.op:
                continue
            if pat.attrs is not None and attrs != pat.attrs:
                continue
            outs.append((dict(env), dict(oenv), dict(denv)))

        elif isinstance(pat, PSeq):
            if en.tag != "Seq" or len(en.children) != 2:
                continue
            left, right = en.children
            for env1, oenv1, denv1 in ematch_at(eg, left, pat.a, env, oenv, denv):
                outs.extend(ematch_at(eg, right, pat.b, env1, oenv1, denv1))

        elif isinstance(pat, PPar):
            if en.tag != "Par" or len(en.children) != 2:
                continue
            left, right = en.children
            for env1, oenv1, denv1 in ematch_at(eg, left, pat.l, env, oenv, denv):
                outs.extend(ematch_at(eg, right, pat.r, env1, oenv1, denv1))

        elif isinstance(pat, PIter):
            if en.tag != "Iter" or len(en.children) != 1:
                continue
            (body_id,) = en.children
            (count,) = en.data
            
            # Match count
            new_denv = dict(denv)
            if isinstance(pat.count, int):
                if count != pat.count: continue
            elif isinstance(pat.count, str):
                if pat.count in new_denv:
                    if new_denv[pat.count] != count: continue
                else:
                    new_denv[pat.count] = count
            
            # Match body
            outs.extend(ematch_at(eg, body_id, pat.body, env, oenv, new_denv))

        else:
            raise TypeError(type(pat))

    return outs


def ematch(eg: EGraph, pat: Pattern) -> list[tuple[int, Subst, ObjSubst, DataSubst]]:
    """Find all matches of `pat` anywhere in the e-graph."""

    matches: list[tuple[int, Subst, ObjSubst, DataSubst]] = []
    for rep in sorted(eg.nodes.keys()):
        # Note: We start with empty dicts
        outs = ematch_at(eg, rep, pat, {}, {}, {})
        for env, oenv, denv in outs:
            matches.append((rep, env, oenv, denv))
    return matches
