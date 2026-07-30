"""tensorgraph_eg.py

TensorGraph v2: An operational 2D-category-theory machine using e-graphs.

Key capabilities
- Objects (types): Obj with tensor product
- 1-morphisms: Expr (Id, Box, Seq, Par)
- 2-morphisms: Rewrite rules as typed equalities
- Coherent reasoning: normalization (lightweight) + equality saturation
- Structured laws up to equivalence: e-graph (many equivalents at once)
- Translation across interfaces: adjunction mates as rewrite synthesis

This is intentionally self-contained and meant to be extended.

Run:
  python tensorgraph_eg.py

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# -----------------------------------------------------------------------------
# 0) Objects (types)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Obj:
    name: str
    left: Optional["Obj"] = None
    right: Optional["Obj"] = None

    @staticmethod
    def tensor(a: "Obj", b: "Obj") -> "Obj":
        return Obj(name="⊗", left=a, right=b)

    def __matmul__(self, other: "Obj") -> "Obj":
        return Obj.tensor(self, other)

    def is_tensor(self) -> bool:
        return self.name == "⊗" and self.left is not None and self.right is not None

    def __str__(self) -> str:
        if self.is_tensor():
            return f"({self.left} ⊗ {self.right})"
        return self.name


@dataclass(frozen=True)
class ObjVar:
    """Pattern variable that can match an Obj."""
    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


ObjLike = Union[Obj, ObjVar]
Sort = Tuple[Obj, Obj]  # (dom, cod)


# -----------------------------------------------------------------------------
# 1) Signature (primitive boxes)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class OpDef:
    name: str
    dom: Obj
    cod: Obj


class Signature:
    def __init__(self) -> None:
        self._ops: Dict[str, OpDef] = {}

    def add(self, name: str, dom: Obj, cod: Obj) -> None:
        if name in self._ops:
            raise ValueError(f"Op '{name}' already exists")
        self._ops[name] = OpDef(name=name, dom=dom, cod=cod)

    def get(self, name: str) -> OpDef:
        if name not in self._ops:
            raise KeyError(f"Unknown op '{name}'")
        return self._ops[name]


# -----------------------------------------------------------------------------
# 2) 1-morphisms: Expr (string-diagram terms)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Id(Expr):
    obj: Obj


@dataclass(frozen=True)
class Box(Expr):
    op: str
    attrs: Tuple[Tuple[str, Any], ...] = ()

    @staticmethod
    def with_attrs(op: str, **attrs: Any) -> "Box":
        return Box(op=op, attrs=tuple(sorted(attrs.items())))


@dataclass(frozen=True)
class Seq(Expr):
    first: Expr
    second: Expr


@dataclass(frozen=True)
class Par(Expr):
    left: Expr
    right: Expr


def pretty(e: Expr) -> str:
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


def normalize(e: Expr) -> Expr:
    """Light coherence: kill identities, right-associate Seq."""
    if isinstance(e, (Id, Box)):
        return e
    if isinstance(e, Par):
        return Par(normalize(e.left), normalize(e.right))
    if isinstance(e, Seq):
        a = normalize(e.first)
        b = normalize(e.second)
        if isinstance(a, Id):
            return b
        if isinstance(b, Id):
            return a
        if isinstance(a, Seq):
            return normalize(Seq(a.first, Seq(a.second, b)))
        return Seq(a, b)
    raise TypeError(type(e))


def infer_type(e: Expr, sig: Signature) -> Sort:
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
    raise TypeError(type(e))


# -----------------------------------------------------------------------------
# 3) E-graph core
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ENode:
    tag: str
    data: Tuple[Any, ...]
    children: Tuple[int, ...]


class UnionFind:
    def __init__(self) -> None:
        self.parent: List[int] = []
        self.rank: List[int] = []

    def make(self) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.rank.append(0)
        return i

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class EGraph:
    """A typed e-graph: each e-class has a fixed (dom,cod) sort."""

    def __init__(self, sig: Signature) -> None:
        self.sig = sig
        self.uf = UnionFind()

        self.sort: Dict[int, Sort] = {}              # rep -> sort
        self.nodes: Dict[int, Set[ENode]] = {}       # rep -> set of enodes

        self.memo: Dict[ENode, int] = {}             # canonical enode -> rep

        self.root: Optional[int] = None

        # Optional: very lightweight explanation log (not a full proof term)
        self.merge_log: List[Tuple[str, int, int]] = []  # (reason, a_rep, b_rep)

    # -------- creation / adding --------

    def _new_class(self, sort: Sort) -> int:
        cid = self.uf.make()
        self.sort[cid] = sort
        self.nodes[cid] = set()
        return cid

    def add_enode(self, en: ENode, sort: Sort) -> int:
        # Canonicalize children
        canon_children = tuple(self.uf.find(c) for c in en.children)
        en = ENode(en.tag, en.data, canon_children)

        if en in self.memo:
            cid = self.uf.find(self.memo[en])
            # sanity check
            if self.sort[cid] != sort:
                raise TypeError(f"ENode sort clash: {en} sort {sort} vs existing {self.sort[cid]}")
            return cid

        cid = self._new_class(sort)
        self.nodes[cid].add(en)
        self.memo[en] = cid
        return cid

    def add_expr(self, e: Expr) -> int:
        e = normalize(e)

        if isinstance(e, Id):
            sort = (e.obj, e.obj)
            return self.add_enode(ENode("Id", (e.obj,), ()), sort)

        if isinstance(e, Box):
            opdef = self.sig.get(e.op)
            sort = (opdef.dom, opdef.cod)
            data = (e.op, e.attrs)
            return self.add_enode(ENode("Box", data, ()), sort)

        if isinstance(e, Par):
            l = self.add_expr(e.left)
            r = self.add_expr(e.right)
            dl, cl = self.sort[self.uf.find(l)]
            dr, cr = self.sort[self.uf.find(r)]
            sort = (dl @ dr, cl @ cr)
            return self.add_enode(ENode("Par", (), (l, r)), sort)

        if isinstance(e, Seq):
            a = self.add_expr(e.first)
            b = self.add_expr(e.second)
            da, ca = self.sort[self.uf.find(a)]
            db, cb = self.sort[self.uf.find(b)]
            if ca != db:
                raise TypeError(f"Seq type mismatch in add_expr: {ca} != {db}")
            sort = (da, cb)
            return self.add_enode(ENode("Seq", (), (a, b)), sort)

        raise TypeError(type(e))

    # -------- merge / rebuild --------

    def merge(self, a: int, b: int, reason: str = "") -> int:
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        if self.sort[ra] != self.sort[rb]:
            raise TypeError(f"Cannot merge e-classes with different sorts: {self.sort[ra]} vs {self.sort[rb]}")

        new_rep = self.uf.union(ra, rb)
        old_rep = rb if new_rep == ra else ra

        # Move nodes and delete old
        self.nodes[new_rep].update(self.nodes[old_rep])
        del self.nodes[old_rep]

        # sort table follows union-find rep; ensure present at rep
        self.sort[new_rep] = self.sort[ra]
        if old_rep in self.sort:
            del self.sort[old_rep]

        if reason:
            self.merge_log.append((reason, ra, rb))
        return new_rep

    def rebuild(self) -> None:
        """Canonicalize children reps, deduplicate enodes, and enforce congruence."""

        changed = True
        while changed:
            changed = False

            # 1) Canonicalize nodes within each class
            new_memo: Dict[ENode, int] = {}
            new_nodes: Dict[int, Set[ENode]] = {}
            new_sort: Dict[int, Sort] = {}

            # map old reps to current reps
            reps = sorted({self.uf.find(cid) for cid in list(self.nodes.keys())})

            for rep in reps:
                # rep might have been deleted; skip
                if rep not in self.nodes:
                    continue

                new_nodes[rep] = set()
                new_sort[rep] = self.sort[rep]

                for en in self.nodes[rep]:
                    canon_children = tuple(self.uf.find(c) for c in en.children)
                    canon = ENode(en.tag, en.data, canon_children)
                    new_nodes[rep].add(canon)

            # 2) Congruence: same enode shape -> same class
            for rep, ens in list(new_nodes.items()):
                for en in list(ens):
                    if en in new_memo:
                        rep2 = self.uf.find(new_memo[en])
                        rep1 = self.uf.find(rep)
                        if rep1 != rep2:
                            self.merge(rep1, rep2, reason="congruence")
                            changed = True
                    else:
                        new_memo[en] = self.uf.find(rep)

            # Refresh tables after merges
            # Recompute node sets for current reps
            if changed:
                # Rebuild nodes dict from scratch using current union-find
                merged_nodes: Dict[int, Set[ENode]] = {}
                merged_sort: Dict[int, Sort] = {}
                for rep, ens in new_nodes.items():
                    r = self.uf.find(rep)
                    merged_nodes.setdefault(r, set()).update(ens)
                for rep in merged_nodes.keys():
                    # Any member rep sort should be consistent; take from any existing
                    # Find one original representative that still exists in sort.
                    merged_sort[rep] = self.sort.get(rep, next(iter(new_sort.values())))

                self.nodes = merged_nodes
                self.sort = merged_sort

            else:
                # No congruence merges; accept canonical tables
                self.nodes = new_nodes
                self.sort = new_sort
                self.memo = new_memo


# -----------------------------------------------------------------------------
# 4) Patterns and e-matching
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    pass


@dataclass(frozen=True)
class PVar(Pattern):
    name: str
    sort: Optional[Sort] = None  # optional type constraint


@dataclass(frozen=True)
class PId(Pattern):
    obj: ObjLike


@dataclass(frozen=True)
class PBox(Pattern):
    op: str
    # attrs can be None (any) or exact tuple match
    attrs: Optional[Tuple[Tuple[str, Any], ...]] = None


@dataclass(frozen=True)
class PSeq(Pattern):
    a: Pattern
    b: Pattern


@dataclass(frozen=True)
class PPar(Pattern):
    l: Pattern
    r: Pattern


Subst = Dict[str, int]
ObjSubst = Dict[str, Obj]


def match_obj(pat: ObjLike, obj: Obj, oenv: ObjSubst) -> bool:
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
) -> List[Tuple[Subst, ObjSubst]]:
    """Return all environments extending (env,oenv) that match pat at eclass."""

    eclass = eg.uf.find(eclass)

    # Variable: bind to eclass
    if isinstance(pat, PVar):
        if pat.sort is not None and eg.sort[eclass] != pat.sort:
            return []
        if pat.name in env:
            return [(env, oenv)] if eg.uf.find(env[pat.name]) == eclass else []
        new_env = dict(env)
        new_env[pat.name] = eclass
        return [(new_env, oenv)]

    # Need an enode match at this class
    outs: List[Tuple[Subst, ObjSubst]] = []
    for en in eg.nodes[eclass]:
        if isinstance(pat, PId):
            if en.tag != "Id":
                continue
            (obj,) = en.data
            new_oenv = dict(oenv)
            if match_obj(pat.obj, obj, new_oenv):
                outs.append((dict(env), new_oenv))

        elif isinstance(pat, PBox):
            if en.tag != "Box":
                continue
            op, attrs = en.data
            if op != pat.op:
                continue
            if pat.attrs is not None and attrs != pat.attrs:
                continue
            outs.append((dict(env), dict(oenv)))

        elif isinstance(pat, PSeq):
            if en.tag != "Seq" or len(en.children) != 2:
                continue
            left, right = en.children
            for env1, oenv1 in ematch_at(eg, left, pat.a, env, oenv):
                outs.extend(ematch_at(eg, right, pat.b, env1, oenv1))

        elif isinstance(pat, PPar):
            if en.tag != "Par" or len(en.children) != 2:
                continue
            left, right = en.children
            for env1, oenv1 in ematch_at(eg, left, pat.l, env, oenv):
                outs.extend(ematch_at(eg, right, pat.r, env1, oenv1))

        else:
            raise TypeError(type(pat))

    return outs


def ematch(eg: EGraph, pat: Pattern) -> List[Tuple[int, Subst, ObjSubst]]:
    """Find all matches of `pat` anywhere in the e-graph."""
    matches: List[Tuple[int, Subst, ObjSubst]] = []
    reps = sorted(eg.nodes.keys())
    for rep in reps:
        env0: Subst = {}
        oenv0: ObjSubst = {}
        outs = ematch_at(eg, rep, pat, env0, oenv0)
        for env, oenv in outs:
            matches.append((rep, env, oenv))
    return matches


# -----------------------------------------------------------------------------
# 5) Rewrites (2-morphisms) and saturation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Rewrite:
    name: str
    lhs: Pattern
    # rhs can either be a Pattern with variables, or a builder producing an e-class
    rhs: Union[Pattern, Callable[[EGraph, int, Subst, ObjSubst], int]]


def instantiate_pattern(eg: EGraph, pat: Pattern, env: Subst, oenv: ObjSubst) -> int:
    """Instantiate a RHS pattern into the e-graph and return its e-class id."""

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
        l = instantiate_pattern(eg, pat.l, env, oenv)
        r = instantiate_pattern(eg, pat.r, env, oenv)
        dl, cl = eg.sort[eg.uf.find(l)]
        dr, cr = eg.sort[eg.uf.find(r)]
        return eg.add_enode(ENode("Par", (), (l, r)), (dl @ dr, cl @ cr))

    if isinstance(pat, PSeq):
        a = instantiate_pattern(eg, pat.a, env, oenv)
        b = instantiate_pattern(eg, pat.b, env, oenv)
        da, ca = eg.sort[eg.uf.find(a)]
        db, cb = eg.sort[eg.uf.find(b)]
        if ca != db:
            raise TypeError(f"instantiate RHS Seq mismatch: {ca} != {db}")
        return eg.add_enode(ENode("Seq", (), (a, b)), (da, cb))

    raise TypeError(type(pat))


def saturate(
    eg: EGraph,
    rewrites: Sequence[Rewrite],
    iters: int = 8,
    max_applications: int = 10_000,
) -> None:
    """Equality saturation loop."""

    for i in range(iters):
        applied = 0
        for rw in rewrites:
            matches = ematch(eg, rw.lhs)
            for root, env, oenv in matches:
                if applied >= max_applications:
                    break

                if isinstance(rw.rhs, Pattern):
                    rhs_id = instantiate_pattern(eg, rw.rhs, env, oenv)
                else:
                    rhs_id = rw.rhs(eg, root, env, oenv)

                eg.merge(root, rhs_id, reason=rw.name)
                applied += 1

            if applied >= max_applications:
                break

        eg.rebuild()
        if applied == 0:
            break


# -----------------------------------------------------------------------------
# 6) Extraction (pick best representative expression)
# -----------------------------------------------------------------------------

def cost_of_enode(en: ENode) -> int:
    return 1 if en.tag == "Box" else 0


class Extractor:
    def __init__(self, eg: EGraph) -> None:
        self.eg = eg
        self.best_cost: Dict[int, int] = {}
        self.best_node: Dict[int, ENode] = {}

    def solve(self, root: int, max_rounds: int = 50) -> None:
        eg = self.eg
        reps = list(eg.nodes.keys())

        # init
        INF = 10**18
        for r in reps:
            self.best_cost[r] = INF

        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            rounds += 1
            changed = False
            for r in reps:
                r = eg.uf.find(r)
                for en in eg.nodes[r]:
                    # if children unknown, skip
                    child_costs = 0
                    ok = True
                    for c in en.children:
                        c = eg.uf.find(c)
                        cc = self.best_cost.get(c, INF)
                        if cc >= INF:
                            ok = False
                            break
                        child_costs += cc
                    if not ok:
                        # Try allowing leaf nodes
                        if en.children:
                            continue
                        child_costs = 0

                    cand = cost_of_enode(en) + child_costs
                    if cand < self.best_cost[r]:
                        self.best_cost[r] = cand
                        self.best_node[r] = en
                        changed = True

        # ensure root reachable
        root = eg.uf.find(root)
        if root not in self.best_node:
            # fallback: pick any node
            self.best_node[root] = next(iter(eg.nodes[root]))
            self.best_cost[root] = 0

    def extract(self, root: int) -> Expr:
        eg = self.eg
        root = eg.uf.find(root)

        visiting: Set[int] = set()

        def build(r: int) -> Expr:
            r = eg.uf.find(r)
            if r in visiting:
                # cycle breaker: choose something cheap-ish
                return Box("_cycle")
            visiting.add(r)

            en = self.best_node.get(r)
            if en is None:
                en = next(iter(eg.nodes[r]))

            if en.tag == "Id":
                (obj,) = en.data
                out = Id(obj)
            elif en.tag == "Box":
                op, attrs = en.data
                out = Box(op, attrs)
            elif en.tag == "Par":
                out = Par(build(en.children[0]), build(en.children[1]))
            elif en.tag == "Seq":
                out = Seq(build(en.children[0]), build(en.children[1]))
            else:
                out = Box("_unknown")

            visiting.remove(r)
            return normalize(out)

        return build(root)


# -----------------------------------------------------------------------------
# 7) Adjunction mates (rewrite synthesis)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Adjunction:
    """Operational adjunction f ⊣ g used to synthesize mate rewrites."""
    f_lower: Box
    g_lift: Box

    def mate_left_to_right(self, alpha: Rewrite) -> Rewrite:
        """If alpha is (f ; u)  <->  (v ; f), produce mate u <-> (g ; v ; f)."""
        # We enforce the *shape* of alpha.lhs and alpha.rhs patterns.
        if not isinstance(alpha.lhs, PSeq):
            raise ValueError("alpha.lhs must be a Seq pattern")

        # For MVP we require concrete PBox f on LHS prefix.
        if not (isinstance(alpha.lhs.a, PBox) and alpha.lhs.a.op == self.f_lower.op):
            raise ValueError("alpha.lhs must begin with f_lower")

        # RHS must be a Pattern (not builder) with suffix f
        if not isinstance(alpha.rhs, Pattern):
            raise ValueError("alpha.rhs must be a Pattern for mates MVP")
        rhs = alpha.rhs
        if not isinstance(rhs, PSeq):
            raise ValueError("alpha.rhs must be Seq(v, f)")
        if not (isinstance(rhs.b, PBox) and rhs.b.op == self.f_lower.op):
            raise ValueError("alpha.rhs must end with f_lower")

        u = alpha.lhs.b
        v = rhs.a

        mate_lhs = u
        # We want the mate u  <->  g ∘ v ∘ f.
        # In our Seq syntax ("do first, then second"),
        #   g ∘ v ∘ f  means:  f ; v ; g
        mate_rhs = PSeq(PBox(self.f_lower.op), PSeq(v, PBox(self.g_lift.op)))

        return Rewrite(
            name=f"mate({alpha.name})",
            lhs=mate_lhs,
            rhs=mate_rhs,
        )


# -----------------------------------------------------------------------------
# 8) Demo rewrites + run
# -----------------------------------------------------------------------------

def _count_boxes_expr(e: Expr) -> int:
    if isinstance(e, Box):
        return 1
    if isinstance(e, Id):
        return 0
    if isinstance(e, Seq):
        return _count_boxes_expr(e.first) + _count_boxes_expr(e.second)
    if isinstance(e, Par):
        return _count_boxes_expr(e.left) + _count_boxes_expr(e.right)
    return 0


def _get_single_inject_deltas(eg: EGraph, cid: int) -> Tuple[str, ...]:
    """Find InjectLoRA deltas from an eclass (assumes pattern ensured it exists)."""
    cid = eg.uf.find(cid)
    for en in eg.nodes[cid]:
        if en.tag == "Box":
            op, attrs = en.data
            if op == "InjectLoRA":
                d = dict(attrs).get("deltas", ())
                if not isinstance(d, tuple):
                    raise ValueError("deltas must be tuple")
                return d
    raise ValueError("No InjectLoRA node found in e-class")


def make_fuse_injects(sig: Signature) -> Rewrite:
    """Fuse consecutive LoRA injections while preserving the tail.

    Normalization right-associates Seq, so a typical program shape is:

        (inj1 ⊗ Id[X]) ; ((inj2 ⊗ Id[X]) ; tail)

    We rewrite that to:

        (inj12 ⊗ Id[X]) ; tail

    where inj12 merges the deltas of inj1 and inj2.
    """

    lhs = PSeq(
        PPar(PVar("i1"), PId(ObjVar("X"))),
        PSeq(
            PPar(PVar("i2"), PId(ObjVar("X"))),
            PVar("tail"),
        ),
    )

    def rhs_builder(eg: EGraph, root: int, env: Subst, oenv: ObjSubst) -> int:
        i1 = env["i1"]
        i2 = env["i2"]
        tail = env["tail"]
        X = oenv["X"]

        d1 = _get_single_inject_deltas(eg, i1)
        d2 = _get_single_inject_deltas(eg, i2)

        fused = Box.with_attrs("InjectLoRA", deltas=d1 + d2)
        lhs_par = eg.add_expr(Par(fused, Id(X)))

        # Build Seq(lhs_par, tail) by referencing existing e-classes directly.
        lhs_par = eg.uf.find(lhs_par)
        tail = eg.uf.find(tail)
        d1_, c1_ = eg.sort[lhs_par]
        d2_, c2_ = eg.sort[tail]
        if c1_ != d2_:
            # If the tail isn't composable, this match is ill-typed; ignore.
            raise TypeError(f"FuseInjectLoRA RHS Seq mismatch: {c1_} != {d2_}")
        sort = (d1_, c2_)
        return eg.add_enode(ENode("Seq", (), (lhs_par, tail)), sort)

    return Rewrite(name="FuseInjectLoRA", lhs=lhs, rhs=rhs_builder)


def demo_lora_fusion() -> None:
    W = Obj("W")
    X = Obj("X")
    Y = Obj("Y")

    sig = Signature()
    sig.add("InjectLoRA", W, W)
    sig.add("LinearApply", W @ X, Y)

    inj1 = Box.with_attrs("InjectLoRA", deltas=("A1B1",))
    inj2 = Box.with_attrs("InjectLoRA", deltas=("A2B2",))
    lin = Box("LinearApply")

    prog = normalize(
        Seq(
            Seq(
                Par(inj1, Id(X)),
                Par(inj2, Id(X)),
            ),
            lin,
        )
    )

    print("\n=== LoRA fusion demo (e-graph equality saturation) ===")
    print("Original:", pretty(prog))
    print("Boxes:", _count_boxes_expr(prog))
    print("Type:", infer_type(prog, sig))

    eg = EGraph(sig)
    root = eg.add_expr(prog)
    eg.root = root

    rewrites = [
        make_fuse_injects(sig),
        # NOTE: coherence/assoc rules could be added too, but normalize already helps.
    ]

    saturate(eg, rewrites, iters=6)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("Best :", pretty(best))
    print("Boxes:", _count_boxes_expr(best))


def demo_mates_transport() -> None:
    A = Obj("A")
    B = Obj("B")

    sig = Signature()
    sig.add("Lower", A, B)
    sig.add("Lift", B, A)
    sig.add("OptA", A, A)
    sig.add("OptB", B, B)

    # alpha: (Lower ; OptA) <-> (OptB ; Lower)
    alpha = Rewrite(
        name="LowerCommutesWithOpt",
        lhs=PSeq(PBox("Lower"), PBox("OptA")),
        rhs=PSeq(PBox("OptB"), PBox("Lower")),
    )

    adj = Adjunction(f_lower=Box("Lower"), g_lift=Box("Lift"))
    mate = adj.mate_left_to_right(alpha)

    print("\n=== Mates transport demo ===")
    print("alpha.lhs:", alpha.lhs)
    print("mate.lhs :", mate.lhs)

    # Show that mate rule can be applied in an e-graph
    # Start with u = OptA, and saturate with mate: OptA <-> (Lift ; OptB ; Lower)

    eg = EGraph(sig)
    root = eg.add_expr(Box("OptA"))
    eg.root = root

    saturate(eg, [mate], iters=3)

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    print("Extracted best for OptA class:", pretty(best))


if __name__ == "__main__":
    demo_lora_fusion()
    demo_mates_transport()
