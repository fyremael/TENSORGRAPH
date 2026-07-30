"""
TENSORGRAPH Library: Dynamic Control Flow
Defines rewrite rules for optimizing bounded iteration (Iter).
"""
from __future__ import annotations

from typing import Any

from ..egraph.egraph import EGraph
from ..egraph.enode import ENode
from ..ir import Iter, Seq, Box, Id, Par
from ..types import ObjVar
from ..rewrite import Rewrite, PBox, PSeq, PVar, Pattern, PIter, PPar, PId, Subst, ObjSubst, DataSubst


def peel_iter_rhs(eg: EGraph, root: int, env: Subst, oenv: ObjSubst, denv: DataSubst) -> int:
    """RHS builder for Iter(f, n) -> Seq(f, Iter(f, n-1))"""
    n = denv["n"]
    f_class = env["f"]
    
    if n <= 0:
        # Should be Id. Need domain object.
        # root is the Iter eclass. We can get sort from it.
        dom, cod = eg.sort[eg.uf.find(root)]
        # Add Id(dom)
        # We need to construct Id expr or Enode.
        # eg.add_expr(Id(dom)) handles normalization.
        return eg.add_expr(Id(dom))
        
    if n > 0:
        # Construct Iter(f, n-1)
        # We need to add ENode manually or via add_expr if we have Exprs.
        # But f is an EClass ID. We can't wrap EClass ID in Expr (yet).
        # We must uses add_enode.
        
        # Iter(f, n-1)
        dom, cod = eg.sort[eg.uf.find(f_class)]
        # Iter sort is same (dom, dom)
        iter_rest = eg.add_enode(ENode("Iter", (n - 1,), (f_class,)), (dom, cod))
        
        # Seq(f, iter_rest)
        seq_op = eg.add_enode(ENode("Seq", (), (f_class, iter_rest)), (dom, cod))
        
        return seq_op
    
    return root # No change for invalid n?


peel_iter = Rewrite(
    name="peel_iter",
    lhs=PIter(PVar("f"), "n"),
    rhs=peel_iter_rhs,
)

# Unroll 0 -> Id
def unroll_zero_rhs(eg: EGraph, root: int, env: Subst, oenv: ObjSubst, denv: DataSubst) -> int:
    # Explicit rule for 0 case to be cleaner?
    # root is Iter(f, 0).
    dom, cod = eg.sort[eg.uf.find(root)]
    return eg.add_expr(Id(dom))

unroll_zero = Rewrite(
    name="unroll_zero",
    lhs=PIter(PVar("f"), 0),
    rhs=unroll_zero_rhs
)

# Algebra Rules to ensure convergence with normalization
# Seq(x, Id) -> x
# Seq(Id, x) -> x
# Seq(x, Seq(y, z)) <-> Seq(Seq(x, y), z)

right_unit = Rewrite(
    name="right_unit",
    lhs=PSeq(PVar("x"), PId(ObjVar("u"))), # PId matches Id(obj)
    rhs=PVar("x")
)

left_unit = Rewrite(
    name="left_unit",
    lhs=PSeq(PId(ObjVar("u")), PVar("x")),
    rhs=PVar("x")
)

assoc_l = Rewrite(
    name="assoc_l",
    lhs=PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
    rhs=PSeq(PVar("a"), PSeq(PVar("b"), PVar("c")))
)

assoc_r = Rewrite(
    name="assoc_r",
    lhs=PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
    rhs=PSeq(PSeq(PVar("a"), PVar("b")), PVar("c"))
)


# Iter(f, n) ; Iter(f, m) -> Iter(f, n+m)
def iter_fusion_rhs(eg: EGraph, root: int, env: Subst, oenv: ObjSubst, denv: DataSubst) -> int:
    f = env["f"]
    n = denv["n"]
    m = denv["m"]
    
    # We need to find the sort of f to construct Iter
    # root is Seq(Iter(n), Iter(m)).
    # We can get sort from root.
    dom, cod = eg.sort[eg.uf.find(root)]
    
    # Construct Iter(f, n+m)
    iter_node = eg.add_enode(ENode("Iter", (n + m,), (f,)), (dom, cod))
    return iter_node

iter_fusion = Rewrite(
    name="iter_fusion",
    lhs=PSeq(PIter(PVar("f"), "n"), PIter(PVar("f"), "m")),
    rhs=iter_fusion_rhs
)

# Iter(Par(f, g), n) -> Par(Iter(f, n), Iter(g, n))
# Useful for Loop Invariant Code Motion (g=Id)
def iter_product_rhs(eg: EGraph, root: int, env: Subst, oenv: ObjSubst, denv: DataSubst) -> int:
    f = env["f"]
    g = env["g"]
    n = denv["n"]
    
    # root is Iter(Par(f, g)).
    # Sort of f and g?
    # Par(f, g) is child of root.
    # We can lookup Par node?
    # Or just re-infer?
    dom_f, cod_f = eg.sort[eg.uf.find(f)]
    dom_g, cod_g = eg.sort[eg.uf.find(g)]
    
    iter_f = eg.add_enode(ENode("Iter", (n,), (f,)), (dom_f, cod_f))
    iter_g = eg.add_enode(ENode("Iter", (n,), (g,)), (dom_g, cod_g))
    
    # Par(iter_f, iter_g)
    # Result sort: dom_f@dom_g -> cod_f@cod_g
    # This matches root sort.
    return eg.add_enode(ENode("Par", (), (iter_f, iter_g)), (dom_f @ dom_g, cod_f @ cod_g))

iter_product = Rewrite(
    name="iter_product",
    lhs=PIter(PPar(PVar("f"), PVar("g")), "n"),
    rhs=iter_product_rhs
)

# Iter(Id, n) -> Id
iter_id = Rewrite(
    name="iter_id",
    lhs=PIter(PId(ObjVar("u")), "n"), # u is obj
    rhs=PId(ObjVar("u"))
)

ALL_RULES = [
    # unroll_zero, 
    peel_iter,
    iter_fusion,
    iter_product,
    iter_id,
    right_unit,
    left_unit,
    assoc_l,
    assoc_r,
]
