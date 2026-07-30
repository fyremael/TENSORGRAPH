from __future__ import annotations

from collections.abc import Callable

from ..ir import Box, Expr, Id, Par, Seq, normalize
from ..ir.primitives import Case, Del, Dup, Iter, Swap
from ..signature import Signature
from ..types import Obj, Sort
from .enode import ENode
from .unionfind import UnionFind


class EGraph:
    """A typed e-graph with fail-closed node admission.

    Every e-class has one fixed ``(domain, codomain)`` sort. All construction
    paths, including rewrite RHS instantiation, pass through ``add_enode`` and
    the same validation rules.
    """

    def __init__(self, sig: Signature) -> None:
        self.sig = sig
        self.uf = UnionFind()

        self.sort: dict[int, Sort] = {}
        self.nodes: dict[int, set[ENode]] = {}
        self.memo: dict[ENode, int] = {}
        self.parents: dict[int, list[tuple[ENode, int]]] = {}

        self.worklist: list[tuple[ENode, int]] = []
        self.pending: list[int] = []

        self.root: int | None = None
        self.merge_log: list[tuple[str, int, int]] = []
        self.on_merge: list[Callable[[int, int], None]] = []

    def _new_class(self, sort: Sort) -> int:
        cid = self.uf.make()
        self.sort[cid] = sort
        self.nodes[cid] = set()
        self.parents[cid] = []
        return cid

    def _child_sort(self, cid: int) -> Sort:
        return self.sort[self.uf.find(cid)]

    def _validate_enode(self, en: ENode, sort: Sort) -> None:
        """Validate an enode and its declared sort before admission."""

        if en.tag == "Id":
            obj = en.data[0]
            expected = (obj, obj)
        elif en.tag == "Box":
            op = self.sig.get(en.data[0])
            expected = (op.dom, op.cod)
        elif en.tag == "Par":
            if len(en.children) != 2:
                raise TypeError("Par requires exactly two children")
            dl, cl = self._child_sort(en.children[0])
            dr, cr = self._child_sort(en.children[1])
            expected = (dl @ dr, cl @ cr)
        elif en.tag == "Seq":
            if len(en.children) != 2:
                raise TypeError("Seq requires exactly two children")
            da, ca = self._child_sort(en.children[0])
            db, cb = self._child_sort(en.children[1])
            if ca != db:
                raise TypeError(f"Seq type mismatch in add_enode: {ca} != {db}")
            expected = (da, cb)
        elif en.tag == "Dup":
            obj = en.data[0]
            expected = (obj, obj @ obj)
        elif en.tag == "Del":
            obj = en.data[0]
            expected = (obj, Obj("I"))
        elif en.tag == "Swap":
            left, right = en.data
            expected = (left @ right, right @ left)
        elif en.tag == "Case":
            if len(en.children) != 2:
                raise TypeError("Case requires exactly two branch children")
            dl, cl = self._child_sort(en.children[0])
            dr, cr = self._child_sort(en.children[1])
            if dl != Obj("I"):
                raise TypeError(f"Case left branch must have domain I, got {dl}")
            if cl != cr:
                raise TypeError(f"Case branches must have same codomain, got {cl} and {cr}")
            expected = (Obj("I") + dr, cl)
        elif en.tag == "Iter":
            if len(en.children) != 1 or len(en.data) != 1:
                raise TypeError("Iter requires one body child and one static count")
            count = en.data[0]
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("Iter count must be a statically known integer")
            if count < 0:
                raise ValueError("Iter count must be non-negative")
            dom, cod = self._child_sort(en.children[0])
            if dom != cod:
                raise TypeError(f"Iter requires endomorphism: {dom} -> {cod}")
            expected = (dom, cod)
        else:
            raise TypeError(f"Unsupported enode tag: {en.tag}")

        if sort != expected:
            raise TypeError(f"{en.tag} declared sort {sort} does not match inferred sort {expected}")

    def add_enode(self, en: ENode, sort: Sort) -> int:
        canon_children = tuple(self.uf.find(c) for c in en.children)
        en = ENode(en.tag, en.data, canon_children)
        self._validate_enode(en, sort)

        if en in self.memo:
            cid = self.uf.find(self.memo[en])
            if self.sort[cid] != sort:
                raise TypeError(f"ENode sort clash: {en} sort {sort} vs existing {self.sort[cid]}")
            return cid

        cid = self._new_class(sort)
        self.nodes[cid].add(en)
        self.memo[en] = cid
        self.worklist.append((en, cid))

        for child in canon_children:
            self.parents[child].append((en, cid))
        return cid

    def add_expr(self, e: Expr) -> int:
        e = normalize(e)

        if isinstance(e, Id):
            return self.add_enode(ENode("Id", (e.obj,), ()), (e.obj, e.obj))
        if isinstance(e, Box):
            opdef = self.sig.get(e.op)
            return self.add_enode(ENode("Box", (e.op, e.attrs), ()), (opdef.dom, opdef.cod))
        if isinstance(e, Par):
            left = self.add_expr(e.left)
            right = self.add_expr(e.right)
            dl, cl = self._child_sort(left)
            dr, cr = self._child_sort(right)
            return self.add_enode(ENode("Par", (), (left, right)), (dl @ dr, cl @ cr))
        if isinstance(e, Seq):
            first = self.add_expr(e.first)
            second = self.add_expr(e.second)
            da, ca = self._child_sort(first)
            db, cb = self._child_sort(second)
            if ca != db:
                raise TypeError(f"Seq type mismatch in add_expr: {ca} != {db}")
            return self.add_enode(ENode("Seq", (), (first, second)), (da, cb))
        if isinstance(e, Dup):
            return self.add_enode(ENode("Dup", (e.obj,), ()), (e.obj, e.obj @ e.obj))
        if isinstance(e, Del):
            return self.add_enode(ENode("Del", (e.obj,), ()), (e.obj, Obj("I")))
        if isinstance(e, Swap):
            return self.add_enode(
                ENode("Swap", (e.left, e.right), ()),
                (e.left @ e.right, e.right @ e.left),
            )
        if isinstance(e, Case):
            left = self.add_expr(e.left_branch)
            right = self.add_expr(e.right_branch)
            dl, cl = self._child_sort(left)
            dr, cr = self._child_sort(right)
            if dl != Obj("I"):
                raise TypeError(f"Case left branch must have domain I, got {dl}")
            if cl != cr:
                raise TypeError(f"Case branches must have same codomain, got {cl} and {cr}")
            return self.add_enode(ENode("Case", (), (left, right)), (Obj("I") + dr, cl))
        if isinstance(e, Iter):
            body = self.add_expr(e.body)
            dom, cod = self._child_sort(body)
            if dom != cod:
                raise TypeError(f"Iter requires endomorphism: {dom} -> {cod}")
            return self.add_enode(ENode("Iter", (e.count,), (body,)), (dom, cod))
        raise TypeError(type(e))

    def merge(self, a: int, b: int, reason: str = "") -> int:
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        if self.sort[ra] != self.sort[rb]:
            raise TypeError(f"Cannot merge different sorts: {self.sort[ra]} vs {self.sort[rb]}")

        new_rep = self.uf.union(ra, rb)
        old_rep = rb if new_rep == ra else ra
        self.worklist.extend(self.parents[old_rep])
        self.pending.append(old_rep)

        self.nodes[new_rep].update(self.nodes[old_rep])
        del self.nodes[old_rep]
        self.parents[new_rep].extend(self.parents[old_rep])
        del self.parents[old_rep]

        self.sort[new_rep] = self.sort[ra]
        self.sort.pop(old_rep, None)

        if reason:
            self.merge_log.append((reason, ra, rb))
        for callback in self.on_merge:
            callback(ra, rb)
        return new_rep

    def _eclass_is_pure(self, cid: int, active: set[int] | None = None) -> bool:
        """Conservatively determine whether every represented term is pure."""

        rep = self.uf.find(cid)
        active = set() if active is None else active
        if rep in active:
            return True
        active.add(rep)
        try:
            for node in self.nodes.get(rep, set()):
                if node.tag == "Box" and not self.sig.is_pure(node.data[0]):
                    return False
                if any(not self._eclass_is_pure(child, active) for child in node.children):
                    return False
            return True
        finally:
            active.remove(rep)

    def rebuild(self) -> None:
        """Repair congruence and apply admitted structural equations."""

        while self.pending or self.worklist:
            if self.pending:
                self.pending = []

            todo = self.worklist
            self.worklist = []
            seen: set[ENode] = set()

            for enode_stub, container_id in todo:
                container_id = self.uf.find(container_id)
                if enode_stub in seen:
                    continue
                seen.add(enode_stub)

                if enode_stub in self.nodes.get(container_id, set()):
                    self.nodes[container_id].remove(enode_stub)
                self.memo.pop(enode_stub, None)

                canon_children = tuple(self.uf.find(c) for c in enode_stub.children)
                new_enode = ENode(enode_stub.tag, enode_stub.data, canon_children)
                self._validate_enode(new_enode, self.sort[container_id])

                if new_enode in self.memo:
                    existing_class = self.memo[new_enode]
                    if self.uf.find(existing_class) != container_id:
                        self.merge(container_id, existing_class, reason="congruence")
                    continue

                self.memo[new_enode] = container_id
                self.nodes[container_id].add(new_enode)
                for child in canon_children:
                    self.parents[child].append((new_enode, container_id))

                if new_enode.tag != "Seq":
                    continue

                left, right = canon_children
                right_nodes = self.nodes.get(right, set())
                pure_left = self._eclass_is_pure(left)

                if pure_left and any(node.tag == "Del" for node in right_nodes):
                    dom_a, _ = self.sort[left]
                    del_a = self.add_enode(ENode("Del", (dom_a,), ()), (dom_a, Obj("I")))
                    if self.uf.find(del_a) != container_id:
                        self.merge(container_id, del_a, reason="del_naturality_pure")

                if pure_left and any(node.tag == "Dup" for node in right_nodes):
                    dom_a, cod_b = self.sort[left]
                    dup_a = self.add_enode(ENode("Dup", (dom_a,), ()), (dom_a, dom_a @ dom_a))
                    f_tensor_f = self.add_enode(
                        ENode("Par", (), (left, left)),
                        (dom_a @ dom_a, cod_b @ cod_b),
                    )
                    rhs = self.add_enode(
                        ENode("Seq", (), (dup_a, f_tensor_f)),
                        (dom_a, cod_b @ cod_b),
                    )
                    if self.uf.find(rhs) != container_id:
                        self.merge(container_id, rhs, reason="dup_naturality_pure")

                if any(node.tag == "Swap" for node in right_nodes):
                    left_nodes = self.nodes.get(left, set())
                    par_node = next((node for node in left_nodes if node.tag == "Par"), None)
                    if par_node is None:
                        continue
                    f, g = par_node.children
                    dom_f, cod_f = self.sort[f]
                    dom_g, cod_g = self.sort[g]
                    swap_ac = self.add_enode(
                        ENode("Swap", (dom_f, dom_g), ()),
                        (dom_f @ dom_g, dom_g @ dom_f),
                    )
                    g_tensor_f = self.add_enode(
                        ENode("Par", (), (g, f)),
                        (dom_g @ dom_f, cod_g @ cod_f),
                    )
                    rhs = self.add_enode(
                        ENode("Seq", (), (swap_ac, g_tensor_f)),
                        (dom_f @ dom_g, cod_g @ cod_f),
                    )
                    if self.uf.find(rhs) != container_id:
                        self.merge(container_id, rhs, reason="swap_naturality")
