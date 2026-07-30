from __future__ import annotations

from ..ir import Box, Expr, Id, Par, Seq, normalize
from ..ir.primitives import Dup, Del, Swap, Case, Iter
from ..signature import Signature
from ..types import Sort, Obj
from .enode import ENode
from .unionfind import UnionFind


class EGraph:
    """A typed e-graph.

    Each e-class has a fixed sort `(dom, cod)`.
    """

    def __init__(self, sig: Signature) -> None:
        self.sig = sig
        self.uf = UnionFind()

        self.sort: dict[int, Sort] = {}            # rep -> sort
        self.nodes: dict[int, set[ENode]] = {}     # rep -> enodes
        self.memo: dict[ENode, int] = {}           # enode -> rep
        self.parents: dict[int, list[tuple[ENode, int]]] = {} # rep -> [(node, rep)]
        
        self.worklist: list[tuple[ENode, int]] = [] # Nodes that need re-canonicalization
        self.pending: list[int] = []               # list of reps merged since last rebuild

        self.root: int | None = None
        self.merge_log: list[tuple[str, int, int]] = []  # (reason, a, b)
        
        # Callback: (a: int, b: int) -> None
        self.on_merge: list[callable] = []

    def _new_class(self, sort: Sort) -> int:
        cid = self.uf.make()
        self.sort[cid] = sort
        self.nodes[cid] = set()
        self.parents[cid] = []
        return cid

    def add_enode(self, en: ENode, sort: Sort) -> int:
        canon_children = tuple(self.uf.find(c) for c in en.children)
        en = ENode(en.tag, en.data, canon_children)

        if en in self.memo:
            cid = self.uf.find(self.memo[en])
            if self.sort[cid] != sort:
                raise TypeError(
                    f"ENode sort clash: {en} sort {sort} vs existing {self.sort[cid]}"
                )
            return cid

        cid = self._new_class(sort)
        self.nodes[cid].add(en)
        self.memo[en] = cid
        
        # New nodes must be processed by rebuild for structural rules
        self.worklist.append((en, cid))
        
        # Register parents
        for child in canon_children:
            self.parents[child].append((en, cid))
            
        return cid

    def add_expr(self, e: Expr) -> int:
        e = normalize(e)

        if isinstance(e, Id):
            sort = (e.obj, e.obj)
            return self.add_enode(ENode("Id", (e.obj,), ()), sort)

        if isinstance(e, Box):
            opdef = self.sig.get(e.op)
            sort = (opdef.dom, opdef.cod)
            return self.add_enode(ENode("Box", (e.op, e.attrs), ()), sort)

        if isinstance(e, Par):
            l = self.add_expr(e.left)
            r = self.add_expr(e.right)
            dl, cl = self.sort[self.uf.find(l)]
            dr, cr = self.sort[self.uf.find(r)]
            return self.add_enode(ENode("Par", (), (l, r)), (dl @ dr, cl @ cr))

        if isinstance(e, Seq):
            a = self.add_expr(e.first)
            b = self.add_expr(e.second)
            da, ca = self.sort[self.uf.find(a)]
            db, cb = self.sort[self.uf.find(b)]
            if ca != db:
                raise TypeError(f"Seq type mismatch in add_expr: {ca} != {db}")
            return self.add_enode(ENode("Seq", (), (a, b)), (da, cb))

        if isinstance(e, Dup):
            return self.add_enode(ENode("Dup", (e.obj,), ()), (e.obj, e.obj @ e.obj))

        if isinstance(e, Del):
            return self.add_enode(ENode("Del", (e.obj,), ()), (e.obj, Obj("I")))

        if isinstance(e, Swap):
            return self.add_enode(ENode("Swap", (e.left, e.right), ()), (e.left @ e.right, e.right @ e.left))

        if isinstance(e, Case):
            l = self.add_expr(e.left_branch)
            r = self.add_expr(e.right_branch)
            dl, cl = self.sort[self.uf.find(l)]
            dr, cr = self.sort[self.uf.find(r)]
            return self.add_enode(ENode("Case", (), (l, r)), (Obj.sum_type(Obj("I"), dr), cl))

        if isinstance(e, Iter):
            body = self.add_expr(e.body)
            dom, cod = self.sort[self.uf.find(body)]
            if dom != cod:
                raise TypeError(f"Iter requires endomorphism: {dom} -> {cod}")
            return self.add_enode(ENode("Iter", (e.count,), (body,)), (dom, cod))

        raise TypeError(type(e))

    def merge(self, a: int, b: int, reason: str = "") -> int:
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        if self.sort[ra] != self.sort[rb]:
            raise TypeError(
                f"Cannot merge different sorts: {self.sort[ra]} vs {self.sort[rb]}"
            )

        new_rep = self.uf.union(ra, rb)
        old_rep = rb if new_rep == ra else ra
        
        # Add parents of the consumed class to worklist for repair
        self.worklist.extend(self.parents[old_rep])
        
        # Track pending merges for logs/debug (optional, but keep for robustness)
        self.pending.append(old_rep)

        self.nodes[new_rep].update(self.nodes[old_rep])
        del self.nodes[old_rep]
        
        self.parents[new_rep].extend(self.parents[old_rep])
        del self.parents[old_rep]
        
        self.sort[new_rep] = self.sort[ra]
        if old_rep in self.sort:
            del self.sort[old_rep]

        if reason:
            self.merge_log.append((reason, ra, rb))
            
        for cb in self.on_merge:
            cb(ra, rb)
            
        return new_rep

    def rebuild(self) -> None:
        """Optimized rebuild using worklist algorithm."""
        
        while self.pending or self.worklist:
            # Drain pending merges logic (mostly handled by merge populating worklist now)
            if self.pending:
                self.pending = [] 
                
            # Pop next batch from worklist
            todo = self.worklist
            self.worklist = []
            
            # Deduplicate nodes to process
            seen = set()
            
            for enode_stub, container_id in todo:
                container_id = self.uf.find(container_id)
                # Check if we already processed this specific enode instance in this batch
                if enode_stub in seen: continue
                seen.add(enode_stub)

                if enode_stub in self.nodes.get(container_id, set()):
                    self.nodes[container_id].remove(enode_stub)
                if enode_stub in self.memo:
                    del self.memo[enode_stub]
                
                # Re-canonicalize children
                canon_children = tuple(self.uf.find(c) for c in enode_stub.children)
                new_enode = ENode(enode_stub.tag, enode_stub.data, canon_children)
                
                if new_enode in self.memo:
                    existing_class = self.memo[new_enode]
                    if self.uf.find(existing_class) != container_id:
                        self.merge(container_id, existing_class, reason="congruence")
                else:
                    self.memo[new_enode] = container_id
                    self.nodes[container_id].add(new_enode)
                    
                    # Update parents for future merges
                    for child in canon_children:
                        self.parents[child].append((new_enode, container_id))
                        
                    # === INCREMENTAL STRUCTURAL RULES ===
                    if new_enode.tag == "Seq":
                        left, right = canon_children
                        
                        # Optimization: Access nodes directly for checks
                        right_nodes = self.nodes.get(right, set())
                        
                        # Rule: f ; Del(B) -> Del(A)
                        if any(n.tag == "Del" for n in right_nodes):
                            dom_a, _ = self.sort[left]
                            del_a_id = self.add_enode(ENode("Del", (dom_a,), ()), (dom_a, Obj("I")))
                            if self.uf.find(del_a_id) != container_id:
                                self.merge(container_id, del_a_id, reason="del_naturality")

                        # Rule: f ; Dup(B) -> Dup(A) ; (f ⊗ f)
                        if any(n.tag == "Dup" for n in right_nodes):
                             dom_a, cod_b = self.sort[left] # f: A -> B
                             dup_a = self.add_enode(ENode("Dup", (dom_a,), ()), (dom_a, dom_a @ dom_a))
                             f_f = self.add_enode(ENode("Par", (), (left, left)), (dom_a @ dom_a, cod_b @ cod_b))
                             rhs = self.add_enode(ENode("Seq", (), (dup_a, f_f)), (dom_a, cod_b @ cod_b))
                             if self.uf.find(rhs) != container_id:
                                 self.merge(container_id, rhs, reason="dup_naturality")

                        # Rule: (f ⊗ g) ; Swap(B, D) -> Swap(A, C) ; (g ⊗ f)
                        if any(n.tag == "Swap" for n in right_nodes):
                            left_nodes = self.nodes.get(left, set())
                            par_node = next((n for n in left_nodes if n.tag == "Par"), None)
                            if par_node:
                                f, g = par_node.children
                                dom_f, cod_f = self.sort[f]
                                dom_g, cod_g = self.sort[g]
                                
                                swap_ac = self.add_enode(ENode("Swap", (dom_f, dom_g), ()), (dom_f @ dom_g, dom_g @ dom_f))
                                g_f = self.add_enode(ENode("Par", (), (g, f)), (dom_g @ dom_f, cod_g @ cod_f))
                                rhs = self.add_enode(ENode("Seq", (), (swap_ac, g_f)), (dom_f @ dom_g, cod_g @ cod_f))
                                if self.uf.find(rhs) != container_id:
                                    self.merge(container_id, rhs, reason="swap_naturality")
