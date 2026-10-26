from __future__ import annotations
from typing import Any, Callable

from ..ir import Box, Expr, Id, Par, Seq, normalize

from ..ir.primitives import Dup, Del, Swap, Case, Iter
from .egraph import EGraph
from .enode import ENode


def default_cost(en: ENode) -> int:
    """Default local cost.

    - `Box`: cost 1
    - others: cost 0
    """
    return 1 if en.tag == "Box" else 0


def make_host_aware_cost_function(
    seq_len: int = 1,
    batch_size: int = 1,
    hardware_caps: Any = None,
) -> Callable[[ENode], int]:
    """
    Create a host-aware and shape-aware E-Graph cost function.

    Adjusts node costs based on sequence length phase (decoding vs prefill)
    and host hardware (GPU HBM bandwidth vs launch latency overhead).
    """
    from ..hardware import get_hardware_capabilities

    caps = hardware_caps or get_hardware_capabilities()
    mode = caps.get_optimal_execution_mode(seq_len=seq_len, batch_size=batch_size)

    def host_cost(en: ENode) -> int:
        if en.tag == "Box":
            op = en.data[0] if isinstance(en.data, (tuple, list)) else en.data
            op_str = str(op).lower()

            is_fused = ("fused" in op_str and not "unfused" in op_str) or "graph" in op_str

            # Decoding phase (seq_len <= 8): minimize launch overhead
            if mode == "CUDA_GRAPH" or seq_len <= 8:
                if is_fused:
                    return 1
                return 3

            # Prefill phase (seq_len > 8): minimize HBM DRAM bandwidth allocations
            if mode == "FUSED_TRITON" or seq_len > 8:
                if is_fused:
                    return 1
                return 4


            return 2
        return 0

    return host_cost


class Extractor:
    """Cost-based extractor for e-graph equivalence classes."""


    def __init__(self, eg: EGraph, local_cost=default_cost) -> None:
        self.eg = eg
        self.local_cost = local_cost
        self.best_cost: dict[int, int] = {}
        self.best_node: dict[int, ENode] = {}

    def solve(self, root: int, max_rounds: int = 50) -> None:
        eg = self.eg
        reps = list(eg.nodes.keys())
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
                        if en.children:
                            continue
                        child_costs = 0

                    cand = self.local_cost(en) + child_costs

                    if cand < self.best_cost[r]:
                        self.best_cost[r] = cand
                        self.best_node[r] = en
                        changed = True

        root = eg.uf.find(root)
        if root not in self.best_node:
            self.best_node[root] = next(iter(eg.nodes[root]))
            self.best_cost[root] = 0

    def extract(self, root: int) -> Expr:
        eg = self.eg
        root = eg.uf.find(root)
        visiting: set[int] = set()

        def build(r: int) -> Expr:
            r = eg.uf.find(r)
            if r in visiting:
                return Box("_cycle")
            visiting.add(r)

            en = self.best_node.get(r)
            if en is None:
                en = next(iter(eg.nodes[r]))

            if en.tag == "Id":
                (obj,) = en.data
                out: Expr = Id(obj)
            elif en.tag == "Box":
                op, attrs = en.data
                out = Box(op, attrs)
            elif en.tag == "Par":
                out = Par(build(en.children[0]), build(en.children[1]))
            elif en.tag == "Seq":
                out = Seq(build(en.children[0]), build(en.children[1]))
            elif en.tag == "Dup":
                (obj,) = en.data
                out = Dup(obj)
            elif en.tag == "Del":
                (obj,) = en.data
                out = Del(obj)
            elif en.tag == "Swap":
                left, right = en.data
                out = Swap(left, right)
            elif en.tag == "Case":
                out = Case(build(en.children[0]), build(en.children[1]))
            elif en.tag == "Iter":
                (count,) = en.data
                out = Iter(build(en.children[0]), count)
            else:
                out = Box(f"_unknown_{en.tag}")

            visiting.remove(r)
            return out  # Note: skip global normalize to preserve structure

        return build(root)

        return build(root)
