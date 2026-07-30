from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..egraph import EGraph
from ..ir import Expr
from ..signature import Signature

if TYPE_CHECKING:
    from .mock_fabric import MockFabric


@dataclass
class Partition:
    """Local ownership and ghost mapping for one in-process shard."""

    egraph: EGraph
    owned: dict[int, int] = field(default_factory=dict)
    ghosts: dict[int, int] = field(default_factory=dict)

    def register_ghost(self, global_id: int, local_id: int) -> None:
        if global_id < 0 or local_id < 0:
            raise ValueError("global and local identifiers must be non-negative")
        self.ghosts[global_id] = local_id


class Shard:
    """Bounded local shard used for deterministic merge-propagation tests."""

    def __init__(
        self,
        shard_id: int,
        sig: Signature,
        fabric: MockFabric | None = None,
    ) -> None:
        if shard_id < 0:
            raise ValueError("shard_id must be non-negative")
        self.shard_id = shard_id
        self.fabric = fabric
        self.partition = Partition(EGraph(sig))
        self.partition.egraph.on_merge.append(self._on_local_merge)

    def ingest(self, expr: Expr, global_id: int | None = None) -> int:
        local_id = self.partition.egraph.add_expr(expr)
        if global_id is not None:
            if global_id < 0:
                raise ValueError("global_id must be non-negative")
            self.partition.owned[local_id] = global_id
        return local_id

    def _global_for_local(self, local_id: int) -> int | None:
        if local_id in self.partition.owned:
            return self.partition.owned[local_id]
        for global_id, ghost_local in self.partition.ghosts.items():
            if ghost_local == local_id:
                return global_id
        return None

    def _on_local_merge(self, local_a: int, local_b: int) -> None:
        if self.fabric is None:
            return
        global_a = self._global_for_local(local_a)
        global_b = self._global_for_local(local_b)
        if global_a is None or global_b is None or global_a == global_b:
            return
        self.fabric.send_merge(self.shard_id, global_a, global_b)

    def receive_merge(self, global_a: int, global_b: int) -> bool:
        local_a = self.partition.ghosts.get(global_a)
        local_b = self.partition.ghosts.get(global_b)
        if local_a is None or local_b is None:
            return False
        self.partition.egraph.merge(local_a, local_b, reason="cross_shard_merge")
        return True
