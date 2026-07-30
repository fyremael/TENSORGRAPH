from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sharding import Shard


class MockFabric:
    """Deterministic in-process queue for cross-shard merge notifications."""

    def __init__(self) -> None:
        self.shards: dict[int, Shard] = {}
        self._queue: deque[tuple[int, int, int]] = deque()

    def register(self, shard: Shard) -> None:
        if shard.shard_id in self.shards and self.shards[shard.shard_id] is not shard:
            raise ValueError(f"shard {shard.shard_id} is already registered")
        shard.fabric = self
        self.shards[shard.shard_id] = shard

    def send_merge(self, source_shard: int, global_a: int, global_b: int) -> None:
        self._queue.append((source_shard, global_a, global_b))

    def pump(self, max_messages: int | None = None) -> int:
        processed = 0
        while self._queue and (max_messages is None or processed < max_messages):
            source_shard, global_a, global_b = self._queue.popleft()
            for shard_id, shard in tuple(self.shards.items()):
                if shard_id != source_shard:
                    shard.receive_merge(global_a, global_b)
            processed += 1
        return processed
