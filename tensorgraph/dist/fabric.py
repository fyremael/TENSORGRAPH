from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from .mock_fabric import MockFabric


class MessageType(str, Enum):
    MERGE = "merge"


@dataclass(frozen=True)
class FabricMessage:
    message_type: MessageType
    source_shard: int
    global_a: int
    global_b: int


@dataclass
class FabricStats:
    messages_sent: int = 0
    messages_processed: int = 0


class AsyncFabric(MockFabric):
    """In-process background dispatcher.

    This class provides testable local concurrency. It is not a network fabric
    and does not support durability, retries, ordering across processes, or
    distributed failure recovery.
    """

    def __init__(self, batch_size: int = 100) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._stats = FabricStats()
        self._running = False
        self._dispatcher_thread: threading.Thread | None = None
        self._wake = threading.Event()

    def send_merge(self, source_shard: int, global_a: int, global_b: int) -> None:
        super().send_merge(source_shard, global_a, global_b)
        self._stats.messages_sent += 1
        self._wake.set()

    def pump(self, max_messages: int | None = None) -> int:
        limit = self.batch_size if max_messages is None else max_messages
        processed = super().pump(limit)
        self._stats.messages_processed += processed
        return processed

    def get_stats(self) -> FabricStats:
        return FabricStats(
            messages_sent=self._stats.messages_sent,
            messages_processed=self._stats.messages_processed,
        )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="tensorgraph-local-fabric",
            daemon=True,
        )
        self._dispatcher_thread.start()

    def _dispatch_loop(self) -> None:
        while self._running:
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            self.pump()

    def stop(self, timeout: float | None = None) -> None:
        self._running = False
        self._wake.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=timeout)


def create_fabric(mode: str = "mock", **kwargs: object) -> MockFabric:
    if mode == "mock":
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"mock fabric does not accept options: {unexpected}")
        return MockFabric()
    if mode == "async":
        batch_size = kwargs.pop("batch_size", 100)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected async fabric options: {unexpected}")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        return AsyncFabric(batch_size=batch_size)
    raise ValueError(f"unknown fabric mode: {mode}")
