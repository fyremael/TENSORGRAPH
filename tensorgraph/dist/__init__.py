"""Bounded in-process distributed compatibility layer.

This namespace preserves the original public imports while the separate
``tensorgraph.distributed`` research scaffold is redesigned. It supports local
merge propagation only and makes no multi-node production claim.
"""

from .fabric import AsyncFabric, FabricMessage, FabricStats, MessageType, create_fabric
from .mock_fabric import MockFabric
from .sharding import Partition, Shard

__all__ = [
    "AsyncFabric",
    "FabricMessage",
    "FabricStats",
    "MessageType",
    "MockFabric",
    "Partition",
    "Shard",
    "create_fabric",
]
