"""E-graph data structures.

Note: we keep this module's imports minimal to avoid circular dependencies
between rewriting and saturation utilities.
"""

from .egraph import EGraph
from .enode import ENode
from .trace import Trace, TraceEntry
from .unionfind import UnionFind

__all__ = [
    "ENode",
    "UnionFind",
    "EGraph",
    "Trace",
    "TraceEntry",
]

