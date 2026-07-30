from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ENode:
    """Hashconsable node in the e-graph."""

    tag: str
    data: tuple[Any, ...]
    children: tuple[int, ...]
