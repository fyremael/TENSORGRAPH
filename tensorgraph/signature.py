from __future__ import annotations

from dataclasses import dataclass

from .types import Obj


@dataclass(frozen=True)
class OpDef:
    """Primitive generator declaration: an operation with a domain and codomain."""

    name: str
    dom: Obj
    cod: Obj
    traits: frozenset[str] = frozenset()


class Signature:
    """Mapping from operation names to their type signatures."""

    def __init__(self) -> None:
        self._ops: dict[str, OpDef] = {}

    def add(self, name: str, dom: Obj, cod: Obj, traits: set[str] | None = None) -> None:
        if name in self._ops:
            raise ValueError(f"Op '{name}' already exists")
        self._ops[name] = OpDef(
            name=name, 
            dom=dom, 
            cod=cod, 
            traits=frozenset(traits) if traits else frozenset()
        )

    def get(self, name: str) -> OpDef:
        if name not in self._ops:
            raise KeyError(f"Unknown op '{name}'")
        return self._ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)
