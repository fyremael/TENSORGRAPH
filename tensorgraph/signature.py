from __future__ import annotations

from dataclasses import dataclass

from .types import Obj


@dataclass(frozen=True)
class OpDef:
    """Primitive generator declaration.

    Operations are pure by default. Add the ``effectful`` trait for random,
    stateful, mutating, I/O, or otherwise non-duplicable behavior.
    """

    name: str
    dom: Obj
    cod: Obj
    traits: frozenset[str] = frozenset()

    @property
    def is_pure(self) -> bool:
        return "effectful" not in self.traits


class Signature:
    """Mapping from operation names to typed, effect-annotated declarations."""

    def __init__(self) -> None:
        self._ops: dict[str, OpDef] = {}

    def add(self, name: str, dom: Obj, cod: Obj, traits: set[str] | None = None) -> None:
        if name in self._ops:
            raise ValueError(f"Op '{name}' already exists")
        normalized_traits = frozenset(traits or ())
        if "pure" in normalized_traits and "effectful" in normalized_traits:
            raise ValueError(f"Op '{name}' cannot be both pure and effectful")
        self._ops[name] = OpDef(
            name=name,
            dom=dom,
            cod=cod,
            traits=normalized_traits,
        )

    def get(self, name: str) -> OpDef:
        if name not in self._ops:
            raise KeyError(f"Unknown op '{name}'")
        return self._ops[name]

    def is_pure(self, name: str) -> bool:
        return self.get(name).is_pure

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)
