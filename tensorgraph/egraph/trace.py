"""Proof/Audit Tracing for TENSORGRAPH.

This module implements FR-7 from SPEC.md:
    The system SHALL optionally record rewrite applications and unions.

Tracing records:
- Rule name
- Match environment (expression and object bindings)
- Merged class IDs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Obj

# Type aliases for trace environments
ExprEnv = dict[str, int]  # Pattern variable -> e-class ID
ObjEnv = dict[str, "Obj"]  # Object variable -> Obj


@dataclass
class TraceEntry:
    """A single trace entry recording a rewrite application.

    Attributes:
        rule_name: Name of the rewrite rule applied.
        root_eclass: The e-class where the match was found.
        rhs_eclass: The e-class ID of the instantiated RHS.
        merged_from: The original e-class ID before merge (if different).
        merged_to: The resulting e-class ID after merge.
        expr_env: Mapping from pattern variables to e-class IDs.
        obj_env: Mapping from object variables to concrete Obj instances.
    """

    rule_name: str
    root_eclass: int
    rhs_eclass: int
    merged_from: int
    merged_to: int
    expr_env: ExprEnv = field(default_factory=dict)
    obj_env: ObjEnv = field(default_factory=dict)
    origin_mate: str | None = None  # Name of the original rule if this is a mate

    def __str__(self) -> str:
        return (
            f"TraceEntry(rule={self.rule_name!r}, "
            f"merged {self.merged_from} → {self.merged_to}, "
            f"expr_env={self.expr_env})"
        )


@dataclass
class Trace:
    """Collection of trace entries with inspection utilities.

    The Trace object accumulates entries during saturation and provides
    methods to inspect, filter, and export the transformation history.
    """

    entries: list[TraceEntry] = field(default_factory=list)
    enabled: bool = True

    def record(
        self,
        rule_name: str,
        root_eclass: int,
        rhs_eclass: int,
        merged_from: int,
        merged_to: int,
        expr_env: ExprEnv | None = None,
        obj_env: ObjEnv | None = None,
        origin_mate: str | None = None,
    ) -> None:
        """Record a rewrite application if tracing is enabled."""
        if not self.enabled:
            return

        self.entries.append(
            TraceEntry(
                rule_name=rule_name,
                root_eclass=root_eclass,
                rhs_eclass=rhs_eclass,
                merged_from=merged_from,
                merged_to=merged_to,
                expr_env=dict(expr_env) if expr_env else {},
                obj_env=dict(obj_env) if obj_env else {},
                origin_mate=origin_mate,
            )
        )

    def clear(self) -> None:
        """Clear all trace entries."""
        self.entries.clear()

    def filter_by_rule(self, rule_name: str) -> list[TraceEntry]:
        """Return entries matching a specific rule name."""
        return [e for e in self.entries if e.rule_name == rule_name]

    def filter_by_eclass(self, eclass_id: int) -> list[TraceEntry]:
        """Return entries where the given e-class was involved in a merge."""
        return [
            e
            for e in self.entries
            if e.merged_from == eclass_id or e.merged_to == eclass_id
        ]

    def summary(self) -> dict[str, int]:
        """Return a summary of rule applications by rule name."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.rule_name] = counts.get(entry.rule_name, 0) + 1
        return counts

    def dump(self, max_entries: int | None = None) -> str:
        """Return a human-readable dump of the trace.

        Args:
            max_entries: Maximum number of entries to include. None for all.

        Returns:
            Multi-line string representation of the trace.
        """
        lines = [f"Trace ({len(self.entries)} entries):"]

        entries = self.entries[:max_entries] if max_entries else self.entries

        for i, entry in enumerate(entries):
            lines.append(f"  [{i}] {entry.rule_name}")
            lines.append(f"      root={entry.root_eclass}, rhs={entry.rhs_eclass}")
            lines.append(f"      merge: {entry.merged_from} → {entry.merged_to}")
            if entry.expr_env:
                lines.append(f"      expr_env: {entry.expr_env}")
            if entry.obj_env:
                obj_strs = {k: str(v) for k, v in entry.obj_env.items()}
                lines.append(f"      obj_env: {obj_strs}")

        if max_entries and len(self.entries) > max_entries:
            lines.append(f"  ... and {len(self.entries) - max_entries} more")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
