from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import cast

from ..rewrite.pattern import PBox, PIter, PPar, PSeq, Pattern, ematch_at
from ..rewrite.rule import Rewrite, instantiate_pattern
from .egraph import EGraph
from .trace import Trace


def saturate(
    eg: EGraph,
    rewrites: Sequence[Rewrite],
    iters: int = 8,
    max_applications: int = 10_000,
    trace: Trace | None = None,
) -> None:
    """Apply typed rewrites until a fixed point or a configured bound."""

    rules_by_head: dict[str, list[Rewrite]] = {}
    wildcard_rules: list[Rewrite] = []

    for rewrite in rewrites:
        head_key: str | None = None
        if isinstance(rewrite.lhs, PBox):
            head_key = f"Box:{rewrite.lhs.op}"
        elif isinstance(rewrite.lhs, PSeq):
            head_key = "Seq"
        elif isinstance(rewrite.lhs, PPar):
            head_key = "Par"
        elif isinstance(rewrite.lhs, PIter):
            head_key = "Iter"

        if head_key is None:
            wildcard_rules.append(rewrite)
        else:
            rules_by_head.setdefault(head_key, []).append(rewrite)

    for _ in range(iters):
        applied = 0
        representatives = sorted(eg.nodes)

        for representative in representatives:
            if representative not in eg.nodes:
                continue

            candidates = list(wildcard_rules)
            seen_rules = {id(rule) for rule in wildcard_rules}

            for enode in eg.nodes[representative]:
                key = f"Box:{enode.data[0]}" if enode.tag == "Box" else enode.tag
                for rewrite in rules_by_head.get(key, ()):  # type: ignore[arg-type]
                    if id(rewrite) not in seen_rules:
                        candidates.append(rewrite)
                        seen_rules.add(id(rewrite))

            for rewrite in candidates:
                if applied >= max_applications:
                    break

                matches = ematch_at(eg, representative, rewrite.lhs, {}, {}, {})
                for expression_env, object_env, data_env in matches:
                    if applied >= max_applications:
                        break

                    if isinstance(rewrite.rhs, Pattern):
                        rhs_id = instantiate_pattern(
                            eg,
                            rewrite.rhs,
                            expression_env,
                            object_env,
                            data_env,
                        )
                    else:
                        builder = cast(Callable[..., int], rewrite.rhs)
                        parameter_count = len(inspect.signature(builder).parameters)
                        if parameter_count == 4:
                            rhs_id = builder(eg, representative, expression_env, object_env)
                        elif parameter_count == 5:
                            rhs_id = builder(
                                eg,
                                representative,
                                expression_env,
                                object_env,
                                data_env,
                            )
                        else:
                            raise TypeError(
                                f"rewrite builder {rewrite.name!r} must accept four or five arguments"
                            )

                    root_before = eg.uf.find(representative)
                    rhs_before = eg.uf.find(rhs_id)
                    merged_rep = eg.merge(representative, rhs_id, reason=rewrite.name)

                    if trace is not None:
                        trace.record(
                            rule_name=rewrite.name,
                            root_eclass=root_before,
                            rhs_eclass=rhs_before,
                            merged_from=root_before if merged_rep != root_before else rhs_before,
                            merged_to=merged_rep,
                            expr_env=expression_env,
                            obj_env=object_env,
                            origin_mate=rewrite.origin,
                        )
                    applied += 1

            if applied >= max_applications:
                break

        eg.rebuild()
        if applied == 0:
            break
