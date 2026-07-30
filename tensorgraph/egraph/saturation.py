from __future__ import annotations

from collections.abc import Sequence

from ..rewrite.pattern import Pattern, ematch, ematch_at, PBox, PSeq, PPar, PIter
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
    """Equality saturation loop.

    Iteratively applies rewrites until a fixed point or `iters`.

    Args:
        eg: The e-graph to saturate.
        rewrites: Sequence of rewrite rules to apply.
        iters: Maximum number of saturation iterations.
        max_applications: Maximum rewrite applications per iteration.
        trace: Optional Trace object to record rewrite applications (FR-7).

    Safety:
        - Hard caps on total rewrite applications per iteration.
    """
    
    # Pre-index rules by head symbol for fast lookup
    rules_by_head: dict[str, list[Rewrite]] = {}
    wildcard_rules: list[Rewrite] = []

    for rw in rewrites:
        head_key = None
        if isinstance(rw.lhs, PBox):
            head_key = f"Box:{rw.lhs.op}"
        elif isinstance(rw.lhs, PSeq):
            head_key = "Seq"
        elif isinstance(rw.lhs, PPar):
            head_key = "Par"
        elif isinstance(rw.lhs, PIter):
            head_key = "Iter"
        # Add other structural patterns as needed if we have specific tags for them
        
        if head_key:
            rules_by_head.setdefault(head_key, []).append(rw)
        else:
            # PVar or unknown structure matches everything
            wildcard_rules.append(rw)

    for _ in range(iters):
        applied = 0
        
        # Inverted loop: Iterate E-classes -> Find Rules
        # Sort for determinism
        reps = sorted(eg.nodes.keys())
        
        for rep in reps:
            # Check if rep still exists (might have been merged away)
            if rep not in eg.nodes:
                continue

            # Gather candidate rules for this e-class
            candidates = []
            if wildcard_rules:
                candidates.extend(wildcard_rules)
            
            # Check nodes in the class to find applicable specific rules
            # Use a set to avoid adding same rule multiple times if multiple nodes match head
            seen_rules = {id(r) for r in wildcard_rules}
            
            for en in eg.nodes[rep]:
                key = None
                if en.tag == "Box":
                    key = f"Box:{en.data[0]}"
                else:
                    key = en.tag
                
                if key in rules_by_head:
                    for rw in rules_by_head[key]:
                        # Optimization: Avoid set lookup if list is short? 
                        # Or just use set for seen IDs.
                        if id(rw) not in seen_rules:
                            candidates.append(rw)
                            seen_rules.add(id(rw))

            # Attempt matches
            for rw in candidates:
                if applied >= max_applications:
                    break # Break inner rule loop
                
                # ematch_at returns list[tuple[Subst, ObjSubst, DataSubst]]
                matches = ematch_at(eg, rep, rw.lhs, {}, {}, {})
                
                for env, oenv, denv in matches:
                    if applied >= max_applications:
                        break

                    if isinstance(rw.rhs, Pattern):
                        rhs_id = instantiate_pattern(eg, rw.rhs, env, oenv, denv)
                    else:
                        import inspect
                        sig_len = len(inspect.signature(rw.rhs).parameters) if callable(rw.rhs) else 5
                        rhs_id = rw.rhs(eg, rep, env, oenv) if sig_len == 4 else rw.rhs(eg, rep, env, oenv, denv)

                    root_before = eg.uf.find(rep)
                    rhs_before = eg.uf.find(rhs_id)

                    merged_rep = eg.merge(rep, rhs_id, reason=rw.name)

                    if trace is not None:
                        trace.record(
                            rule_name=rw.name,
                            root_eclass=root_before,
                            rhs_eclass=rhs_before,
                            merged_from=root_before if merged_rep != root_before else rhs_before,
                            merged_to=merged_rep,
                            expr_env=env,
                            obj_env=oenv,
                            origin_mate=rw.origin,
                        )

                    applied += 1
            
            if applied >= max_applications:
                break

        eg.rebuild()

        if applied == 0:
            break
