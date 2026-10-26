"""Neural Scheduler for Guided Saturation.

This module provides a NeuralScheduler that integrates the GNN embedder
and policy network with the saturation loop. It replaces the default
round-robin rule scheduling with learned rule prioritization.

Usage:
    from tensorgraph.neural import NeuralScheduler
    
    scheduler = NeuralScheduler()
    scheduler.saturate(egraph, rewrites, iters=10)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..egraph import EGraph
    from ..egraph.trace import Trace
    from ..rewrite.rule import Rewrite

from .embedder import GNNStateEmbedder
from .policy import ExperienceBuffer, PolicyNetwork


@dataclass
class NeuralScheduler:
    """Neural-guided saturation scheduler.
    
    Integrates GNN state embedding with policy-based rule selection to
    guide the saturation process more efficiently than round-robin.
    
    Attributes:
        embedder: GNN for E-Graph state embedding.
        policy: Policy network for rule selection.
        mode: Scheduling mode - 'greedy', 'sample', or 'hybrid'.
        exploration_rate: Probability of random rule selection (0-1).
        temperature: Sampling temperature for 'sample' mode.
        collect_experience: Whether to collect training data.
    """
    
    embedder: GNNStateEmbedder = field(default_factory=GNNStateEmbedder)
    policy: PolicyNetwork = field(default_factory=PolicyNetwork)
    mode: Literal["greedy", "sample", "hybrid"] = "hybrid"
    exploration_rate: float = 0.1
    temperature: float = 1.0
    collect_experience: bool = False
    experience_buffer: ExperienceBuffer = field(default_factory=ExperienceBuffer)
    
    # Statistics
    _stats: dict[str, int] = field(default_factory=dict, repr=False)
    
    def saturate(
        self,
        eg: EGraph,
        rewrites: Sequence[Rewrite],
        iters: int = 8,
        max_applications: int = 10_000,
        trace: Trace | None = None,
    ) -> dict[str, int]:
        """Neural-guided equality saturation.
        
        Args:
            eg: The E-Graph to saturate.
            rewrites: Sequence of rewrite rules to apply.
            iters: Maximum number of saturation iterations.
            max_applications: Maximum rewrite applications per iteration.
            trace: Optional Trace object for recording rewrites.
            
        Returns:
            Statistics dictionary with application counts.
        """
        from ..rewrite.pattern import ematch
        from ..rewrite.rule import instantiate_pattern
        from ..rewrite.pattern import Pattern
        
        import random
        
        self.policy.set_rules(rewrites)
        self._stats = {"total_applied": 0, "explored": 0, "exploited": 0}
        
        for iteration in range(iters):
            applied = 0
            
            # Get state embedding
            state_embedding = self.embedder.embed(eg)
            
            # Get rule ordering based on mode
            if self.mode == "greedy":
                ordered_rules = self.policy.predict_sorted(state_embedding, rewrites)
            elif self.mode == "sample":
                # Shuffle by sampling without replacement
                ordered_rules = self._sample_ordering(state_embedding, rewrites)
            else:  # hybrid
                ordered_rules = self._hybrid_ordering(state_embedding, rewrites)
            
            # Apply rules in priority order
            for rw, prob in ordered_rules:
                if applied >= max_applications:
                    break
                
                matches = list(ematch(eg, rw.lhs))
                
                for match_tuple in matches:
                    if applied >= max_applications:
                        break
                    root, env, oenv = match_tuple[:3]
                    denv = match_tuple[3] if len(match_tuple) > 3 else {}

                    # Instantiate RHS
                    if isinstance(rw.rhs, Pattern):
                        rhs_id = instantiate_pattern(eg, rw.rhs, env, oenv, denv)
                    else:
                        import inspect
                        sig_len = len(inspect.signature(rw.rhs).parameters) if callable(rw.rhs) else 5
                        rhs_id = rw.rhs(eg, root, env, oenv) if sig_len == 4 else rw.rhs(eg, root, env, oenv, denv)
                    
                    # Record pre-merge state
                    root_before = eg.uf.find(root)
                    rhs_before = eg.uf.find(rhs_id)
                    
                    merged_rep = eg.merge(root, rhs_id, reason=rw.name)
                    
                    # Trace if enabled
                    if trace is not None:
                        trace.record(
                            rule_name=rw.name,
                            root_eclass=root_before,
                            rhs_eclass=rhs_before,
                            merged_from=root_before if merged_rep != root_before else rhs_before,
                            merged_to=merged_rep,
                            expr_env=env,
                            obj_env=oenv,
                        )
                    
                    # Collect experience for training
                    if self.collect_experience:
                        rule_idx = self.policy.rule_names.index(rw.name)
                        # Reward: 1 if merge actually changed something
                        reward = 1.0 if root_before != rhs_before else 0.1
                        self.experience_buffer.add(state_embedding, rule_idx, reward)
                    
                    applied += 1
            
            eg.rebuild()
            self._stats["total_applied"] += applied
            
            if applied == 0:
                break
        
        return self._stats
    
    def _sample_ordering(
        self,
        state_embedding: list[float],
        rewrites: Sequence[Rewrite],
    ) -> list[tuple[Rewrite, float]]:
        """Sample rules according to policy distribution."""
        import random
        
        probs = self.policy.predict(state_embedding)
        paired = list(zip(rewrites, probs))
        
        # Weighted shuffle: sample without replacement proportional to probs
        result: list[tuple[Rewrite, float]] = []
        remaining = list(paired)
        
        while remaining:
            weights = [p for _, p in remaining]
            total = sum(weights)
            if total == 0:
                random.shuffle(remaining)
                result.extend(remaining)
                break
            
            # Sample one
            r = random.random() * total
            cumsum = 0.0
            for i, (rw, p) in enumerate(remaining):
                cumsum += p
                if r < cumsum:
                    result.append((rw, p))
                    remaining.pop(i)
                    break
            else:
                result.append(remaining.pop())
        
        return result
    
    def _hybrid_ordering(
        self,
        state_embedding: list[float],
        rewrites: Sequence[Rewrite],
    ) -> list[tuple[Rewrite, float]]:
        """Hybrid ordering: mostly greedy with exploration."""
        import random
        
        if random.random() < self.exploration_rate:
            # Explore: random ordering
            self._stats["explored"] = self._stats.get("explored", 0) + 1
            paired = [(rw, 1.0 / len(rewrites)) for rw in rewrites]
            random.shuffle(paired)
            return paired
        else:
            # Exploit: greedy ordering
            self._stats["exploited"] = self._stats.get("exploited", 0) + 1
            return self.policy.predict_sorted(state_embedding, rewrites)
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {}
    
    def save(self, path: str) -> None:
        """Save embedder and policy parameters to file."""
        import json
        
        state = {
            "embedder": self.embedder.state_dict(),
            "policy": self.policy.state_dict(),
            "mode": self.mode,
            "exploration_rate": self.exploration_rate,
            "temperature": self.temperature,
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load embedder and policy parameters from file."""
        import json
        
        with open(path) as f:
            state = json.load(f)
        
        self.embedder.load_state_dict(state["embedder"])
        self.policy.load_state_dict(state["policy"])
        self.mode = state.get("mode", "hybrid")
        self.exploration_rate = state.get("exploration_rate", 0.1)
        self.temperature = state.get("temperature", 1.0)
