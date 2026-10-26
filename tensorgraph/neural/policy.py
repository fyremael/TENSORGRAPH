"""Policy Network for Rule Selection (P3-4).

This module provides a neural network that predicts which rewrite rule
to apply given the current E-Graph state. The policy network π(rule|state)
enables learned rule prioritization during saturation.

Architecture:
- Input: E-Graph state embedding (from GNNStateEmbedder)
- Hidden: MLP with ReLU activations
- Output: Softmax probability distribution over rules

Training (future):
- Collect (state, rule, reward) tuples during saturation
- Use policy gradient or actor-critic methods to optimize
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import math

if TYPE_CHECKING:
    from ..rewrite.rule import Rewrite
    from .embedder import GNNStateEmbedder


@dataclass
class PolicyNetwork:
    """Neural policy for rule selection: π(rule|state).
    
    Given an E-Graph state embedding, outputs a probability distribution
    over available rewrite rules. Rules with higher probability are
    prioritized during saturation.
    
    Attributes:
        input_dim: Dimension of state embedding input.
        hidden_dim: Dimension of hidden layer.
        rule_names: List of rule names in fixed order.
    """
    
    input_dim: int = 64
    hidden_dim: int = 32
    rule_names: list[str] = field(default_factory=list)
    
    # Learnable parameters
    _w1: list[list[float]] = field(default_factory=list, repr=False)
    _b1: list[float] = field(default_factory=list, repr=False)
    _w2: list[list[float]] = field(default_factory=list, repr=False)
    _b2: list[float] = field(default_factory=list, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def _init_params(self, num_rules: int) -> None:
        """Initialize network parameters."""
        if self._initialized and len(self._b2) == num_rules:
            return
        
        import random
        
        # First layer: input_dim -> hidden_dim
        scale1 = math.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self._w1 = [
            [random.gauss(0, scale1) for _ in range(self.input_dim)]
            for _ in range(self.hidden_dim)
        ]
        self._b1 = [0.0] * self.hidden_dim
        
        # Second layer: hidden_dim -> num_rules
        scale2 = math.sqrt(2.0 / (self.hidden_dim + num_rules))
        self._w2 = [
            [random.gauss(0, scale2) for _ in range(self.hidden_dim)]
            for _ in range(num_rules)
        ]
        self._b2 = [0.0] * num_rules
        
        self._initialized = True
    
    def set_rules(self, rewrites: Sequence[Rewrite]) -> None:
        """Configure policy for a specific set of rewrite rules.
        
        Args:
            rewrites: The rewrite rules to select from.
        """
        self.rule_names = [rw.name for rw in rewrites]
        self._init_params(len(rewrites))
    
    def predict(self, state_embedding: list[float]) -> list[float]:
        """Predict rule probabilities given state embedding.
        
        Args:
            state_embedding: E-Graph state embedding vector.
            
        Returns:
            Probability distribution over rules (sums to 1).
        """
        if not self.rule_names:
            return []
        
        self._init_params(len(self.rule_names))
        
        # Hidden layer with ReLU
        hidden = self._linear(state_embedding, self._w1, self._b1)
        hidden = [max(0.0, x) for x in hidden]
        
        # Output layer (logits)
        logits = self._linear(hidden, self._w2, self._b2)
        
        # Softmax
        return self._softmax(logits)
    
    def predict_sorted(
        self,
        state_embedding: list[float],
        rewrites: Sequence[Rewrite],
    ) -> list[tuple[Rewrite, float]]:
        """Get rules sorted by predicted probability.
        
        Args:
            state_embedding: E-Graph state embedding vector.
            rewrites: Available rewrite rules.
            
        Returns:
            List of (rule, probability) tuples, sorted by probability descending.
        """
        if len(rewrites) != len(self.rule_names):
            self.set_rules(rewrites)
        
        probs = self.predict(state_embedding)
        
        # Pair with rules and sort
        paired = list(zip(rewrites, probs))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        return paired
    
    def sample(
        self,
        state_embedding: list[float],
        rewrites: Sequence[Rewrite],
        temperature: float = 1.0,
    ) -> Rewrite:
        """Sample a rule according to the policy distribution.
        
        Args:
            state_embedding: E-Graph state embedding vector.
            rewrites: Available rewrite rules.
            temperature: Sampling temperature (higher = more random).
            
        Returns:
            Sampled rewrite rule.
        """
        import random
        
        if len(rewrites) != len(self.rule_names):
            self.set_rules(rewrites)
        
        # Get logits
        hidden = self._linear(state_embedding, self._w1, self._b1)
        hidden = [max(0.0, x) for x in hidden]
        logits = self._linear(hidden, self._w2, self._b2)
        
        # Temperature scaling
        scaled = [l / temperature for l in logits]
        probs = self._softmax(scaled)
        
        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return rewrites[i]
        
        return rewrites[-1]  # Fallback
    
    @staticmethod
    def _linear(x: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
        """Linear layer: y = Wx + b."""
        result = []
        for o in range(len(weight)):
            val = bias[o]
            for i, xi in enumerate(x):
                if i < len(weight[o]):
                    val += weight[o][i] * xi
            result.append(val)
        return result
    
    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        """Numerically stable softmax."""
        if not logits:
            return []
        
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exp_l)
        
        return [e / sum_exp for e in exp_l]
    
    def state_dict(self) -> dict[str, Any]:
        """Export model parameters."""
        return {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "rule_names": self.rule_names,
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Import model parameters."""
        self._w1 = state["w1"]
        self._b1 = state["b1"]
        self._w2 = state["w2"]
        self._b2 = state["b2"]
        self.input_dim = state["input_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.rule_names = state["rule_names"]
        self._initialized = True


@dataclass
class ExperienceBuffer:
    """Buffer for collecting training data during saturation.
    
    Stores (state, action, reward) tuples for policy gradient training.
    
    Attributes:
        max_size: Maximum number of experiences to store.
        experiences: List of (state_embedding, rule_idx, reward) tuples.
    """
    
    max_size: int = 10000
    experiences: list[tuple[list[float], int, float]] = field(default_factory=list)
    
    def add(
        self,
        state_embedding: list[float],
        rule_idx: int,
        reward: float,
    ) -> None:
        """Add an experience to the buffer."""
        self.experiences.append((state_embedding, rule_idx, reward))
        
        # Evict oldest if over capacity
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)
    
    def clear(self) -> None:
        """Clear all experiences."""
        self.experiences = []
    
    def sample_batch(self, batch_size: int) -> list[tuple[list[float], int, float]]:
        """Sample a random batch of experiences."""
        import random
        
        if len(self.experiences) <= batch_size:
            return list(self.experiences)
        
        return random.sample(self.experiences, batch_size)
