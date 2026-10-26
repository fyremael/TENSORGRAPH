"""TENSORGRAPH Neural Heuristics (P3-3/4).

This module provides learned heuristics for guiding equality saturation:
- GNNStateEmbedder: Graph Neural Network for E-Graph state encoding
- PolicyNetwork: Rule selection policy π(rule|state)
- NeuralScheduler: Integration with saturation loop

The neural heuristics enable data-driven rule prioritization, potentially
reducing saturation time by focusing on high-value rewrites.
"""

from .embedder import GNNStateEmbedder
from .policy import PolicyNetwork
from .scheduler import NeuralScheduler

__all__ = ["GNNStateEmbedder", "PolicyNetwork", "NeuralScheduler"]
