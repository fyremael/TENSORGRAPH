"""GNN State Embedder for E-Graph (P3-3).

This module provides a Graph Neural Network that embeds the current E-Graph
state into a fixed-dimensional vector. This embedding captures:
- E-class structure and connectivity
- Node operation types and attributes
- Graph-level statistics (size, depth)

The embedder operates on a graph representation of the E-Graph:
- Nodes: E-classes with features derived from their enodes
- Edges: Parent-child relationships between e-classes
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import math

if TYPE_CHECKING:
    from ..egraph import EGraph


@dataclass(frozen=True)
class EGraphView:
    """Lightweight view of E-Graph for embedding.
    
    Converts the E-Graph's internal representation to a graph-friendly
    format suitable for neural network processing.
    """
    
    node_features: list[list[float]]  # Per-node feature vectors
    edge_index: tuple[list[int], list[int]]  # (sources, targets)
    global_features: list[float]  # Graph-level features
    
    @classmethod
    def from_egraph(cls, eg: EGraph, op_vocab: dict[str, int] | None = None) -> EGraphView:
        """Convert E-Graph to neural-friendly view.
        
        Args:
            eg: The E-Graph to convert.
            op_vocab: Optional mapping from operation names to indices.
        """
        if op_vocab is None:
            op_vocab = {}
        
        # Build rep -> index mapping
        reps = sorted(eg.nodes.keys())
        rep_to_idx = {rep: i for i, rep in enumerate(reps)}
        
        # Node features: encode e-class properties
        node_features: list[list[float]] = []
        for rep in reps:
            enodes = eg.nodes.get(rep, set())
            features = cls._encode_eclass(enodes, op_vocab)
            node_features.append(features)
        
        # Edge index: parent-child relationships via enode children
        sources: list[int] = []
        targets: list[int] = []
        
        for rep in reps:
            enodes = eg.nodes.get(rep, set())
            src_idx = rep_to_idx[rep]
            
            for enode in enodes:
                for child in enode.children:
                    child_rep = eg.uf.find(child)
                    if child_rep in rep_to_idx:
                        tgt_idx = rep_to_idx[child_rep]
                        sources.append(src_idx)
                        targets.append(tgt_idx)
        
        # Global features
        global_features = [
            float(len(reps)),  # Number of e-classes
            float(sum(len(eg.nodes.get(r, set())) for r in reps)),  # Total enodes
            float(len(sources)),  # Number of edges
        ]
        
        return cls(
            node_features=node_features,
            edge_index=(sources, targets),
            global_features=global_features,
        )
    
    @staticmethod
    def _encode_eclass(
        enodes: set[Any],
        op_vocab: dict[str, int],
        feature_dim: int = 16,
    ) -> list[float]:
        """Encode an e-class as a fixed-size feature vector."""
        features = [0.0] * feature_dim
        
        # Feature 0: number of enodes (normalized)
        features[0] = math.tanh(len(enodes) / 10.0)
        
        # Features 1-4: tag distribution
        tag_counts = {"Id": 0, "Box": 0, "Seq": 0, "Par": 0}
        for en in enodes:
            if en.tag in tag_counts:
                tag_counts[en.tag] += 1
        
        total = max(1, sum(tag_counts.values()))
        features[1] = tag_counts["Id"] / total
        features[2] = tag_counts["Box"] / total
        features[3] = tag_counts["Seq"] / total
        features[4] = tag_counts["Par"] / total
        
        # Features 5-7: children statistics
        child_counts = [len(en.children) for en in enodes]
        if child_counts:
            features[5] = sum(child_counts) / len(child_counts)  # Mean
            features[6] = max(child_counts)  # Max
            features[7] = min(child_counts)  # Min
        
        # Features 8-15: reserved for operator embeddings
        for en in enodes:
            if en.tag == "Box" and en.data:
                op_name = en.data[0] if en.data else ""
                if op_name in op_vocab:
                    idx = op_vocab[op_name]
                    features[8 + (idx % 8)] += 1.0
        
        return features


@dataclass
class GNNStateEmbedder:
    """Graph Neural Network for E-Graph state embedding.
    
    This is a lightweight, pure-Python GNN implementation suitable for
    research and prototyping. For production, consider replacing with
    PyTorch Geometric or similar.
    
    Architecture:
    - Message passing: aggregate neighbor features
    - Node update: MLP on concatenated features
    - Global pooling: mean + max pooling
    
    Attributes:
        hidden_dim: Dimension of hidden representations.
        output_dim: Dimension of final state embedding.
        num_layers: Number of message passing layers.
        op_vocab: Mapping from operation names to indices.
    """
    
    hidden_dim: int = 32
    output_dim: int = 64
    num_layers: int = 2
    op_vocab: dict[str, int] = field(default_factory=dict)
    
    # Learnable parameters (initialized lazily)
    _weights: list[list[list[float]]] = field(default_factory=list, repr=False)
    _biases: list[list[float]] = field(default_factory=list, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def _init_params(self, input_dim: int = 16) -> None:
        """Initialize network parameters."""
        if self._initialized:
            return
        
        import random
        
        dims = [input_dim] + [self.hidden_dim] * self.num_layers
        
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            # Xavier initialization
            scale = math.sqrt(2.0 / (in_d + out_d))
            weight = [
                [random.gauss(0, scale) for _ in range(in_d)]
                for _ in range(out_d)
            ]
            bias = [0.0] * out_d
            self._weights.append(weight)
            self._biases.append(bias)
        
        # Output projection
        scale = math.sqrt(2.0 / (self.hidden_dim * 2 + self.output_dim))
        self._weights.append([
            [random.gauss(0, scale) for _ in range(self.hidden_dim * 2 + 3)]
            for _ in range(self.output_dim)
        ])
        self._biases.append([0.0] * self.output_dim)
        
        self._initialized = True
    
    def embed(self, eg: EGraph) -> list[float]:
        """Compute state embedding for an E-Graph.
        
        Args:
            eg: The E-Graph to embed.
            
        Returns:
            Fixed-size embedding vector of dimension output_dim.
        """
        view = EGraphView.from_egraph(eg, self.op_vocab)
        
        if not view.node_features:
            # Empty graph
            return [0.0] * self.output_dim
        
        self._init_params(len(view.node_features[0]))
        
        # Message passing layers
        node_h = view.node_features
        
        for layer_idx in range(self.num_layers):
            node_h = self._message_pass(
                node_h,
                view.edge_index,
                self._weights[layer_idx],
                self._biases[layer_idx],
            )
        
        # Global pooling: mean + max
        n = len(node_h)
        mean_pool = [sum(node_h[i][d] for i in range(n)) / n 
                     for d in range(self.hidden_dim)]
        max_pool = [max(node_h[i][d] for i in range(n)) 
                    for d in range(self.hidden_dim)]
        
        # Concatenate pooled features with global features
        pooled = mean_pool + max_pool + view.global_features
        
        # Final projection
        output = self._linear(pooled, self._weights[-1], self._biases[-1])
        
        return output
    
    def _message_pass(
        self,
        node_h: list[list[float]],
        edge_index: tuple[list[int], list[int]],
        weight: list[list[float]],
        bias: list[float],
    ) -> list[list[float]]:
        """Single message passing layer."""
        n = len(node_h)
        out_dim = len(weight)
        sources, targets = edge_index
        
        # Aggregate neighbor messages
        aggregated = [[0.0] * len(node_h[0]) for _ in range(n)]
        neighbor_counts = [0] * n
        
        for src, tgt in zip(sources, targets):
            for d in range(len(node_h[0])):
                aggregated[tgt][d] += node_h[src][d]
            neighbor_counts[tgt] += 1
        
        # Normalize by degree
        for i in range(n):
            if neighbor_counts[i] > 0:
                for d in range(len(aggregated[i])):
                    aggregated[i][d] /= neighbor_counts[i]
        
        # Update: combine self + aggregated, then linear + relu
        new_h = []
        for i in range(n):
            combined = [node_h[i][d] + aggregated[i][d] for d in range(len(node_h[0]))]
            projected = self._linear(combined, weight, bias)
            activated = [max(0.0, x) for x in projected]  # ReLU
            new_h.append(activated)
        
        return new_h
    
    @staticmethod
    def _linear(x: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
        """Linear layer: y = Wx + b."""
        out_dim = len(weight)
        result = []
        for o in range(out_dim):
            val = bias[o]
            for i, xi in enumerate(x):
                if i < len(weight[o]):
                    val += weight[o][i] * xi
            result.append(val)
        return result
    
    def state_dict(self) -> dict[str, Any]:
        """Export model parameters."""
        return {
            "weights": self._weights,
            "biases": self._biases,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "op_vocab": self.op_vocab,
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Import model parameters."""
        self._weights = state["weights"]
        self._biases = state["biases"]
        self.hidden_dim = state["hidden_dim"]
        self.output_dim = state["output_dim"]
        self.num_layers = state["num_layers"]
        self.op_vocab = state["op_vocab"]
        self._initialized = True
