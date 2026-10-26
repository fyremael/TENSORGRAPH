"""Tests for TENSORGRAPH Neural Heuristics (P3-3/4).

This test module verifies:
- GNNStateEmbedder correctly produces fixed-size embeddings
- PolicyNetwork produces valid probability distributions
- NeuralScheduler integrates with saturation correctly
"""

import pytest

from tensorgraph.egraph import EGraph
from tensorgraph.ir import Box, Par, Seq
from tensorgraph.neural import GNNStateEmbedder, NeuralScheduler, PolicyNetwork
from tensorgraph.neural.embedder import EGraphView
from tensorgraph.neural.policy import ExperienceBuffer
from tensorgraph.rewrite import Rewrite
from tensorgraph.rewrite.pattern import PBox
from tensorgraph.signature import Signature
from tensorgraph.types import Obj


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_sig() -> Signature:
    """Create a simple test signature."""
    sig = Signature()
    T = Obj("T")
    sig.add("f", T, T)
    sig.add("g", T, T)
    sig.add("h", T, T)
    return sig


@pytest.fixture
def simple_egraph(simple_sig: Signature) -> EGraph:
    """Create a simple E-Graph for testing."""
    eg = EGraph(simple_sig)
    
    # Add some expressions
    T = Obj("T")
    f = Box("f")
    g = Box("g")
    h = Box("h")
    
    eg.add_expr(f)
    eg.add_expr(Seq(f, g))
    eg.add_expr(Seq(g, h))
    eg.add_expr(Par(f, g))
    
    return eg


@pytest.fixture
def simple_rewrites(simple_sig: Signature) -> list[Rewrite]:
    """Create simple rewrite rules for testing."""
    return [
        Rewrite("f_to_g", PBox("f"), PBox("g")),
        Rewrite("g_to_h", PBox("g"), PBox("h")),
    ]


# -----------------------------------------------------------------------------
# P3-3: GNN State Embedding Tests
# -----------------------------------------------------------------------------


class TestEGraphView:
    """Tests for E-Graph to neural view conversion."""
    
    def test_from_empty_egraph(self, simple_sig: Signature) -> None:
        """Empty E-Graph produces empty view."""
        eg = EGraph(simple_sig)
        view = EGraphView.from_egraph(eg)
        
        assert view.node_features == []
        assert view.edge_index == ([], [])
        assert view.global_features == [0.0, 0.0, 0.0]
    
    def test_from_simple_egraph(self, simple_egraph: EGraph) -> None:
        """Simple E-Graph produces valid view."""
        view = EGraphView.from_egraph(simple_egraph)
        
        # Should have multiple nodes
        assert len(view.node_features) > 0
        
        # Each node should have 16 features
        for features in view.node_features:
            assert len(features) == 16
        
        # Global features should reflect graph size
        assert view.global_features[0] > 0  # Number of classes
    
    def test_node_features_normalized(self, simple_egraph: EGraph) -> None:
        """Node features should be in reasonable ranges."""
        view = EGraphView.from_egraph(simple_egraph)
        
        for features in view.node_features:
            # First feature is tanh-normalized
            assert -1.0 <= features[0] <= 1.0
            
            # Tag distribution features should sum to ~1
            tag_sum = sum(features[1:5])
            assert 0.99 <= tag_sum <= 1.01


class TestGNNStateEmbedder:
    """Tests for GNN state embedding."""
    
    def test_embed_empty(self, simple_sig: Signature) -> None:
        """Empty E-Graph produces zero embedding."""
        eg = EGraph(simple_sig)
        embedder = GNNStateEmbedder(output_dim=32)
        
        embedding = embedder.embed(eg)
        
        assert len(embedding) == 32
        assert all(x == 0.0 for x in embedding)
    
    def test_embed_simple(self, simple_egraph: EGraph) -> None:
        """Simple E-Graph produces fixed-size embedding."""
        embedder = GNNStateEmbedder(hidden_dim=16, output_dim=32)
        
        embedding = embedder.embed(simple_egraph)
        
        assert len(embedding) == 32
        # Should have non-zero values
        assert any(x != 0.0 for x in embedding)
    
    def test_embed_deterministic(self, simple_egraph: EGraph) -> None:
        """Same embedder produces same embedding for same graph."""
        embedder = GNNStateEmbedder(output_dim=32)
        
        emb1 = embedder.embed(simple_egraph)
        emb2 = embedder.embed(simple_egraph)
        
        assert emb1 == emb2
    
    def test_state_dict_roundtrip(self, simple_egraph: EGraph) -> None:
        """Parameters can be saved and loaded."""
        embedder1 = GNNStateEmbedder(hidden_dim=16, output_dim=32)
        emb1 = embedder1.embed(simple_egraph)
        
        state = embedder1.state_dict()
        
        embedder2 = GNNStateEmbedder()
        embedder2.load_state_dict(state)
        emb2 = embedder2.embed(simple_egraph)
        
        assert emb1 == emb2


# -----------------------------------------------------------------------------
# P3-4: Policy Network Tests
# -----------------------------------------------------------------------------


class TestPolicyNetwork:
    """Tests for policy network rule selection."""
    
    def test_predict_empty(self) -> None:
        """Empty rule list produces empty distribution."""
        policy = PolicyNetwork()
        
        probs = policy.predict([0.0] * 64)
        
        assert probs == []
    
    def test_predict_valid_distribution(
        self, simple_rewrites: list[Rewrite]
    ) -> None:
        """Policy produces valid probability distribution."""
        policy = PolicyNetwork(input_dim=32)
        policy.set_rules(simple_rewrites)
        
        state = [0.5] * 32
        probs = policy.predict(state)
        
        # Should have one probability per rule
        assert len(probs) == len(simple_rewrites)
        
        # Probabilities should sum to 1
        assert abs(sum(probs) - 1.0) < 1e-6
        
        # All probabilities should be non-negative
        assert all(p >= 0.0 for p in probs)
    
    def test_predict_sorted(self, simple_rewrites: list[Rewrite]) -> None:
        """Sorted predictions are in descending order."""
        policy = PolicyNetwork(input_dim=32)
        policy.set_rules(simple_rewrites)
        
        state = [0.5] * 32
        sorted_rules = policy.predict_sorted(state, simple_rewrites)
        
        assert len(sorted_rules) == len(simple_rewrites)
        
        # Check descending order
        probs = [p for _, p in sorted_rules]
        assert probs == sorted(probs, reverse=True)
    
    def test_sample_returns_valid_rule(
        self, simple_rewrites: list[Rewrite]
    ) -> None:
        """Sampling returns a rule from the input set."""
        policy = PolicyNetwork(input_dim=32)
        policy.set_rules(simple_rewrites)
        
        state = [0.5] * 32
        
        # Sample multiple times
        for _ in range(10):
            rule = policy.sample(state, simple_rewrites)
            assert rule in simple_rewrites
    
    def test_state_dict_roundtrip(
        self, simple_rewrites: list[Rewrite]
    ) -> None:
        """Parameters can be saved and loaded."""
        policy1 = PolicyNetwork(input_dim=32)
        policy1.set_rules(simple_rewrites)
        state_in = [0.5] * 32
        probs1 = policy1.predict(state_in)
        
        state_dict = policy1.state_dict()
        
        policy2 = PolicyNetwork()
        policy2.load_state_dict(state_dict)
        probs2 = policy2.predict(state_in)
        
        assert probs1 == probs2


class TestExperienceBuffer:
    """Tests for experience collection buffer."""
    
    def test_add_and_sample(self) -> None:
        """Experiences can be added and sampled."""
        buffer = ExperienceBuffer(max_size=100)
        
        # Add experiences
        for i in range(50):
            buffer.add([float(i)] * 32, i % 3, 1.0)
        
        assert len(buffer.experiences) == 50
        
        # Sample batch
        batch = buffer.sample_batch(10)
        assert len(batch) == 10
    
    def test_max_size_eviction(self) -> None:
        """Old experiences are evicted when over capacity."""
        buffer = ExperienceBuffer(max_size=10)
        
        for i in range(20):
            buffer.add([float(i)] * 32, 0, 1.0)
        
        assert len(buffer.experiences) == 10
        
        # Oldest should be evicted (first element should be i=10)
        assert buffer.experiences[0][0][0] == 10.0
    
    def test_clear(self) -> None:
        """Buffer can be cleared."""
        buffer = ExperienceBuffer()
        buffer.add([0.0] * 32, 0, 1.0)
        
        buffer.clear()
        
        assert len(buffer.experiences) == 0


# -----------------------------------------------------------------------------
# Neural Scheduler Tests
# -----------------------------------------------------------------------------


class TestNeuralScheduler:
    """Tests for neural-guided saturation."""
    
    def test_saturate_basic(
        self,
        simple_sig: Signature,
        simple_rewrites: list[Rewrite],
    ) -> None:
        """Neural scheduler completes saturation."""
        eg = EGraph(simple_sig)
        eg.add_expr(Box("f"))
        
        scheduler = NeuralScheduler(mode="greedy")
        stats = scheduler.saturate(eg, simple_rewrites, iters=5)
        
        assert "total_applied" in stats
        assert stats["total_applied"] >= 0
    
    def test_saturate_all_modes(
        self,
        simple_sig: Signature,
        simple_rewrites: list[Rewrite],
    ) -> None:
        """All scheduling modes work."""
        for mode in ["greedy", "sample", "hybrid"]:
            eg = EGraph(simple_sig)
            eg.add_expr(Box("f"))
            
            scheduler = NeuralScheduler(mode=mode)  # type: ignore[arg-type]
            stats = scheduler.saturate(eg, simple_rewrites, iters=3)
            
            assert stats["total_applied"] >= 0
    
    def test_experience_collection(
        self,
        simple_sig: Signature,
        simple_rewrites: list[Rewrite],
    ) -> None:
        """Experience collection works during saturation."""
        eg = EGraph(simple_sig)
        eg.add_expr(Box("f"))
        
        scheduler = NeuralScheduler(collect_experience=True)
        scheduler.saturate(eg, simple_rewrites, iters=5)
        
        # Should have collected some experiences
        assert len(scheduler.experience_buffer.experiences) > 0
    
    def test_save_load_roundtrip(
        self, simple_sig: Signature, simple_rewrites: list[Rewrite], tmp_path
    ) -> None:
        """Scheduler can be saved and loaded."""
        eg = EGraph(simple_sig)
        eg.add_expr(Box("f"))
        
        scheduler1 = NeuralScheduler(mode="hybrid", exploration_rate=0.2)
        scheduler1.policy.set_rules(simple_rewrites)
        
        # Force parameter initialization
        _ = scheduler1.embedder.embed(eg)
        _ = scheduler1.policy.predict([0.0] * 64)
        
        save_path = str(tmp_path / "scheduler.json")
        scheduler1.save(save_path)
        
        scheduler2 = NeuralScheduler()
        scheduler2.load(save_path)
        
        assert scheduler2.mode == "hybrid"
        assert scheduler2.exploration_rate == 0.2
