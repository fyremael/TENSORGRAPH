"""
Tests for TENSORGRAPH Hardware Awareness, Hybrid Engine Routing, and E-Graph Cost Model Tuning.
"""

from __future__ import annotations

import pytest

from tensorgraph.egraph import EGraph, ENode
from tensorgraph.egraph.extract import Extractor, make_host_aware_cost_function

from tensorgraph.engine import HybridEngine
from tensorgraph.hardware import HardwareCapabilities, get_hardware_capabilities
from tensorgraph.ir import Box
from tensorgraph.signature import Signature
from tensorgraph.types import Obj


class TestHardwareCapabilities:
    """Test hardware probing and routing decisions."""

    def test_get_hardware_capabilities_returns_singleton(self):
        """get_hardware_capabilities returns a valid HardwareCapabilities instance."""
        caps1 = get_hardware_capabilities()
        caps2 = get_hardware_capabilities()

        assert isinstance(caps1, HardwareCapabilities)
        assert caps1 is caps2

    def test_optimal_execution_mode_decoding_vs_prefill(self):
        """Routing decision maps seq_len <= 8 to CUDA_GRAPH and seq_len > 8 to FUSED_TRITON."""
        caps = HardwareCapabilities(has_cuda=True, has_triton=True, has_cuda_graph=True)

        assert caps.get_optimal_execution_mode(seq_len=1) == "CUDA_GRAPH"
        assert caps.get_optimal_execution_mode(seq_len=8) == "CUDA_GRAPH"
        assert caps.get_optimal_execution_mode(seq_len=512) == "FUSED_TRITON"

    def test_optimal_execution_mode_cpu_fallback(self):
        """Routing falls back to EAGER_FALLBACK when CUDA is absent."""
        caps = HardwareCapabilities(has_cuda=False, has_triton=False, has_cuda_graph=False)

        assert caps.get_optimal_execution_mode(seq_len=1) == "EAGER_FALLBACK"
        assert caps.get_optimal_execution_mode(seq_len=512) == "EAGER_FALLBACK"


class TestHostAwareCostModel:
    """Test shape and hardware-aware cost function generator for EGraph Extractor."""

    def test_cost_function_decoding_phase(self):
        """Decoding phase cost function prioritizes CUDA Graph/Fused nodes."""
        caps = HardwareCapabilities(has_cuda=True, has_triton=True, has_cuda_graph=True)
        cost_fn = make_host_aware_cost_function(seq_len=1, hardware_caps=caps)

        graph_node = ENode("Box", ("fused_swiglu_graph",), tuple())
        eager_node = ENode("Box", ("unfused_matmul",), tuple())

        assert cost_fn(graph_node) < cost_fn(eager_node)

    def test_cost_function_prefill_phase(self):
        """Prefill phase cost function prioritizes fused memory kernels."""
        caps = HardwareCapabilities(has_cuda=True, has_triton=True, has_cuda_graph=True)
        cost_fn = make_host_aware_cost_function(seq_len=512, hardware_caps=caps)

        fused_triton_node = ENode("Box", ("fused_triton_swiglu",), tuple())
        unfused_node = ENode("Box", ("eager_allocation",), tuple())

        assert cost_fn(fused_triton_node) < cost_fn(unfused_node)



class TestHybridEngine:
    """Test Host-Aware Hybrid Execution Engine."""

    def test_hybrid_engine_execution_routing(self):
        """HybridEngine correctly updates last_execution_mode during execution."""
        caps = HardwareCapabilities(has_cuda=True, has_triton=True, has_cuda_graph=True)

        def sample_model(x):
            return x * 2.0

        engine = HybridEngine(sample_model, hardware_caps=caps, auto_capture_graph=False)

        # Single-token decoding (seq_len=1)
        res_dec = engine(10.0, seq_len=1)
        assert res_dec == 20.0
        assert engine.last_execution_mode == "CUDA_GRAPH"

        # Prompt prefill (seq_len=512)
        res_pref = engine(10.0, seq_len=512)
        assert res_pref == 20.0
        assert engine.last_execution_mode == "FUSED_TRITON"
