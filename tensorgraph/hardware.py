"""
TENSORGRAPH Hardware & Host Awareness Module.

Probes host system capabilities (GPU model, VRAM, memory bandwidth, Triton/CUDA support)
and determines optimal execution routing for model inference.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Optional


@dataclasses.dataclass
class HardwareCapabilities:
    """Captured host hardware capabilities and optimal engine settings."""

    has_cuda: bool = False
    gpu_name: str = "CPU / Fallback"
    vram_gb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    peak_bandwidth_gbps: float = 0.0
    has_triton: bool = False
    has_cuda_graph: bool = False

    def get_optimal_execution_mode(self, seq_len: int = 1, batch_size: int = 1) -> str:
        """
        Determine optimal execution mode for a given tensor sequence length.

        Returns:
            "CUDA_GRAPH" - for single-token decoding loops (seq_len <= 8)
            "FUSED_TRITON" - for prompt prefill / batch processing (seq_len > 8)
            "EAGER_FALLBACK" - when CUDA is unavailable
        """
        if not self.has_cuda:
            return "EAGER_FALLBACK"

        # Autoregressive decoding phase: CUDA Graph Stream Capture eliminates launch overhead
        if seq_len <= 8 and self.has_cuda_graph:
            return "CUDA_GRAPH"

        # Prompt prefill phase: Fused Triton maximizes HBM DRAM throughput
        if seq_len > 8 and self.has_triton:
            return "FUSED_TRITON"

        return "CUDA_GRAPH" if self.has_cuda_graph else "EAGER_FALLBACK"


_CACHED_HARDWARE: Optional[HardwareCapabilities] = None


def get_hardware_capabilities(force_refresh: bool = False) -> HardwareCapabilities:
    """Probe and return the host hardware capabilities profile (cached singleton)."""
    global _CACHED_HARDWARE
    if _CACHED_HARDWARE is not None and not force_refresh:
        return _CACHED_HARDWARE

    caps = HardwareCapabilities()

    # Check Triton support
    try:
        import triton
        caps.has_triton = True
    except ImportError:
        caps.has_triton = False

    # Check CUDA support via PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            caps.has_cuda = True
            caps.gpu_name = torch.cuda.get_device_name(0)
            caps.vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            caps.compute_capability = torch.cuda.get_device_capability(0)
            caps.has_cuda_graph = hasattr(torch.cuda, "CUDAGraph")

            # Estimate peak HBM bandwidth by GPU model heuristics
            name_upper = caps.gpu_name.upper()
            if "A100" in name_upper:
                caps.peak_bandwidth_gbps = 1555.0  # SXM4 40GB/80GB
            elif "H100" in name_upper:
                caps.peak_bandwidth_gbps = 3350.0
            elif "T4" in name_upper:
                caps.peak_bandwidth_gbps = 300.0
            elif "L4" in name_upper:
                caps.peak_bandwidth_gbps = 300.0
            elif "V100" in name_upper:
                caps.peak_bandwidth_gbps = 900.0
            elif "3090" in name_upper or "4090" in name_upper:
                caps.peak_bandwidth_gbps = 1008.0
            else:
                caps.peak_bandwidth_gbps = 300.0
    except Exception:
        pass

    _CACHED_HARDWARE = caps
    return caps
