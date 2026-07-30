"""
TENSORGRAPH Hybrid Execution Engine.

Host-aware hybrid engine routing execution between CUDA Graph stream capture
(for single-token autoregressive decoding) and Fused Triton kernels (for prompt prefill).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .codegen.cuda import PyTorchCUDAGraphWrapper
from .hardware import HardwareCapabilities, get_hardware_capabilities


class HybridEngine:
    """
    Host-aware hybrid execution engine.

    Automatically routes model evaluation depending on host hardware capabilities
    and sequence length parameters (CUDA Graph for decoding vs Fused Triton for prefill).
    """

    def __init__(
        self,
        model_fn: Callable[..., Any],
        hardware_caps: Optional[HardwareCapabilities] = None,
        auto_capture_graph: bool = True,
    ) -> None:
        self.model_fn = model_fn
        self.caps = hardware_caps or get_hardware_capabilities()
        self.auto_capture_graph = auto_capture_graph
        self._graph_wrapper: Optional[PyTorchCUDAGraphWrapper] = None
        self.last_execution_mode: str = "UNINITIALIZED"
        self.last_latency_ms: float = 0.0

    def execute(self, inputs: Any, seq_len: int = 1, batch_size: int = 1) -> Any:
        """
        Execute model function using host-optimal routing.

        Args:
            inputs: Input tensor or tuple of tensors
            seq_len: Current sequence length (seq_len <= 8 triggers CUDA Graph)
            batch_size: Batch size

        Returns:
            Output tensor or computation result
        """
        mode = self.caps.get_optimal_execution_mode(seq_len=seq_len, batch_size=batch_size)
        self.last_execution_mode = mode

        start_time = time.perf_counter()

        if mode == "CUDA_GRAPH":
            if self._graph_wrapper is None and self.auto_capture_graph:
                self._graph_wrapper = PyTorchCUDAGraphWrapper(self.model_fn)
                self._graph_wrapper.capture(inputs)

            if self._graph_wrapper is not None:
                result = self._graph_wrapper(inputs)
            else:
                result = self.model_fn(inputs)

        else:  # FUSED_TRITON or EAGER_FALLBACK
            result = self.model_fn(inputs)

        self.last_latency_ms = (time.perf_counter() - start_time) * 1000.0
        return result

    def __call__(self, inputs: Any, seq_len: int = 1, batch_size: int = 1) -> Any:
        return self.execute(inputs, seq_len=seq_len, batch_size=batch_size)
