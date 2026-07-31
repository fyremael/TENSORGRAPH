"""Compile, load, launch, and CUDA-Graph replay exact generated native CUDA.

There is no CPU, eager, Triton, or handwritten-kernel fallback in this module.
Failure to satisfy the compiler, device, ABI, or graph-capture contract raises
before positive execution evidence can be produced.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from ..codegen.native_cuda import NativeCUDAArtifact

_CPP_DECLARATIONS = r"""
#include <torch/extension.h>

torch::Tensor tensorgraph_native_run(torch::Tensor input);
void tensorgraph_native_run_out(torch::Tensor input, torch::Tensor output);
"""

_DTYPE_NAMES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for native CUDA execution") from exc
    return torch


def _expected_dtype(torch: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:  # pragma: no cover - artifact constructor rejects first
        raise RuntimeError(f"unsupported artifact dtype: {dtype_name}") from exc


def _validate_tensor(torch: Any, artifact: NativeCUDAArtifact, tensor: Any) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("native CUDA input must be a torch.Tensor")
    if not tensor.is_cuda:
        raise RuntimeError("native CUDA execution requires a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError("native CUDA execution requires contiguous input")
    if tensor.dtype != _expected_dtype(torch, artifact.dtype):
        raise TypeError(
            f"native CUDA artifact expects {_DTYPE_NAMES[artifact.dtype]}, got {tensor.dtype}"
        )
    if artifact.dtype == "bfloat16":
        major, _minor = torch.cuda.get_device_capability(tensor.device)
        if major < 8:
            raise RuntimeError("bfloat16 native CUDA execution requires Ampere-or-newer hardware")


def _module_name(artifact: NativeCUDAArtifact) -> str:
    identity = hashlib.sha256(
        (_CPP_DECLARATIONS + "\n" + artifact.generated_source).encode("utf-8")
    ).hexdigest()
    return f"tensorgraph_native_{identity[:20]}"


@dataclass
class NativeCUDAExecutable:
    """Loaded exact native-CUDA artifact."""

    artifact: NativeCUDAArtifact
    module: Any
    phase_ns: dict[str, int]
    compiler_identity: dict[str, object]

    def run(self, input_tensor: Any) -> Any:
        torch = _require_torch()
        _validate_tensor(torch, self.artifact, input_tensor)
        return self.module.tensorgraph_native_run(input_tensor)

    def run_out(self, input_tensor: Any, output_tensor: Any) -> None:
        torch = _require_torch()
        _validate_tensor(torch, self.artifact, input_tensor)
        _validate_tensor(torch, self.artifact, output_tensor)
        if input_tensor.device != output_tensor.device:
            raise ValueError("native CUDA input and output must use the same device")
        if input_tensor.shape != output_tensor.shape:
            raise ValueError("native CUDA input and output shapes must match")
        if input_tensor.data_ptr() == output_tensor.data_ptr():
            raise ValueError("native CUDA input and output must not alias")
        self.module.tensorgraph_native_run_out(input_tensor, output_tensor)

    def capture(self, example_input: Any, *, warmup: int = 3) -> NativeCUDAGraph:
        return NativeCUDAGraph.capture(self, example_input, warmup=warmup)


@dataclass
class NativeCUDAGraph:
    """Static-buffer CUDA Graph for one loaded native-CUDA artifact."""

    executable: NativeCUDAExecutable
    graph: Any
    static_input: Any
    static_output: Any
    capture_stream: Any
    capture_ns: int

    @classmethod
    def capture(
        cls,
        executable: NativeCUDAExecutable,
        example_input: Any,
        *,
        warmup: int = 3,
    ) -> NativeCUDAGraph:
        torch = _require_torch()
        _validate_tensor(torch, executable.artifact, example_input)
        if warmup < 1:
            raise ValueError("warmup must be at least one")

        static_input = torch.empty_like(example_input)
        static_output = torch.empty_like(example_input)
        static_input.copy_(example_input)
        capture_stream = torch.cuda.Stream(device=example_input.device)
        current_stream = torch.cuda.current_stream(example_input.device)
        capture_stream.wait_stream(current_stream)

        with torch.cuda.stream(capture_stream):
            for _ in range(warmup):
                executable.run_out(static_input, static_output)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(example_input.device)

        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter_ns()
        with torch.cuda.graph(graph, stream=capture_stream):
            executable.run_out(static_input, static_output)
        torch.cuda.synchronize(example_input.device)
        capture_ns = time.perf_counter_ns() - start
        return cls(
            executable=executable,
            graph=graph,
            static_input=static_input,
            static_output=static_output,
            capture_stream=capture_stream,
            capture_ns=capture_ns,
        )

    def replay(self, input_tensor: Any, *, clone_output: bool = False) -> Any:
        torch = _require_torch()
        _validate_tensor(torch, self.executable.artifact, input_tensor)
        if input_tensor.device != self.static_input.device:
            raise ValueError("captured graph device cannot change")
        if input_tensor.shape != self.static_input.shape:
            raise ValueError("captured graph shape cannot change")
        if input_tensor.data_ptr() == self.static_output.data_ptr():
            raise ValueError("replay input must not alias the static output")

        self.static_input.copy_(input_tensor)
        self.graph.replay()
        if clone_output:
            return self.static_output.clone()
        return self.static_output


def compile_native_cuda(
    artifact: NativeCUDAArtifact,
    *,
    verbose: bool = False,
    extra_cuda_cflags: tuple[str, ...] = (),
) -> NativeCUDAExecutable:
    """Compile and load the exact generated source as a PyTorch CUDA extension."""

    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compile native TENSORGRAPH code")

    try:
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch CUDA extension tooling is unavailable") from exc
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit was not found; native CUDA compilation is unavailable")

    actual_sha = hashlib.sha256(artifact.generated_source.encode("utf-8")).hexdigest()
    if actual_sha != artifact.source_sha256:
        raise RuntimeError("generated native CUDA source identity mismatch")

    if artifact.dtype == "bfloat16":
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError("bfloat16 native CUDA compilation requires Ampere-or-newer hardware")

    module_name = _module_name(artifact)
    phase_ns: dict[str, int] = {}
    start = time.perf_counter_ns()
    module = load_inline(
        name=module_name,
        cpp_sources=[_CPP_DECLARATIONS],
        cuda_sources=[artifact.generated_source],
        functions=["tensorgraph_native_run", "tensorgraph_native_run_out"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--std=c++17", *extra_cuda_cflags],
        with_cuda=True,
        verbose=verbose,
    )
    phase_ns["compile_and_load"] = time.perf_counter_ns() - start

    compiler_identity: dict[str, object] = {
        "module_name": module_name,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_home": str(CUDA_HOME),
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "extra_cuda_cflags": ["-O3", "--std=c++17", *extra_cuda_cflags],
    }
    return NativeCUDAExecutable(
        artifact=artifact,
        module=module,
        phase_ns=phase_ns,
        compiler_identity=compiler_identity,
    )
