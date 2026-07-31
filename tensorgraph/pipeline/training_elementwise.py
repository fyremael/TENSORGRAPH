"""Bounded forward/backward support for generated unary elementwise kernels.

The training path is intentionally narrow. It accepts only Sigmoid or Tanh,
optionally preceded by one ReLU after equality-saturation. The forward kernel is
the exact artifact produced by :mod:`verified_elementwise`; the backward kernel
is generated from that optimized expression and consumes the exact forward
output. No source mutation or handwritten substitute is admitted.
"""

from __future__ import annotations

import hashlib
import linecache
import time
from dataclasses import dataclass
from typing import Any

from ..ir import Box, Expr, Id, Seq
from .verified_elementwise import (
    CompiledElementwise,
    GeneratedElementwiseKernel,
    compile_fx_elementwise,
    load_generated_kernel,
)

_BACKWARD_KERNEL_NAME = "tensorgraph_elementwise_backward_kernel"


@dataclass(frozen=True)
class CompiledElementwiseTraining:
    """Forward artifact plus generated input-gradient source."""

    forward: CompiledElementwise
    optimized_ops: tuple[str, ...]
    terminal_op: str
    generated_backward_source: str
    backward_source_sha256: str
    backward_generation_ns: int


@dataclass
class GeneratedElementwiseBackwardKernel:
    """Loaded generated input-gradient kernel for a bounded unary graph."""

    kernel: Any
    source_sha256: str

    def run(
        self,
        x: Any,
        y: Any,
        grad_output: Any,
        block_size: int = 256,
    ) -> Any:
        import torch
        import triton

        tensors = {"x": x, "y": y, "grad_output": grad_output}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"generated backward {name} must be a torch.Tensor")
            if not value.is_cuda:
                raise RuntimeError("generated TENSORGRAPH backward requires CUDA tensors")
            if not value.is_contiguous():
                raise ValueError("generated TENSORGRAPH backward requires contiguous tensors")
            if not value.dtype.is_floating_point:
                raise TypeError("generated TENSORGRAPH backward requires floating dtypes")

        if x.shape != y.shape or x.shape != grad_output.shape:
            raise ValueError("generated backward tensors must have identical shapes")
        if x.dtype != y.dtype or x.dtype != grad_output.dtype:
            raise TypeError("generated backward tensors must have identical dtypes")
        if x.device != y.device or x.device != grad_output.device:
            raise ValueError("generated backward tensors must be on the same device")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")

        grad_input = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements == 0:
            return grad_input
        grid = (triton.cdiv(n_elements, block_size),)
        self.kernel[grid](
            x,
            y,
            grad_output,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        return grad_input


@dataclass
class GeneratedElementwiseTraining:
    """Loaded exact forward and generated backward kernels."""

    forward: GeneratedElementwiseKernel
    backward: GeneratedElementwiseBackwardKernel


def _collect_ops(expr: Expr) -> tuple[str, ...]:
    operations: list[str] = []

    def collect(term: Expr) -> None:
        if isinstance(term, Id):
            return
        if isinstance(term, Box):
            operations.append(term.op)
            return
        if isinstance(term, Seq):
            collect(term.first)
            collect(term.second)
            return
        raise ValueError(
            "training lowering accepts only Id, Box, and Seq; "
            f"got {type(term).__name__}"
        )

    collect(expr)
    return tuple(operations)


def _validate_training_ops(operations: tuple[str, ...]) -> str:
    allowed = {
        ("Sigmoid",): "Sigmoid",
        ("Tanh",): "Tanh",
        ("ReLU", "Sigmoid"): "Sigmoid",
        ("ReLU", "Tanh"): "Tanh",
    }
    try:
        return allowed[operations]
    except KeyError as exc:
        raise ValueError(
            "bounded training lowering requires Sigmoid or Tanh, optionally preceded "
            f"by one optimized ReLU; got {operations!r}"
        ) from exc


def _emit_backward_triton(
    operations: tuple[str, ...],
    kernel_name: str = _BACKWARD_KERNEL_NAME,
) -> str:
    terminal_op = _validate_training_ops(operations)
    has_relu = operations[0] == "ReLU"

    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        (
            f"def {kernel_name}(x_ptr, y_ptr, grad_output_ptr, grad_input_ptr, "
            "n_elements, BLOCK_SIZE: tl.constexpr):"
        ),
        "    pid = tl.program_id(axis=0)",
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n_elements",
        "    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)",
        "    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)",
        "    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)",
    ]
    if terminal_op == "Sigmoid":
        lines.append("    derivative = y * (1.0 - y)")
    else:
        lines.append("    derivative = 1.0 - y * y")
    if has_relu:
        lines.append(
            "    derivative = tl.where((x > 0.0) | (x != x), derivative, 0.0)"
        )
    lines.extend(
        [
            "    grad_input = grad_output * derivative",
            "    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)",
            "",
        ]
    )
    return "\n".join(lines)


def compile_fx_elementwise_training(model: Any) -> CompiledElementwiseTraining:
    """Compile a supported model for exact generated forward and input gradient."""

    forward = compile_fx_elementwise(model)
    optimized_ops = _collect_ops(forward.optimized_expr)
    terminal_op = _validate_training_ops(optimized_ops)

    start = time.perf_counter_ns()
    generated_backward_source = _emit_backward_triton(optimized_ops)
    backward_generation_ns = time.perf_counter_ns() - start
    backward_source_sha256 = hashlib.sha256(
        generated_backward_source.encode("utf-8")
    ).hexdigest()

    return CompiledElementwiseTraining(
        forward=forward,
        optimized_ops=optimized_ops,
        terminal_op=terminal_op,
        generated_backward_source=generated_backward_source,
        backward_source_sha256=backward_source_sha256,
        backward_generation_ns=backward_generation_ns,
    )


def load_generated_backward_kernel(
    artifact: CompiledElementwiseTraining,
) -> GeneratedElementwiseBackwardKernel:
    """Load the exact generated backward source and fail closed without CUDA."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute generated TENSORGRAPH backward code")
    try:
        import triton  # noqa: F401
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Triton is required to execute generated backward code") from exc

    actual_sha = hashlib.sha256(
        artifact.generated_backward_source.encode("utf-8")
    ).hexdigest()
    if actual_sha != artifact.backward_source_sha256:
        raise RuntimeError("generated backward source identity does not match the artifact")

    filename = f"<tensorgraph-backward:{artifact.backward_source_sha256}>"
    source_lines = artifact.generated_backward_source.splitlines(keepends=True)
    linecache.cache[filename] = (
        len(artifact.generated_backward_source),
        None,
        source_lines,
        filename,
    )
    namespace: dict[str, Any] = {}
    exec(compile(artifact.generated_backward_source, filename, "exec"), namespace)
    kernel = namespace.get(_BACKWARD_KERNEL_NAME)
    if kernel is None:
        raise RuntimeError("generated source did not define the expected backward kernel")
    return GeneratedElementwiseBackwardKernel(
        kernel=kernel,
        source_sha256=artifact.backward_source_sha256,
    )


def load_generated_training(
    artifact: CompiledElementwiseTraining,
) -> GeneratedElementwiseTraining:
    """Load exact generated forward and backward kernels."""

    return GeneratedElementwiseTraining(
        forward=load_generated_kernel(artifact.forward),
        backward=load_generated_backward_kernel(artifact),
    )
