"""Governed executable specialization of the historical native CUDA emitter.

The historical :class:`tensorgraph.codegen.cuda.CUDAEmitter` established source
emission for native CUDA.  This module keeps that lineage, but narrows the
executable contract to one contiguous unary tensor and emits a complete PyTorch
CUDA-extension translation unit.  The emitted kernel body is derived only from
the supplied TENSORGRAPH expression.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from ..ir import Box, Expr, Id, Seq, pretty
from ..signature import Signature
from .cuda import CUDAEmitter

NativeDType = Literal["float16", "bfloat16", "float32"]
LogDomain = Literal["strict_positive"] | None

_SUPPORTED_OPS = frozenset({"ReLU", "Neg", "Sigmoid", "Tanh", "Exp", "Log"})


@dataclass(frozen=True)
class NativeCUDAArtifact:
    """Exact generated native-CUDA source and its bounded ABI contract."""

    expression: Expr
    expression_pretty: str
    operations: tuple[str, ...]
    dtype: NativeDType
    kernel_name: str
    generated_source: str
    source_sha256: str
    log_domain: LogDomain
    abi: dict[str, object]


@dataclass(frozen=True)
class _DTypeSpec:
    storage_type: str
    scalar_type: str
    load_expr: str
    store_expr: str
    required_header: str


_DTYPE_SPECS: dict[NativeDType, _DTypeSpec] = {
    "float32": _DTypeSpec(
        storage_type="float",
        scalar_type="at::ScalarType::Float",
        load_expr="x[idx]",
        store_expr="value",
        required_header="",
    ),
    "float16": _DTypeSpec(
        storage_type="__half",
        scalar_type="at::ScalarType::Half",
        load_expr="__half2float(x[idx])",
        store_expr="__float2half_rn(value)",
        required_header="#include <cuda_fp16.h>",
    ),
    "bfloat16": _DTypeSpec(
        storage_type="__nv_bfloat16",
        scalar_type="at::ScalarType::BFloat16",
        load_expr="__bfloat162float(x[idx])",
        store_expr="__float2bfloat16_rn(value)",
        required_header="#include <cuda_bf16.h>",
    ),
}


def _collect_operations(expr: Expr) -> tuple[str, ...]:
    operations: list[str] = []

    def visit(term: Expr) -> None:
        if isinstance(term, Id):
            return
        if isinstance(term, Box):
            if term.op not in _SUPPORTED_OPS:
                raise ValueError(f"unsupported native CUDA unary operation: {term.op}")
            operations.append(term.op)
            return
        if isinstance(term, Seq):
            visit(term.first)
            visit(term.second)
            return
        raise ValueError(
            "native CUDA inference lowering accepts only Id, Box, and Seq; "
            f"got {type(term).__name__}"
        )

    visit(expr)
    return tuple(operations)


def _validate_signature(operations: tuple[str, ...], signature: Signature) -> None:
    for operation in operations:
        definition = signature.get(operation)
        if "elementwise" not in definition.traits:
            raise ValueError(f"operation {operation!r} is not declared elementwise")
        if definition.dom != definition.cod:
            raise ValueError(f"operation {operation!r} is not unary endomorphic")


def _validate_log_domain(operations: tuple[str, ...], contract: LogDomain) -> None:
    """Reject Log unless strict positivity is established at every Log site."""

    domain = "positive" if contract == "strict_positive" else "unknown"
    for operation in operations:
        if operation == "Log":
            if domain != "positive":
                raise ValueError(
                    "Log requires a strict-positive input contract or an immediately "
                    "preceding operation that guarantees strict positivity"
                )
            domain = "unknown"
        elif operation in {"Sigmoid", "Exp"}:
            domain = "positive"
        elif operation == "ReLU":
            domain = "nonnegative"
        elif operation == "Neg":
            domain = "unknown"
        elif operation == "Tanh":
            domain = "unknown"


def _lower_scalar(operations: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for operation in operations:
        if operation == "ReLU":
            lines.append("value = isnan(value) ? value : fmaxf(value, 0.0f);")
        elif operation == "Neg":
            lines.append("value = -value;")
        elif operation == "Sigmoid":
            lines.append("value = 1.0f / (1.0f + expf(-value));")
        elif operation == "Tanh":
            lines.append("value = tanhf(value);")
        elif operation == "Exp":
            lines.append("value = expf(value);")
        elif operation == "Log":
            lines.append("value = logf(value);")
        else:  # pragma: no cover - collection rejects first
            raise AssertionError(operation)
    return lines


class NativeCUDAEmitter(CUDAEmitter):
    """Executable unary specialization of the historical ``CUDAEmitter``."""

    def emit_artifact(
        self,
        expr: Expr,
        *,
        dtype: NativeDType = "float32",
        kernel_name: str = "tensorgraph_native_unary_kernel",
        log_domain: LogDomain = None,
    ) -> NativeCUDAArtifact:
        if dtype not in _DTYPE_SPECS:
            raise ValueError(f"unsupported native CUDA dtype: {dtype}")
        if not kernel_name.isidentifier():
            raise ValueError("kernel_name must be a valid C/C++ identifier")

        operations = _collect_operations(expr)
        _validate_signature(operations, self.sig)
        _validate_log_domain(operations, log_domain)
        spec = _DTYPE_SPECS[dtype]
        scalar_lines = _lower_scalar(operations)

        source_lines = [
            "#include <torch/extension.h>",
            "#include <ATen/cuda/CUDAContext.h>",
            "#include <c10/cuda/CUDAGuard.h>",
            "#include <c10/cuda/CUDAException.h>",
            "#include <cuda_runtime.h>",
            "#include <math.h>",
            "#include <cstdint>",
        ]
        if spec.required_header:
            source_lines.append(spec.required_header)
        source_lines.extend(
            [
                "",
                f"using tensorgraph_storage_t = {spec.storage_type};",
                "",
                f"__global__ void {kernel_name}(",
                "    const tensorgraph_storage_t* __restrict__ x,",
                "    tensorgraph_storage_t* __restrict__ y,",
                "    const int64_t n) {",
                "    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
                "    if (idx >= n) return;",
                f"    float value = {spec.load_expr};",
            ]
        )
        source_lines.extend(f"    {line}" for line in scalar_lines)
        source_lines.extend(
            [
                f"    y[idx] = {spec.store_expr};",
                "}",
                "",
                "void tensorgraph_native_run_out(torch::Tensor input, torch::Tensor output) {",
                "    TORCH_CHECK(input.is_cuda(), \"native CUDA input must be CUDA\");",
                "    TORCH_CHECK(output.is_cuda(), \"native CUDA output must be CUDA\");",
                "    TORCH_CHECK(input.device() == output.device(), \"input/output device mismatch\");",
                "    TORCH_CHECK(input.is_contiguous(), \"native CUDA input must be contiguous\");",
                "    TORCH_CHECK(output.is_contiguous(), \"native CUDA output must be contiguous\");",
                "    TORCH_CHECK(input.sizes().equals(output.sizes()), \"input/output shape mismatch\");",
                f"    TORCH_CHECK(input.scalar_type() == {spec.scalar_type}, \"input dtype mismatch\");",
                f"    TORCH_CHECK(output.scalar_type() == {spec.scalar_type}, \"output dtype mismatch\");",
                "    if (input.numel() == 0) return;",
                "    c10::cuda::CUDAGuard device_guard(input.device());",
                "    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());",
                "    constexpr int block_size = 256;",
                "    const int64_t grid_size = (input.numel() + block_size - 1) / block_size;",
                f"    {kernel_name}<<<grid_size, block_size, 0, stream.stream()>>>(",
                "        reinterpret_cast<const tensorgraph_storage_t*>(input.data_ptr()),",
                "        reinterpret_cast<tensorgraph_storage_t*>(output.data_ptr()),",
                "        input.numel());",
                "    C10_CUDA_KERNEL_LAUNCH_CHECK();",
                "}",
                "",
                "torch::Tensor tensorgraph_native_run(torch::Tensor input) {",
                "    auto output = torch::empty_like(input);",
                "    tensorgraph_native_run_out(input, output);",
                "    return output;",
                "}",
                "",
            ]
        )
        generated_source = "\n".join(source_lines)
        source_sha256 = hashlib.sha256(generated_source.encode("utf-8")).hexdigest()
        abi: dict[str, object] = {
            "arity": 1,
            "input_count": 1,
            "output_count": 1,
            "dtype": dtype,
            "layout": "contiguous",
            "shape_relation": "output_same_as_input",
            "aliasing": "input_output_must_not_alias",
            "stream": "current_pytorch_cuda_stream",
            "block_size": 256,
            "dynamic_extent": "numel",
            "log_domain": log_domain,
        }
        return NativeCUDAArtifact(
            expression=expr,
            expression_pretty=pretty(expr),
            operations=operations,
            dtype=dtype,
            kernel_name=kernel_name,
            generated_source=generated_source,
            source_sha256=source_sha256,
            log_domain=log_domain,
            abi=abi,
        )
