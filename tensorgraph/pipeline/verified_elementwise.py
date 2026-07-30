"""Verified, bounded FX -> TENSORGRAPH -> Triton compiler path.

This module intentionally supports only linear unary elementwise graphs. It
fails closed for parameters, branching, mutation, unsupported functions, and
non-linear dataflow. Generated source is executed directly on CUDA/Triton; no
handwritten substitute kernel is used.
"""

from __future__ import annotations

import hashlib
import linecache
import operator
import time
from dataclasses import dataclass
from typing import Any

from ..egraph import EGraph, Trace
from ..egraph.extract import Extractor
from ..egraph.saturation import saturate
from ..ir import Box, Expr, Id, Seq, pretty
from ..rewrite import PBox, PSeq, PVar, Rewrite
from ..signature import Signature
from ..types import Obj

_SUPPORTED_OPS = frozenset({"ReLU", "Sigmoid", "Tanh", "Neg", "Exp", "Log"})
_KERNEL_NAME = "tensorgraph_elementwise_kernel"


@dataclass(frozen=True)
class CompiledElementwise:
    """Compilation artifact for the bounded elementwise pipeline."""

    graph_module: Any
    signature: Signature
    source_expr: Expr
    optimized_expr: Expr
    rewrite_summary: dict[str, int]
    generated_source: str
    source_sha256: str
    phase_ns: dict[str, int]

    @property
    def source_pretty(self) -> str:
        return pretty(self.source_expr)

    @property
    def optimized_pretty(self) -> str:
        return pretty(self.optimized_expr)


@dataclass
class GeneratedElementwiseKernel:
    """Loaded Triton kernel generated from one ``CompiledElementwise`` artifact."""

    kernel: Any
    source_sha256: str

    def run(self, x: Any, block_size: int = 256) -> Any:
        import torch
        import triton

        if not isinstance(x, torch.Tensor):
            raise TypeError("generated kernel input must be a torch.Tensor")
        if not x.is_cuda:
            raise RuntimeError("generated TENSORGRAPH execution requires a CUDA tensor")
        if not x.is_contiguous():
            raise ValueError("generated TENSORGRAPH execution requires contiguous input")
        if not x.dtype.is_floating_point:
            raise TypeError("generated TENSORGRAPH execution requires a floating dtype")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")

        output = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements == 0:
            return output
        grid = (triton.cdiv(n_elements, block_size),)
        self.kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
        return output


def _compose(ops: list[str], tensor_obj: Obj) -> Expr:
    if not ops:
        return Id(tensor_obj)

    expression: Expr = Box(ops[0])
    for op in ops[1:]:
        expression = Seq(expression, Box(op))
    return expression


def _module_op(module: Any) -> str | None:
    import torch.nn as nn

    if isinstance(module, nn.Identity):
        return None
    if isinstance(module, nn.ReLU):
        if module.inplace:
            raise ValueError("in-place ReLU is outside the verified pipeline")
        return "ReLU"
    if isinstance(module, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(module, nn.Tanh):
        return "Tanh"
    raise ValueError(f"unsupported FX module in verified pipeline: {type(module).__name__}")


def _function_op(target: Any, kwargs: dict[str, Any]) -> str | None:
    import torch
    import torch.nn.functional as functional

    if target in {torch.relu, functional.relu}:
        if kwargs.get("inplace", False):
            raise ValueError("in-place ReLU is outside the verified pipeline")
        return "ReLU"
    if target in {torch.sigmoid, functional.sigmoid}:
        return "Sigmoid"
    if target in {torch.tanh, functional.tanh}:
        return "Tanh"
    if target in {torch.neg, operator.neg}:
        return "Neg"
    if target is torch.exp:
        return "Exp"
    if target is torch.log:
        return "Log"
    raise ValueError(f"unsupported FX function in verified pipeline: {target}")


def _method_op(target: str, kwargs: dict[str, Any]) -> str | None:
    if target == "relu":
        if kwargs.get("inplace", False):
            raise ValueError("in-place ReLU is outside the verified pipeline")
        return "ReLU"
    mapping = {
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "neg": "Neg",
        "exp": "Exp",
        "log": "Log",
    }
    if target in mapping:
        return mapping[target]
    raise ValueError(f"unsupported FX method in verified pipeline: {target}")


def _linear_fx_ops(graph_module: Any) -> list[str]:
    """Return canonical operations or reject non-linear/unsupported FX graphs."""

    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(placeholders) != 1 or len(outputs) != 1:
        raise ValueError("verified pipeline requires exactly one input and one output")

    current = placeholders[0]
    ops: list[str] = []

    for node in nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            raise ValueError("parameters and captured attributes are outside the verified pipeline")
        if not node.args or node.args[0] is not current:
            raise ValueError("verified pipeline requires a single linear unary dataflow chain")
        if any(arg is not current for arg in node.args if hasattr(arg, "op")):
            raise ValueError("verified pipeline rejects multi-input or branching nodes")

        if node.op == "call_module":
            op = _module_op(graph_module.get_submodule(str(node.target)))
        elif node.op == "call_function":
            op = _function_op(node.target, dict(node.kwargs))
        elif node.op == "call_method":
            op = _method_op(str(node.target), dict(node.kwargs))
        else:
            raise ValueError(f"unsupported FX node kind in verified pipeline: {node.op}")

        if len(node.args) != 1:
            raise ValueError("verified pipeline requires unary operations")
        if op is not None:
            ops.append(op)
        current = node

    if outputs[0].args[0] is not current:
        raise ValueError("FX output does not return the terminal unary-chain value")
    return ops


def _rules(tensor_obj: Obj) -> list[Rewrite]:
    del tensor_obj
    origin = "pointwise max(0, max(0, x)) = max(0, x)"
    return [
        Rewrite(
            name="relu_idempotence",
            lhs=PSeq(PBox("ReLU"), PBox("ReLU")),
            rhs=PBox("ReLU"),
            origin=origin,
        ),
        Rewrite(
            name="relu_idempotence",
            lhs=PSeq(PBox("ReLU"), PSeq(PBox("ReLU"), PVar("tail"))),
            rhs=PSeq(PBox("ReLU"), PVar("tail")),
            origin=origin,
        ),
    ]


def _emit_triton(expr: Expr, kernel_name: str = _KERNEL_NAME) -> str:
    """Emit a complete Triton module for a verified unary expression."""

    operations: list[str] = []

    def collect(term: Expr) -> None:
        if isinstance(term, Id):
            return
        if isinstance(term, Box):
            if term.op not in _SUPPORTED_OPS:
                raise ValueError(f"unsupported operation during verified lowering: {term.op}")
            operations.append(term.op)
            return
        if isinstance(term, Seq):
            collect(term.first)
            collect(term.second)
            return
        raise ValueError(f"verified lowering accepts only Id, Box, and Seq; got {type(term).__name__}")

    collect(expr)
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        f"def {kernel_name}(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):",
        "    pid = tl.program_id(axis=0)",
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n_elements",
        "    value = tl.load(x_ptr + offsets, mask=mask, other=0.0)",
    ]

    for op in operations:
        if op == "ReLU":
            lines.append("    value = tl.where(value > 0.0, value, 0.0)")
        elif op == "Sigmoid":
            lines.append("    value = tl.sigmoid(value)")
        elif op == "Tanh":
            lines.append("    value = 2.0 * tl.sigmoid(2.0 * value) - 1.0")
        elif op == "Neg":
            lines.append("    value = -value")
        elif op == "Exp":
            lines.append("    value = tl.exp(value)")
        elif op == "Log":
            lines.append("    value = tl.log(value)")
        else:  # pragma: no cover - collect() rejects this first
            raise AssertionError(op)

    lines.append("    tl.store(y_ptr + offsets, value, mask=mask)")
    lines.append("")
    return "\n".join(lines)


def compile_fx_elementwise(model: Any) -> CompiledElementwise:
    """Compile a supported PyTorch module into a generated Triton artifact."""

    from torch.fx import symbolic_trace

    phase_ns: dict[str, int] = {}

    start = time.perf_counter_ns()
    graph_module = symbolic_trace(model)
    phase_ns["fx_capture"] = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    ops = _linear_fx_ops(graph_module)
    tensor_obj = Obj("Tensor")
    signature = Signature()
    for op in sorted(set(ops)):
        signature.add(op, tensor_obj, tensor_obj, traits={"elementwise"})
    source_expr = _compose(ops, tensor_obj)
    phase_ns["ir_construction"] = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    egraph = EGraph(signature)
    root = egraph.add_expr(source_expr)
    trace = Trace()
    applicable_rules = _rules(tensor_obj) if "ReLU" in signature else []
    saturate(egraph, applicable_rules, iters=8, max_applications=1_000, trace=trace)
    phase_ns["saturation"] = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    extractor = Extractor(egraph)
    extractor.solve(root)
    optimized_expr = extractor.extract(root)
    phase_ns["extraction"] = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    generated_source = _emit_triton(optimized_expr)
    phase_ns["source_generation"] = time.perf_counter_ns() - start
    source_sha256 = hashlib.sha256(generated_source.encode("utf-8")).hexdigest()

    return CompiledElementwise(
        graph_module=graph_module,
        signature=signature,
        source_expr=source_expr,
        optimized_expr=optimized_expr,
        rewrite_summary=trace.summary(),
        generated_source=generated_source,
        source_sha256=source_sha256,
        phase_ns=phase_ns,
    )


def load_generated_kernel(artifact: CompiledElementwise) -> GeneratedElementwiseKernel:
    """Load the exact generated source as a Triton JIT kernel.

    This function fails closed when CUDA or Triton is unavailable. Registering
    the generated source in ``linecache`` allows Triton's JIT decorator to
    inspect dynamically compiled source without writing a substitute file.
    """

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute generated TENSORGRAPH code")
    try:
        import triton  # noqa: F401
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Triton is required to execute generated TENSORGRAPH code") from exc

    actual_sha = hashlib.sha256(artifact.generated_source.encode("utf-8")).hexdigest()
    if actual_sha != artifact.source_sha256:
        raise RuntimeError("generated source identity does not match the compilation artifact")

    filename = f"<tensorgraph:{artifact.source_sha256}>"
    source_lines = artifact.generated_source.splitlines(keepends=True)
    linecache.cache[filename] = (
        len(artifact.generated_source),
        None,
        source_lines,
        filename,
    )
    namespace: dict[str, Any] = {}
    exec(compile(artifact.generated_source, filename, "exec"), namespace)
    kernel = namespace.get(_KERNEL_NAME)
    if kernel is None:
        raise RuntimeError("generated source did not define the expected Triton kernel")
    return GeneratedElementwiseKernel(kernel=kernel, source_sha256=artifact.source_sha256)
