"""
TENSORGRAPH Codegen: Triton Emitter.
Generates Triton kernels from diagrammatic expressions.
Supports: elementwise ops, reduction ops (Sum, Mean, Softmax)
"""
from __future__ import annotations

from typing import Any
from ..ir import Expr, Box, Par, Seq, Id
from ..signature import Signature

class TritonEmitter:
    def __init__(self, sig: Signature) -> None:
        self.sig = sig
        self.code: list[str] = []
        self.indent: int = 0
        self._var_counter: int = 0
        
    def _fresh_var(self, prefix: str = "v") -> str:
        """Generate a fresh variable name."""
        self._var_counter += 1
        return f"{prefix}{self._var_counter}"
        
    def emit(self, expr: Expr, kernel_name: str = "fused_kernel") -> str:
        """Generate a full Triton kernel module for the given expression."""
        self.code = []
        self._var_counter = 0
        
        # Analyze expression to determine kernel type
        is_reduction = self._has_reduction(expr)
        
        self.code.append("import torch")
        self.code.append("import triton")
        self.code.append("import triton.language as tl")
        self.code.append("")
        
        self.code.append(f"@triton.jit")
        
        if is_reduction:
            # Reduction kernel: single block processes all elements
            self.code.append(f"def {kernel_name}(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):")
            self.indent = 1
            
            self._writeln("# Reduction kernel - single program processes all")
            self._writeln("offsets = tl.arange(0, BLOCK_SIZE)")
            self._writeln("mask = offsets < n_elements")
            self._writeln("x0 = tl.load(x_ptr + offsets, mask=mask, other=0.0)")
            
            inputs = ["x0"]
            results = self._visit(expr, inputs)
            
            # Reduction result is scalar
            self._writeln(f"tl.store(y_ptr, {results[0]})")
        else:
            # Elementwise kernel (original logic)
            self.code.append(f"def {kernel_name}(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):")
            self.indent = 1
            
            self._writeln("pid = tl.program_id(axis=0)")
            self._writeln("block_start = pid * BLOCK_SIZE")
            self._writeln("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
            self._writeln("mask = offsets < n_elements")
            self._writeln("x0 = tl.load(x_ptr + offsets, mask=mask)")
            
            inputs = ["x0"]
            results = self._visit(expr, inputs)
            
            if len(results) == 1:
                self._writeln(f"tl.store(y_ptr + offsets, {results[0]}, mask=mask)")
        
        return "\n".join(self.code)
    
    def _has_reduction(self, expr: Expr) -> bool:
        """Check if expression contains any reduction operations."""
        if isinstance(expr, Box):
            op = self.sig.get(expr.op)
            return "reduction" in op.traits
        if isinstance(expr, Seq):
            return self._has_reduction(expr.first) or self._has_reduction(expr.second)
        if isinstance(expr, Par):
            return self._has_reduction(expr.left) or self._has_reduction(expr.right)
        return False

    def _visit(self, expr: Expr, inputs: list[str]) -> list[str]:
        from ..ir.normalize import infer_type
        
        if isinstance(expr, Box):
            op = self.sig.get(expr.op)
            input_var = inputs[0] if inputs else "x0"
            
            # Reduction operations
            if "reduction" in op.traits:
                if op.name == "Sum":
                    return [f"tl.sum({input_var}, axis=0)"]
                if op.name == "Mean":
                    sum_var = self._fresh_var("sum")
                    self._writeln(f"{sum_var} = tl.sum({input_var}, axis=0)")
                    # n_elements comes from kernel param
                    return [f"{sum_var} / n_elements"]
                if op.name == "Max":
                    return [f"tl.max({input_var}, axis=0)"]
                if op.name == "Min":
                    return [f"tl.min({input_var}, axis=0)"]
                return [f"tl.sum({input_var}, axis=0)"]  # Default reduction
            
            # Elementwise operations
            if "elementwise" in op.traits:
                if op.name in ("Relu", "ReLU"):
                    return [f"tl.where({input_var} > 0, {input_var}, 0)"]
                if op.name == "Sigmoid":
                    return [f"tl.sigmoid({input_var})"]
                if op.name == "Exp":
                    return [f"tl.exp({input_var})"]
                if op.name == "Log":
                    return [f"tl.log({input_var})"]
                if op.name == "Neg":
                    return [f"-{input_var}"]
                return [f"{op.name.lower()}({input_var})"]
            
            return inputs
            
        if isinstance(expr, Seq):
            # x -> f -> g
            intermediate = self._visit(expr.first, inputs)
            return self._visit(expr.second, intermediate)
            
        if isinstance(expr, Par):
            # Split inputs based on domain sort of left child
            dom_l, _ = infer_type(expr.left, self.sig)
            
            def width(o: Any) -> int:
                if o.is_tensor():
                    return width(o.left) + width(o.right)
                if o.name == "I": return 0
                return 1
                
            n_left = width(dom_l)
            
            inputs_l = inputs[:n_left]
            inputs_r = inputs[n_left:]
            
            res_l = self._visit(expr.left, inputs_l)
            res_r = self._visit(expr.right, inputs_r)
            
            return res_l + res_r
            
        if isinstance(expr, Id):
            return inputs
            
        raise NotImplementedError(f"Unsupported expr for fusion: {expr}")

    def _writeln(self, line: str) -> None:
        self.code.append("    " * self.indent + line)


def emit_softmax(sig: Signature) -> str:
    """
    Generate a fused Softmax kernel.
    Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    code = []
    code.append("import torch")
    code.append("import triton")
    code.append("import triton.language as tl")
    code.append("")
    code.append("@triton.jit")
    code.append("def fused_softmax(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):")
    code.append("    # Load all elements")
    code.append("    offsets = tl.arange(0, BLOCK_SIZE)")
    code.append("    mask = offsets < n_elements")
    code.append("    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))")
    code.append("    # Numerically stable softmax")
    code.append("    x_max = tl.max(x, axis=0)")
    code.append("    x_shifted = x - x_max")
    code.append("    x_exp = tl.exp(x_shifted)")
    code.append("    x_sum = tl.sum(x_exp, axis=0)")
    code.append("    y = x_exp / x_sum")
    code.append("    tl.store(y_ptr + offsets, y, mask=mask)")
    return "\n".join(code)

