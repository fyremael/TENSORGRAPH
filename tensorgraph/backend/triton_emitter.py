from tensorgraph.ir import Expr, Box, Seq, Par, Dup, Del, Id
from tensorgraph.ir.primitives import Swap
from typing import List, Tuple

class TritonEmitter:
    """
    Emits Triton kernel code from TENSORGRAPH IR expressions.
    Currently supports Elementwise operations.
    """
    
    def __init__(self):
        self.code_lines = []
        self.counter = 0
        self.indent_level = 1
        
    def _new_var(self, prefix="t"):
        self.counter += 1
        return f"{prefix}{self.counter}"
    
    def _emit_line(self, line):
        self.code_lines.append("    " * self.indent_level + line)

    def get_arity(self, expr: Expr) -> Tuple[int, int]:
        """Returns (in_arity, out_arity) for an expression."""
        if isinstance(expr, Box):
            op = expr.op.lower()
            if op in ["add", "mul", "sub", "div"]: return (2, 1)
            if op in ["relu", "gelu", "sigmoid"]: return (1, 1)
            if op in ["linear"]: raise NotImplementedError("Linear needs matrix emission")
            return (1, 1) # Default unary
            
        elif isinstance(expr, Seq):
            in1, out1 = self.get_arity(expr.first)
            in2, out2 = self.get_arity(expr.second)
            # Connectivity check: out1 must == in2 for valid Seq (implicit in typed IR)
            return (in1, out2)
            
        elif isinstance(expr, Par):
            in1, out1 = self.get_arity(expr.left)
            in2, out2 = self.get_arity(expr.right)
            return (in1 + in2, out1 + out2)
            
        elif isinstance(expr, Dup):
            return (1, 2)
            
        elif isinstance(expr, Swap):
            return (2, 2)
            
        elif isinstance(expr, Del):
            return (1, 0)
            
        elif isinstance(expr, Id):
            return (1, 1)
            
        raise NotImplementedError(f"Arity for {type(expr)}")

    def emit_kernel(self, expr: Expr, kernel_name="fused_kernel"):
        """Main entry point. Generates full kernel string."""
        self.code_lines = []
        self.counter = 0
        
        in_arity, out_arity = self.get_arity(expr)
        
        # Header
        args = [f"in{i}_ptr" for i in range(in_arity)] + \
               [f"out{i}_ptr" for i in range(out_arity)] + \
               ["n_elements", "BLOCK_SIZE: tl.constexpr"]
               
        self.code_lines.append("import triton")
        self.code_lines.append("import triton.language as tl")
        self.code_lines.append("")
        self.code_lines.append(f"@triton.jit")
        self.code_lines.append(f"def {kernel_name}({', '.join(args)}):")
        
        # Preamble
        self._emit_line("pid = tl.program_id(axis=0)")
        self._emit_line("block_start = pid * BLOCK_SIZE")
        self._emit_line("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
        self._emit_line("mask = offsets < n_elements")
        self._emit_line("")
        
        # Load Inputs
        input_vars = []
        for i in range(in_arity):
            var = self._new_var("in")
            self._emit_line(f"{var} = tl.load(in{i}_ptr + offsets, mask=mask)")
            input_vars.append(var)
            
        self._emit_line("")
        self._emit_line("# Compute Body")
        
        # Generate Body
        output_vars = self.visit(expr, input_vars)
        
        self._emit_line("")
        self._emit_line("# Store Outputs")
        if len(output_vars) != out_arity:
            raise ValueError(f"Expected {out_arity} outputs, got {len(output_vars)}")
            
        for i, var in enumerate(output_vars):
            self._emit_line(f"tl.store(out{i}_ptr + offsets, {var}, mask=mask)")
            
        return "\n".join(self.code_lines)

    def visit(self, expr: Expr, inputs: List[str]) -> List[str]:
        """Recursive Code Gen."""
        
        if isinstance(expr, Box):
            return self._visit_box(expr, inputs)
            
        elif isinstance(expr, Seq):
            # inputs -> first -> middle -> second -> outputs
            mid_vars = self.visit(expr.first, inputs)
            return self.visit(expr.second, mid_vars)
            
        elif isinstance(expr, Par):
            # Split inputs based on left child requirements
            left_in_arity, _ = self.get_arity(expr.left)
            
            left_inputs = inputs[:left_in_arity]
            right_inputs = inputs[left_in_arity:]
            
            left_outs = self.visit(expr.left, left_inputs)
            right_outs = self.visit(expr.right, right_inputs)
            
            return left_outs + right_outs
            
        elif isinstance(expr, Dup):
            # A -> (A, A)
            if len(inputs) != 1: raise ValueError("Dup expects 1 input")
            return [inputs[0], inputs[0]]
            
        elif isinstance(expr, Swap):
            return [inputs[1], inputs[0]]
            
        elif isinstance(expr, Id):
            return inputs
            
        raise NotImplementedError(f"Visit not implemented for {type(expr)}")

    def _visit_box(self, box: Box, inputs: List[str]) -> List[str]:
        op = box.op.lower()
        out_var = self._new_var("tmp")
        
        if op == "add":
            self._emit_line(f"{out_var} = {inputs[0]} + {inputs[1]}")
        elif op == "mul":
            self._emit_line(f"{out_var} = {inputs[0]} * {inputs[1]}")
        elif op == "sub":
            self._emit_line(f"{out_var} = {inputs[0]} - {inputs[1]}")
        elif op == "div":
            self._emit_line(f"{out_var} = {inputs[0]} / {inputs[1]}")
        elif op == "relu":
            self._emit_line(f"{out_var} = tl.where({inputs[0]} > 0, {inputs[0]}, 0)")
        elif op == "gelu":
            # Simple approx or call lib?
            # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            # For simplicity emit a comment or simple standard
            self._emit_line(f"# GELU approx")
            self._emit_line(f"{out_var} = {inputs[0]} * 0.5 * (1.0 + tl.libdevice.tanh(0.79788456 * ({inputs[0]} + 0.044715 * {inputs[0]} * {inputs[0]} * {inputs[0]})))")
            
        else:
            self._emit_line(f"{out_var} = {op}({', '.join(inputs)}) # Generic call")
            
        return [out_var]
