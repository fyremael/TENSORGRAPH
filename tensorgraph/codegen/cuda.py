"""
TENSORGRAPH Codegen: CUDA Emitter.
Generates raw CUDA kernels and CUDA Graph stream launchers from diagrammatic expressions.

Alternative to TritonEmitter for environments without Triton support or single-token decoding.
"""
from __future__ import annotations

from typing import Any
from ..ir import Expr, Box, Par, Seq, Id
from ..signature import Signature


class CUDAEmitter:
    """
    Generates CUDA C++ kernel code from diagrammatic expressions.
    
    Produces code compatible with nvcc compilation, including CUDA Graph capture API wrappers.
    """
    
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
        """Generate a complete CUDA kernel for the given expression."""
        self.code = []
        self._var_counter = 0
        
        is_reduction = self._has_reduction(expr)
        
        # CUDA headers
        self.code.append("#include <cuda_runtime.h>")
        self.code.append("#include <math.h>")
        self.code.append("")
        
        if is_reduction:
            return self._emit_reduction_kernel(expr, kernel_name)
        else:
            return self._emit_elementwise_kernel(expr, kernel_name)
    
    def _emit_elementwise_kernel(self, expr: Expr, kernel_name: str) -> str:
        """Generate an elementwise CUDA kernel."""
        self.code.append(f"__global__ void {kernel_name}(")
        self.code.append("    const float* __restrict__ x,")
        self.code.append("    float* __restrict__ y,")
        self.code.append("    const int n")
        self.code.append(") {")
        
        self.indent = 1
        self._writeln("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._writeln("if (idx < n) {")
        
        self.indent = 2
        self._writeln("float x0 = x[idx];")
        
        inputs = ["x0"]
        results = self._visit(expr, inputs)
        
        self._writeln(f"y[idx] = {results[0]};")
        
        self.indent = 1
        self._writeln("}")
        
        self.code.append("}")
        self.code.append("")
        
        # Add launcher function
        self._emit_launcher(kernel_name, is_reduction=False)
        self._emit_cuda_graph_launcher(kernel_name, is_reduction=False)
        
        return "\n".join(self.code)
    
    def emit_with_cuda_graph(self, expr: Expr, kernel_name: str = "fused_kernel") -> str:
        """Generate CUDA kernel and host launchers including CUDA Graph stream capture."""
        return self.emit(expr, kernel_name=kernel_name)

    def _emit_reduction_kernel(self, expr: Expr, kernel_name: str) -> str:
        """Generate a reduction CUDA kernel using shared memory."""
        self.code.append(f"__global__ void {kernel_name}(")
        self.code.append("    const float* __restrict__ x,")
        self.code.append("    float* __restrict__ y,")
        self.code.append("    const int n")
        self.code.append(") {")
        
        self.indent = 1
        self._writeln("extern __shared__ float sdata[];")
        self._writeln("")
        self._writeln("int tid = threadIdx.x;")
        self._writeln("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._writeln("")
        
        # Load with optional elementwise transform
        self._writeln("// Load and apply elementwise ops")
        self._writeln("float val = (idx < n) ? x[idx] : 0.0f;")
        
        # Apply pre-reduction elementwise ops
        pre_red_expr = self._get_pre_reduction_expr(expr)
        if pre_red_expr:
            inputs = ["val"]
            results = self._visit(pre_red_expr, inputs)
            self._writeln(f"val = {results[0]};")
        
        self._writeln("sdata[tid] = val;")
        self._writeln("__syncthreads();")
        self._writeln("")
        
        # Reduction loop
        self._writeln("// Parallel reduction in shared memory")
        self._writeln("for (int s = blockDim.x / 2; s > 0; s >>= 1) {")
        self.indent = 2
        self._writeln("if (tid < s) {")
        self.indent = 3
        
        reduction_op = self._get_reduction_op(expr)
        if reduction_op == "Max":
            self._writeln("sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);")
        elif reduction_op == "Min":
            self._writeln("sdata[tid] = fminf(sdata[tid], sdata[tid + s]);")
        else:  # Sum, Mean
            self._writeln("sdata[tid] += sdata[tid + s];")
        
        self.indent = 2
        self._writeln("}")
        self._writeln("__syncthreads();")
        self.indent = 1
        self._writeln("}")
        self._writeln("")
        
        # Write result
        self._writeln("// Write block result")
        self._writeln("if (tid == 0) {")
        self.indent = 2
        if reduction_op == "Mean":
            self._writeln("atomicAdd(y, sdata[0] / (float)n);")
        else:
            self._writeln("atomicAdd(y, sdata[0]);")
        self.indent = 1
        self._writeln("}")
        
        self.code.append("}")
        self.code.append("")
        
        self._emit_launcher(kernel_name, is_reduction=True)
        self._emit_cuda_graph_launcher(kernel_name, is_reduction=True)
        
        return "\n".join(self.code)
    
    def _emit_launcher(self, kernel_name: str, is_reduction: bool) -> None:
        """Generate a host-side launcher function."""
        self.code.append(f"void launch_{kernel_name}(")
        self.code.append("    const float* d_x,")
        self.code.append("    float* d_y,")
        self.code.append("    int n,")
        self.code.append("    cudaStream_t stream = 0")
        self.code.append(") {")
        
        self.indent = 1
        self._writeln("const int block_size = 256;")
        self._writeln("const int grid_size = (n + block_size - 1) / block_size;")
        
        if is_reduction:
            self._writeln(f"{kernel_name}<<<grid_size, block_size, block_size * sizeof(float), stream>>>(d_x, d_y, n);")
        else:
            self._writeln(f"{kernel_name}<<<grid_size, block_size, 0, stream>>>(d_x, d_y, n);")
        
        self.code.append("}")
        self.code.append("")

    def _emit_cuda_graph_launcher(self, kernel_name: str, is_reduction: bool) -> None:
        """Generate a CUDA Graph captured launcher for sub-microsecond single-token decoding loops."""
        self.code.append(f"// CUDA Graph container for sub-microsecond launch overhead elimination in single-token decoding")
        self.code.append(f"struct GraphContainer_{kernel_name} {{")
        self.code.append("    cudaGraph_t graph = NULL;")
        self.code.append("    cudaGraphExec_t instance = NULL;")
        self.code.append("    bool is_captured = false;")
        self.code.append("};")
        self.code.append(f"static GraphContainer_{kernel_name} g_graph_container_{kernel_name};")
        self.code.append("")
        self.code.append(f"void launch_{kernel_name}_graph(")
        self.code.append("    const float* d_x,")
        self.code.append("    float* d_y,")
        self.code.append("    int n,")
        self.code.append("    cudaStream_t stream = 0")
        self.code.append(") {")
        self.indent = 1
        self._writeln(f"if (!g_graph_container_{kernel_name}.is_captured) {{")
        self.indent = 2
        self._writeln("cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);")
        self._writeln(f"launch_{kernel_name}(d_x, d_y, n, stream);")
        self._writeln(f"cudaStreamEndCapture(stream, &g_graph_container_{kernel_name}.graph);")
        self._writeln(f"cudaGraphInstantiate(&g_graph_container_{kernel_name}.instance, g_graph_container_{kernel_name}.graph, NULL, NULL, 0);")
        self._writeln(f"g_graph_container_{kernel_name}.is_captured = true;")
        self.indent = 1
        self._writeln("}")
        self._writeln(f"cudaGraphLaunch(g_graph_container_{kernel_name}.instance, stream);")
        self.code.append("}")

    def _has_reduction(self, expr: Expr) -> bool:
        """Check if expression contains reduction operations."""
        if isinstance(expr, Box):
            op = self.sig.get(expr.op)
            return "reduction" in op.traits
        if isinstance(expr, Seq):
            return self._has_reduction(expr.first) or self._has_reduction(expr.second)
        if isinstance(expr, Par):
            return self._has_reduction(expr.left) or self._has_reduction(expr.right)
        return False
    
    def _get_pre_reduction_expr(self, expr: Expr) -> Expr | None:
        """Extract elementwise ops before reduction."""
        if isinstance(expr, Seq):
            if self._has_reduction(expr.second) and not self._has_reduction(expr.first):
                return expr.first
        return None
    
    def _get_reduction_op(self, expr: Expr) -> str:
        """Get the name of the reduction operation."""
        if isinstance(expr, Box):
            return expr.op
        if isinstance(expr, Seq):
            if self._has_reduction(expr.second):
                return self._get_reduction_op(expr.second)
            return self._get_reduction_op(expr.first)
        return "Sum"

    def _visit(self, expr: Expr, inputs: list[str]) -> list[str]:
        """Generate CUDA expression code."""
        if isinstance(expr, Box):
            op = self.sig.get(expr.op)
            input_var = inputs[0] if inputs else "x0"
            
            # Reduction operations
            if "reduction" in op.traits:
                return [input_var]  # Reduction handled separately
            
            # Elementwise operations
            if "elementwise" in op.traits:
                if op.name in ("Relu", "ReLU"):
                    return [f"fmaxf({input_var}, 0.0f)"]
                if op.name == "Sigmoid":
                    return [f"(1.0f / (1.0f + expf(-{input_var})))"]
                if op.name == "Exp":
                    return [f"expf({input_var})"]
                if op.name == "Log":
                    return [f"logf({input_var})"]
                if op.name == "Neg":
                    return [f"(-{input_var})"]
                if op.name == "Tanh":
                    return [f"tanhf({input_var})"]
                return [f"{op.name.lower()}f({input_var})"]
            
            return inputs
            
        if isinstance(expr, Seq):
            intermediate = self._visit(expr.first, inputs)
            return self._visit(expr.second, intermediate)
            
        if isinstance(expr, Par):
            from ..ir.normalize import infer_type
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
            
        raise NotImplementedError(f"Unsupported expr: {expr}")

    def _writeln(self, line: str) -> None:
        self.code.append("    " * self.indent + line)


class PyTorchCUDAGraphWrapper:
    """Wraps PyTorch / TENSORGRAPH functions in a CUDA Graph to eliminate single-token decoding launch overhead."""

    def __init__(self, model_fn: Any) -> None:
        self.model_fn = model_fn
        self.graph: Any = None
        self.static_input: Any = None
        self.static_output: Any = None

    def capture(self, sample_input: Any) -> PyTorchCUDAGraphWrapper:
        """Capture CUDA execution graph using sample_input tensor."""
        import torch

        if not torch.cuda.is_available() or not hasattr(sample_input, "clone"):
            return self

        self.static_input = sample_input.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.static_output = self.model_fn(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model_fn(self.static_input)

        return self


    def __call__(self, input_tensor: Any) -> Any:
        if self.graph is None:
            return self.model_fn(input_tensor)

        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output
