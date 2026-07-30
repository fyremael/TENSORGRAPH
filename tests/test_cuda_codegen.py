"""
Tests for CUDA kernel code generation.
"""
import pytest
from tensorgraph.codegen.cuda import CUDAEmitter
from tensorgraph.codegen.triton import TritonEmitter
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.ir import Box, Seq


class TestCUDAElementwise:
    """Test CUDA elementwise kernel generation."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        sig.add("Sigmoid", T, T, traits={"elementwise"})
        sig.add("Exp", T, T, traits={"elementwise"})
        sig.add("Tanh", T, T, traits={"elementwise"})
        return sig
    
    def test_relu_kernel(self, sig: Signature):
        """Test ReLU generates fmaxf."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit(Box("ReLU"), kernel_name="relu_kernel")
        
        assert "__global__ void relu_kernel" in code
        assert "fmaxf(x0, 0.0f)" in code
        assert "y[idx] =" in code
    
    def test_sigmoid_kernel(self, sig: Signature):
        """Test Sigmoid generates correct formula."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit(Box("Sigmoid"), kernel_name="sigmoid_kernel")
        
        assert "1.0f / (1.0f + expf(-" in code
    
    def test_fused_seq(self, sig: Signature):
        """Test fused Seq generates single kernel."""
        emitter = CUDAEmitter(sig)
        expr = Seq(Box("ReLU"), Box("Sigmoid"))
        code = emitter.emit(expr, kernel_name="fused_relu_sigmoid")
        
        # Should have single kernel definition
        assert code.count("__global__ void") == 1
        assert "fmaxf" in code
        assert "expf" in code
    
    def test_launcher_generated(self, sig: Signature):
        """Test host launcher function is generated."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit(Box("ReLU"))
        
        assert "void launch_" in code
        assert "cudaStream_t stream" in code
        assert "<<<grid_size, block_size" in code


class TestCUDAReduction:
    """Test CUDA reduction kernel generation."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        R = Obj("Scalar")
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        sig.add("Sum", T, R, traits={"reduction"})
        sig.add("Mean", T, R, traits={"reduction"})
        sig.add("Max", T, R, traits={"reduction"})
        return sig
    
    def test_sum_reduction(self, sig: Signature):
        """Test Sum generates shared memory reduction."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit(Box("Sum"), kernel_name="sum_kernel")
        
        assert "extern __shared__ float sdata[]" in code
        assert "__syncthreads()" in code
        assert "sdata[tid] +=" in code
        assert "atomicAdd" in code
    
    def test_max_reduction(self, sig: Signature):
        """Test Max uses fmaxf."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit(Box("Max"), kernel_name="max_kernel")
        
        assert "fmaxf(sdata[tid], sdata[tid + s])" in code
    
    def test_fused_relu_sum(self, sig: Signature):
        """Test fused ReLU -> Sum applies elementwise before reduction."""
        emitter = CUDAEmitter(sig)
        expr = Seq(Box("ReLU"), Box("Sum"))
        code = emitter.emit(expr, kernel_name="relu_sum")
        
        assert "__shared__ float sdata[]" in code
        assert "fmaxf" in code  # ReLU applied
        assert "sdata[tid] +=" in code  # Sum reduction


class TestCUDATritonParity:
    """Verify CUDA and Triton emit same operations."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        sig.add("Sigmoid", T, T, traits={"elementwise"})
        return sig
    
    def test_elementwise_parity(self, sig: Signature):
        """Both emitters should handle same expression."""
        expr = Seq(Box("ReLU"), Box("Sigmoid"))
        
        cuda = CUDAEmitter(sig)
        triton = TritonEmitter(sig)
        
        cuda_code = cuda.emit(expr)
        triton_code = triton.emit(expr)
        
        # Both should be non-empty
        assert len(cuda_code) > 100
        assert len(triton_code) > 100
        
        # Both should reference the operations
        assert "sigmoid" in triton_code.lower()
        assert "sigmoid" in cuda_code.lower() or "expf" in cuda_code


class TestCUDAGraphCapture:
    """Test CUDA Graph capture code generation and PyTorch wrapper."""

    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        return sig

    def test_cuda_graph_launcher_generated(self, sig: Signature):
        """CUDA Graph capture routines are emitted."""
        emitter = CUDAEmitter(sig)
        code = emitter.emit_with_cuda_graph(Box("ReLU"), kernel_name="relu_decoding")

        assert "cudaStreamBeginCapture" in code
        assert "cudaStreamEndCapture" in code
        assert "cudaGraphInstantiate" in code
        assert "cudaGraphLaunch" in code
        assert "GraphContainer_relu_decoding" in code

    def test_pytorch_graph_wrapper_fallback(self):
        """Test PyTorchCUDAGraphWrapper gracefully falls back when CUDA is unavailable."""
        from tensorgraph.codegen.cuda import PyTorchCUDAGraphWrapper

        def dummy_fn(x):
            return x * 2.0

        wrapper = PyTorchCUDAGraphWrapper(dummy_fn)
        # Capture should return self even on CPU
        res_wrapper = wrapper.capture("dummy_input")
        assert res_wrapper is wrapper

        # Executing wrapper returns underlying function output on fallback
        assert wrapper(5.0) == 10.0

