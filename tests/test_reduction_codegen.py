"""
Tests for reduction kernel code generation.
"""
import pytest
from tensorgraph.codegen.triton import TritonEmitter, emit_softmax
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.ir import Box, Seq


class TestReductionCodegen:
    """Test reduction kernel generation."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        R = Obj("Scalar")  # Reduction output
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        sig.add("Sum", T, R, traits={"reduction"})
        sig.add("Mean", T, R, traits={"reduction"})
        sig.add("Max", T, R, traits={"reduction"})
        sig.add("Min", T, R, traits={"reduction"})
        sig.add("Exp", T, T, traits={"elementwise"})
        return sig
    
    def test_sum_reduction(self, sig: Signature):
        """Test Sum generates tl.sum."""
        emitter = TritonEmitter(sig)
        expr = Box("Sum")
        code = emitter.emit(expr, kernel_name="sum_kernel")
        
        assert "tl.sum(x0, axis=0)" in code
        assert "# Reduction kernel" in code
        # Reduction stores scalar, not masked array
        assert "tl.store(y_ptr," in code
        assert "mask=mask)" not in code.split("tl.store")[-1]
    
    def test_mean_reduction(self, sig: Signature):
        """Test Mean generates sum/n_elements."""
        emitter = TritonEmitter(sig)
        expr = Box("Mean")
        code = emitter.emit(expr, kernel_name="mean_kernel")
        
        assert "tl.sum(" in code
        assert "/ n_elements" in code
    
    def test_max_reduction(self, sig: Signature):
        """Test Max generates tl.max."""
        emitter = TritonEmitter(sig)
        expr = Box("Max")
        code = emitter.emit(expr, kernel_name="max_kernel")
        
        assert "tl.max(x0, axis=0)" in code
    
    def test_relu_then_sum(self, sig: Signature):
        """Test fused ReLU -> Sum generates reduction kernel."""
        emitter = TritonEmitter(sig)
        expr = Seq(Box("ReLU"), Box("Sum"))
        code = emitter.emit(expr, kernel_name="relu_sum")
        
        # Should detect reduction and use reduction template
        assert "# Reduction kernel" in code
        # Should have ReLU applied before sum
        assert "tl.where(" in code
        assert "tl.sum(" in code
    
    def test_softmax_standalone(self, sig: Signature):
        """Test standalone softmax kernel generation."""
        code = emit_softmax(sig)
        
        assert "fused_softmax" in code
        assert "x_max = tl.max" in code
        assert "x_shifted = x - x_max" in code
        assert "tl.exp(x_shifted)" in code
        assert "tl.sum(x_exp" in code
        assert "x_exp / x_sum" in code


class TestElementwiseStillWorks:
    """Ensure elementwise codegen is not broken."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("ReLU", T, T, traits={"elementwise"})
        sig.add("Sigmoid", T, T, traits={"elementwise"})
        sig.add("Exp", T, T, traits={"elementwise"})
        sig.add("Log", T, T, traits={"elementwise"})
        return sig
    
    def test_elementwise_uses_pid(self, sig: Signature):
        """Elementwise kernels should use program_id for parallelism."""
        emitter = TritonEmitter(sig)
        expr = Seq(Box("ReLU"), Box("Sigmoid"))
        code = emitter.emit(expr)
        
        assert "pid = tl.program_id" in code
        assert "block_start = pid * BLOCK_SIZE" in code
    
    def test_new_elementwise_ops(self, sig: Signature):
        """Test Exp and Log generate correct code."""
        emitter = TritonEmitter(sig)
        
        expr = Seq(Box("Exp"), Box("Log"))
        code = emitter.emit(expr)
        
        assert "tl.exp(" in code
        assert "tl.log(" in code
