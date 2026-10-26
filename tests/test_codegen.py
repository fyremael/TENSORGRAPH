
from tensorgraph.codegen.triton import TritonEmitter
from tensorgraph.signature import Signature
from tensorgraph.ir import Box, Seq
from tensorgraph.types import Obj

T = Obj("T")

def test_triton_emit_elementwise():
    sig = Signature()
    # Define ops with traits
    sig.add("Relu", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    
    # Expr: Relu ; Sigmoid
    expr = Seq(Box("Relu"), Box("Sigmoid"))
    
    emitter = TritonEmitter(sig)
    code = emitter.emit(expr, kernel_name="test_kernel")
    
    print(code)
    
    # Basic checks
    assert "@triton.jit" in code
    assert "def test_kernel" in code
    assert "tl.where(x0 > 0, x0, 0)" in code
    assert "tl.sigmoid" in code
    assert "tl.store" in code

    assert "tl.store" in code

def test_triton_emit_par():
    sig = Signature()
    sig.add("Relu", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    
    # Par(Relu, Sigmoid)
    # Requires 2 inputs. Our current emit() hardcodes 1 input "x0".
    # This test will likely crash or produce weird code unless we update emit() 
    # to handle multi-input signature if inferred?
    # For now, let's manually inspect _visit behavior or just see if it runs.
    
    from tensorgraph.ir import Par
    expr = Par(Box("Relu"), Box("Sigmoid"))
    
    emitter = TritonEmitter(sig)
    
    # Manually invoke visit to test logic
    inputs = ["x0", "x1"]
    res = emitter._visit(expr, inputs)
    
    print("\nPar Result:", res)
    assert len(res) == 2
    assert "tl.where(x0" in res[0]
    assert "tl.sigmoid(x1)" in res[1]

if __name__ == "__main__":
    test_triton_emit_elementwise()
    test_triton_emit_par()
