
from tensorgraph.ir import Box, Seq, Par, Dup, Id
from tensorgraph.types import Obj
from tensorgraph.ir.primitives import Swap
from tensorgraph.backend.triton_emitter import TritonEmitter

def demo():
    print("="*60)
    print("TENSORGRAPH v0.5.0: Triton Kernel Generation Demo")
    print("="*60)
    
    emitter = TritonEmitter()
    
    # ---------------------------------------------------------
    # Scenario 1: Parallel Independent Ops (The Mandate)
    # Graph: Par(Add, Mul)
    # Inputs: [a, b, c, d] -> [a+b, c*d]
    # ---------------------------------------------------------
    print("\nScenario 1: Par(Add, Mul) [Independent Parallelism]")
    expr1 = Par(Box("Add"), Box("Mul"))
    
    print("IR Expression:", expr1)
    code1 = emitter.emit_kernel(expr1, kernel_name="parallel_add_mul")
    print("\nGenerated Triton Kernel:\n")
    print(code1)
    
    # ---------------------------------------------------------
    # Scenario 2: Complex Wiring (Data Reuse & Shuffle)
    # Graph: (x+y) * (x-y)
    # Inputs: [x, y]
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("Scenario 2: (x+y)*(x-y) [Data Reuse & Swap]")
    
    # 1. Duplicate Inputs: [x, y] -> [x, x, y, y]
    stage1 = Par(Dup(Obj("x")), Dup(Obj("y")))
    
    # 2. Shuffle: [x, x, y, y] -> [x, y, x, y]
    #    Middle two (x, y) need swap.
    #    Par(Id, Par(Swap, Id))
    #    (Id on x_a) ; (Swap on x_b, y_a) ; (Id on y_b)
    #    Wait, Par is binary. Par(Id, Par(Swap, Id)) works?
    #    Inputs: 1 + (2 + 1) = 4. Checks out.
    stage2 = Par(Id(Obj("x")), Par(Swap(Obj("x"), Obj("y")), Id(Obj("y"))))
    
    # 3. Compute: [x, y, x, y] -> [x+y, x-y]
    stage3 = Par(Box("Add"), Box("Sub"))
    
    # 4. Combine: [add_res, sub_res] -> [mul_res]
    stage4 = Box("Mul")
    
    # Compose all
    expr2 = Seq(stage1, Seq(stage2, Seq(stage3, stage4)))
    
    print("IR Expression:", expr2)
    code2 = emitter.emit_kernel(expr2, kernel_name="diff_of_squares")
    print("\nGenerated Triton Kernel:\n")
    print(code2)

if __name__ == "__main__":
    demo()
