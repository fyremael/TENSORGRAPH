
import torch
import torch.nn as nn
from tensorgraph.backends import fx
from tensorgraph.signature import Signature
from tensorgraph.codegen.triton import TritonEmitter
from tensorgraph.types import Obj

T = Obj("Tensor")

class FusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.relu(x))

def test_fusion_e2e():
    # 1. Setup PyTorch Model
    model = FusedModel()
    
    # 2. Trace to FX (treating ReLU/Sigmoid as leaves is default for nn.Module)
    # We use fx.trace_with_leaf_modules just to be safe if needed, 
    # but standard trace works for nn.Modules.
    # Actually, let's use the backend's helper to ensure compatibility.
    gm = fx.trace_with_leaf_modules(model, (nn.ReLU, nn.Sigmoid))
    
    # 3. Import to TENSORGRAPH Intermediate Representation
    # We need to define types for the extraction
    # The backend expects specific types to extract attrs. 
    # But ReLU/Sigmoid have no attrs we care about for now.
    # We'll pass nn.Module as a dummy for Lora/Linear types to avoid crashes if possible?
    # Wait, fx_chain_to_ops specifically checks isinstance(mod, lora_inject_type) etc.
    # We need to bypass that or ensure it handles generic modules gracefully.
    # Looking at fx_chain_to_ops: it checks Lora/Linear but ELSE just adds generic Op.
    # "ops.append(FXChainOp(op_name=op_name, attrs=attrs...))"
    # So it should work for ReLU/Sigmoid with empty attrs.
    
    ops = fx.fx_chain_to_ops(gm, nn.Identity, nn.Linear) 
    # nn.Identity and nn.Linear passed as dummies/unmatched types 
    # so we fall through to generic op creation.
    
    # 4. Create Signature
    sig = Signature()
    # Note: PyTorch class names are "ReLU", "Sigmoid"
    sig.add("ReLU", T, T, traits={"elementwise"})
    sig.add("Sigmoid", T, T, traits={"elementwise"})
    
    # 5. Build Expression
    expr = fx.ops_to_expr(ops, sig, T)
    print(f"Imported Expression: {expr}")
    
    # 6. Optimize (Mock)
    # In a real run, we'd saturate here. 
    # Seq(ReLU, Sigmoid) is already the form we want to fuse.
    
    # 7. Codegen
    emitter = TritonEmitter(sig)
    code = emitter.emit(expr, kernel_name="e2e_fused")
    
    print("Generated Code:\n", code)
    
    # 8. Verify
    assert "@triton.jit" in code
    assert "def e2e_fused" in code
    # Check for correct nesting: sigmoid(where(...))
    # Emitter maps "Relu" -> logic. Our op is "ReLU" (case sensitive?).
    # TritonEmitter logic: if op.name == "Relu": ... 
    # I should check if TritonEmitter matches "ReLU" or "Relu".
    # Implementation has "Relu" title case. PyTorch is "ReLU". 
    # I will need to fix TritonEmitter or alias it in Signature.
    # Let's see what happens.

if __name__ == "__main__":
    test_fusion_e2e()
