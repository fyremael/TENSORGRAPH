
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCoreModule(nn.Module):
    def __init__(self, head_dim=None):
        super().__init__()
        self.scale = (head_dim if head_dim else 64) ** -0.5
        
    def forward(self, qkv_tuple):
        print(f"DEBUG: Input type {type(qkv_tuple)}")
        if isinstance(qkv_tuple, tuple):
             print(f"DEBUG: Tuple len {len(qkv_tuple)}")
        
        q, k, v = qkv_tuple
        return q

def test():
    core = AttentionCoreModule()
    q = torch.randn(1, 10, 64)
    tup = (q, q, q)
    
    print("Testing direct call:")
    res = core(tup)
    print("Direct call OK")
    
    print("Testing nn.Sequential:")
    # Sequential can't handle tuple input between layers usually?
    # Wait. nn.Sequential passes output of prev to next.
    # If prev returns tuple, next receives tuple.
    
    class Unzip(nn.Module):
        def forward(self, x):
            print("Unzip returning tuple")
            return (x, x, x)
            
    seq = nn.Sequential(
        Unzip(),
        core
    )
    
    print("Running seq(x)...")
    try:
        seq(q)
        print("Seq OK")
    except Exception as e:
        print(f"Seq FAILED: {e}")

if __name__ == "__main__":
    test()
