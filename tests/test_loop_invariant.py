
from tensorgraph.egraph.egraph import EGraph
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.ir import Box, Iter, Par, Id
from tensorgraph.egraph.saturation import saturate
from tensorgraph.library.control_flow import ALL_RULES

T = Obj("T")

def test_loop_invariant_hoisting():
    sig = Signature()
    sig.add("f", T, T) # Variant
    sig.add("g", T, T) # Invariant if g=Id? No, invariant means it doesn't change per iter.
    
    # Loop Invariant Code Motion (LICM)
    # Target: Iter(Par(f, Id), n) -> Par(Iter(f), Iter(Id)) -> Par(Iter(f), Id)
    
    eg = EGraph(sig)
    f = Box("f")
    i = Id(T)
    
    # Iter(f x I, 10)
    # f changes state, I is invariant
    expr = Iter(Par(f, i), 10)
    root = eg.add_expr(expr)
    
    saturate(eg, ALL_RULES, iters=10)
    
    # Expected: Par(Iter(f, 10), Id(T)) because Iter(Id, n) -> Id
    # We need a rule Iter(Id, n) -> Id. We implemented unroll_zero (n=0) but not general identity.
    # Iter(Id, n) = Id ; Iter(Id, n-1) ... = Id
    
    # Let's add the target expression and see if it unifies
    # Note: Iter(Id, 10) should reduce to Id(T) if we add a rule or if unrolling happens.
    # Unrolling 10 times is expensive. Better to have Iter(Id, n) -> Id rule.
    
    expected = Par(Iter(f, 10), i)
    expected_id = eg.add_expr(expected)
    
    if eg.uf.find(root) == eg.uf.find(expected_id):
        print("LICM confirmed with unrolling/rules.")
    else:
        print("LICM failed. Missing Iter(Id) -> Id rule?")
        # Let's check intermediate: Par(Iter(f), Iter(Id))
        inter = Par(Iter(f, 10), Iter(i, 10))
        inter_id = eg.add_expr(inter)
        print(f"Split confirmed? {eg.uf.find(root) == eg.uf.find(inter_id)}")

if __name__ == "__main__":
    test_loop_invariant_hoisting()
