
from tensorgraph.dist.sharding import Shard
from tensorgraph.dist.mock_fabric import MockFabric
from tensorgraph.ir import Box, Id, Seq
from tensorgraph.types import Obj
from tensorgraph.signature import Signature as Sig
from tensorgraph.rewrite import Rewrite, PBox, PVar
from tensorgraph.egraph.saturation import saturate

T = Obj("T")

def test_dist_e2e_propagation():
    """
    Scenario:
    Shard 1 handles 'A'. It knows rule A -> B.
    Shard 2 handles larger graph containing 'A' and 'B'.
    Shard 1 proves A=B. Shard 2 learns this and proves Seq(A, C) = Seq(B, C).
    """
    fabric = MockFabric()
    sig = Sig()
    sig.add("A", T, T)
    sig.add("B", T, T)
    sig.add("C", T, T)
    
    # Rule: A -> B
    rule_a_b = Rewrite(
        name="a_to_b",
        lhs=PBox("A"),
        rhs=PBox("B")
    )
    
    # --- SETUP ---
    
    # Shard 1: "Local Optimizer" for A
    s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
    lid_a1 = s1.ingest(Box("A"), global_id=100)
    lid_b1 = s1.ingest(Box("B"), global_id=101) # B exists strictly speaking
    fabric.register(s1)
    
    # Shard 2: "Global Graph"
    s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
    # It has subgraphs using global IDs 100 (A) and 101 (B)
    lid_a2 = s2.ingest(Box("A")) 
    s2.partition.register_ghost(global_id=100, local_id=lid_a2)
    
    lid_b2 = s2.ingest(Box("B"))
    s2.partition.register_ghost(global_id=101, local_id=lid_b2)
    
    lid_c2 = s2.ingest(Box("C"))

    fabric.register(s2)
    
    # Expressions: Seq(A, C) and Seq(B, C)
    seq_ac = s2.ingest(Seq(Box("A"), Box("C")))
    seq_bc = s2.ingest(Seq(Box("B"), Box("C")))
    
    # Initially distinct
    assert s2.partition.egraph.uf.find(seq_ac) != s2.partition.egraph.uf.find(seq_bc)
    
    # --- EXECUTION ---
    
    print("Shard 1 saturating...")
    s1.saturate_local([rule_a_b])
    
    print(f"S1 A~B? {s1.partition.egraph.uf.find(lid_a1) == s1.partition.egraph.uf.find(lid_b1)}")
    print(f"Queue 2 before pump: {fabric.queues[2]}")
    
    # 2. Fabric propagation
    # Wait, simple rewrite instantiates B. It doesn't merge with explicit existing B unless we unify?
    # saturate instantiates RHS. If RHS node already exists, hashconsing finds it?
    # yes. But we added Box("B") explicitly.
    # So rewrite produces Box("B"). Add_enode finds existing Box("B").
    # Matches A. Merges A with B.
    # So s1 should trigger merge(lid_a1, lid_b1).
    # Since both have global IDs, send_merge(100, 101) should fire.
    
    # 2. Fabric propagation
    print("Fabric pumping...")
    fabric.pump()
    
    # 3. Shard 2 receives merge(100, 101)
    # It should merge lid_a2 and lid_b2.
    # Then rebuild() or just the merge itself should trigger congruence.
    # Seq(A, C) -> (A, C). Seq(B, C) -> (B, C). 
    # If A~B, then parents should be merged?
    # Congruence happens in rebuild().
    # Shard.receive_merge calls egraph.merge.
    # egraph.merge calls worklist stuff.
    # But we need EXPLICIT rebuild() call if we rely on congruence?
    # Or does saturated/merge handle it?
    # egraph.merge doesn't call rebuild().
    # We need to force a rebuild on Shard 2.
    
    print("Shard 2 rebuilding...")
    s2.partition.egraph.rebuild()
    
    # --- VERIFICATION ---
    root_ac = s2.partition.egraph.uf.find(seq_ac)
    root_bc = s2.partition.egraph.uf.find(seq_bc)
    
    if root_ac == root_bc:
        print("Distributed Congruence Verified!")
    else:
        print(f"Failed: {root_ac} != {root_bc}")
        # Debug
        print(f"A2~B2? {s2.partition.egraph.uf.find(lid_a2) == s2.partition.egraph.uf.find(lid_b2)}")

if __name__ == "__main__":
    test_dist_e2e_propagation()
