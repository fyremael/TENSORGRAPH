
from tensorgraph.dist.sharding import Shard
from tensorgraph.dist.mock_fabric import MockFabric
from tensorgraph.ir import Box, Id
from tensorgraph.types import Obj
from tensorgraph.signature import Signature as Sig

T = Obj("T")

def test_dist_rewrite_convergence():
    fabric = MockFabric()
    sig = Sig()
    sig.add("A", T, T)
    sig.add("B", T, T)
    
    # Shard 1: Owns global A and B
    s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
    lid_a1 = s1.ingest(Box("A"), global_id=100)
    lid_b1 = s1.ingest(Box("B"), global_id=200)
    fabric.register(s1)
    
    # Shard 2: Has ghosts of A and B
    s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
    lid_a2 = s2.ingest(Box("A")) # Local copy
    lid_b2 = s2.ingest(Box("B")) # Local copy
    s2.partition.register_ghost(global_id=100, local_id=lid_a2)
    s2.partition.register_ghost(global_id=200, local_id=lid_b2)
    fabric.register(s2)
    
    # Step 1: Merge A and B on Shard 1
    s1.partition.egraph.merge(lid_a1, lid_b1, "manual_merge")
    
    # Check Shard 1 is merged
    assert s1.partition.egraph.uf.find(lid_a1) == s1.partition.egraph.uf.find(lid_b1)
    
    # Check Shard 2 is NOT merged yet
    assert s2.partition.egraph.uf.find(lid_a2) != s2.partition.egraph.uf.find(lid_b2)
    
    # Step 2: Pump fabric
    fabric.pump()
    
    # Check Shard 2 is NOW merged
    assert s2.partition.egraph.uf.find(lid_a2) == s2.partition.egraph.uf.find(lid_b2)
    print("Distributed merge converged!")

if __name__ == "__main__":
    test_dist_rewrite_convergence()
