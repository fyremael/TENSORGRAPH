
from tensorgraph.dist.sharding import Shard
from tensorgraph.ir import Box, Id
from tensorgraph.types import Obj

T = Obj("T")

from tensorgraph.signature import Signature as Sig

def test_shard_ingest():
    sig = Sig()
    sig.add("A", T, T)
    s1 = Shard(shard_id=1, sig=sig)
    expr = Box("A")
    lid = s1.ingest(expr, global_id=100)
    
    assert lid in s1.partition.owned
    assert s1.partition.owned[lid] == 100

def test_ghost_merge():
    # Simulate Shard 2 tracking a ghost from Shard 1
    sig = Sig()
    sig.add("A", T, T)
    sig.add("B", T, T)
    s2 = Shard(shard_id=2, sig=sig)
    
    # Register global ID 100 as local ID 5
    s2.partition.register_ghost(global_id=100, local_id=5)
    
    # Register global ID 200 as local ID 6
    s2.partition.register_ghost(global_id=200, local_id=6)
    
    assert s2.partition.ghosts[100] == 5
    assert s2.partition.ghosts[200] == 6
    
    # Shard 1 merges 100 and 200. Notify Shard 2.
    # Note: local_id 5 and 6 should be merged in s2's EGraph if they exist.
    # But here we just registered IDs. We need to create nodes if we want EGraph merge.
    # Let's add nodes first.
    lid_a = s2.ingest(Box("A")) # 0
    lid_b = s2.ingest(Box("B")) # 1
    
    s2.partition.ghosts[100] = lid_a
    s2.partition.ghosts[200] = lid_b
    
    assert s2.partition.egraph.uf.find(lid_a) != s2.partition.egraph.uf.find(lid_b)
    
    s2.receive_merge(global_a=100, global_b=200)
    
    # Now they should be merged locally
    assert s2.partition.egraph.uf.find(lid_a) == s2.partition.egraph.uf.find(lid_b)

if __name__ == "__main__":
    test_shard_ingest()
    test_ghost_merge()
    print("Sharding tests passed")
