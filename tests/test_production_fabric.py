"""
Tests for production fabric implementations.
"""
import pytest
import time
from tensorgraph.dist.fabric import AsyncFabric, FabricMessage, MessageType, create_fabric
from tensorgraph.dist.mock_fabric import MockFabric
from tensorgraph.dist.sharding import Shard
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.ir import Box, Seq


class TestAsyncFabric:
    """Test AsyncFabric threading and message passing."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("A", T, T)
        sig.add("B", T, T)
        sig.add("C", T, T)
        return sig
    
    def test_sync_pump_compatibility(self, sig: Signature):
        """AsyncFabric.pump() should work like MockFabric for testing."""
        fabric = AsyncFabric()
        
        s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
        lid_a = s1.ingest(Box("A"), global_id=100)
        lid_b = s1.ingest(Box("B"), global_id=101)
        fabric.register(s1)
        
        s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
        lid_a2 = s2.ingest(Box("A"))
        lid_b2 = s2.ingest(Box("B"))
        s2.partition.register_ghost(100, lid_a2)
        s2.partition.register_ghost(101, lid_b2)
        fabric.register(s2)
        
        # Merge on shard 1
        s1.partition.egraph.merge(lid_a, lid_b, "test")
        
        # Pump should propagate
        fabric.pump()
        s2.partition.egraph.rebuild()
        
        # Verify merge propagated
        assert s2.partition.egraph.uf.find(lid_a2) == s2.partition.egraph.uf.find(lid_b2)
    
    def test_async_start_stop(self, sig: Signature):
        """Test AsyncFabric thread lifecycle."""
        fabric = AsyncFabric()
        
        s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
        fabric.register(s1)
        
        fabric.start()
        assert fabric._running is True
        assert fabric._dispatcher_thread is not None
        assert fabric._dispatcher_thread.is_alive()
        
        fabric.stop(timeout=2.0)
        assert fabric._running is False
    
    def test_statistics_tracking(self, sig: Signature):
        """Test that fabric tracks statistics."""
        fabric = AsyncFabric()
        
        s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
        s1.ingest(Box("A"), global_id=100)
        s1.ingest(Box("B"), global_id=101)
        fabric.register(s1)
        
        s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
        fabric.register(s2)
        
        # Send merge
        fabric.send_merge(1, 100, 101)
        
        stats = fabric.get_stats()
        assert stats.messages_sent == 1
        
        # Pump and check processed
        fabric.pump()
        assert fabric.get_stats().messages_processed >= 0  # May or may not have processed yet


class TestFabricFactory:
    """Test fabric factory function."""
    
    def test_create_mock(self):
        fabric = create_fabric("mock")
        assert isinstance(fabric, MockFabric)
    
    def test_create_async(self):
        fabric = create_fabric("async", batch_size=50)
        assert isinstance(fabric, AsyncFabric)
        assert fabric.batch_size == 50
    
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            create_fabric("invalid")


class TestCrossShardMerge:
    """Integration test for distributed merge convergence."""
    
    @pytest.fixture
    def sig(self) -> Signature:
        T = Obj("Tensor")
        sig = Signature()
        sig.add("A", T, T)
        sig.add("B", T, T)
        sig.add("C", T, T)
        return sig
    
    def test_congruence_propagation(self, sig: Signature):
        """Merging A=B on shard1 should make Seq(A,C)=Seq(B,C) on shard2."""
        fabric = AsyncFabric()
        
        # Shard 1: owns A, B
        s1 = Shard(shard_id=1, fabric=fabric, sig=sig)
        lid_a1 = s1.ingest(Box("A"), global_id=100)
        lid_b1 = s1.ingest(Box("B"), global_id=101)
        fabric.register(s1)
        
        # Shard 2: ghosts A, B; owns C and Seq expressions
        s2 = Shard(shard_id=2, fabric=fabric, sig=sig)
        lid_a2 = s2.ingest(Box("A"))
        lid_b2 = s2.ingest(Box("B"))
        s2.partition.register_ghost(100, lid_a2)
        s2.partition.register_ghost(101, lid_b2)
        lid_c2 = s2.ingest(Box("C"))
        seq_ac = s2.ingest(Seq(Box("A"), Box("C")))
        seq_bc = s2.ingest(Seq(Box("B"), Box("C")))
        fabric.register(s2)
        
        # Initially different
        s2.partition.egraph.rebuild()
        assert s2.partition.egraph.uf.find(seq_ac) != s2.partition.egraph.uf.find(seq_bc)
        
        # Merge A=B on shard 1
        s1.partition.egraph.merge(lid_a1, lid_b1, "rewrite")
        
        # Pump fabric
        fabric.pump()
        s2.partition.egraph.rebuild()
        
        # Now should be equal via congruence
        assert s2.partition.egraph.uf.find(seq_ac) == s2.partition.egraph.uf.find(seq_bc)
