"""TENSORGRAPH E-Graph Tests — Equality Graph Engine.

Comprehensive tests for the e-graph implementation:
- E-Graph construction
- Expression addition
- Union-Find operations
- E-Class management
- Saturation
- Extraction
"""

from __future__ import annotations

import pytest

from tensorgraph import Obj, Signature
from tensorgraph.egraph import EGraph
from tensorgraph.egraph.extract import Extractor
from tensorgraph.egraph.saturation import saturate
from tensorgraph.ir import Box, Id, Seq


class TestEGraphConstruction:
    """Test e-graph creation and basic operations."""
    
    def test_egraph_creation(self, signature):
        """E-graph can be created with a signature."""
        eg = EGraph(signature)
        assert eg is not None
        assert eg.sig == signature
    
    def test_add_simple_expr(self, signature):
        """Simple expressions can be added to e-graph."""
        eg = EGraph(signature)
        f = Box("f")
        cid = eg.add_expr(f)
        assert isinstance(cid, int)
        assert cid >= 0
    
    def test_add_seq_expr(self, signature):
        """Sequential expressions can be added."""
        eg = EGraph(signature)
        fg = Seq(Box("f"), Box("g"))
        cid = eg.add_expr(fg)
        assert isinstance(cid, int)
    
    def test_add_duplicate_returns_same(self, signature):
        """Adding same expression returns same e-class."""
        eg = EGraph(signature)
        f1 = Box("f")
        f2 = Box("f")
        cid1 = eg.add_expr(f1)
        cid2 = eg.add_expr(f2)
        assert eg.uf.find(cid1) == eg.uf.find(cid2)


class TestUnionFind:
    """Test union-find for e-class management."""
    
    def test_find_identity(self, signature):
        """Find of new class returns itself."""
        eg = EGraph(signature)
        cid = eg.add_expr(Box("f"))
        assert eg.uf.find(cid) == cid
    
    def test_union_merges_classes(self, signature):
        """Union merges two e-classes."""
        eg = EGraph(signature)
        f = eg.add_expr(Box("f"))
        g = eg.add_expr(Box("g"))
        
        # Initially different
        assert eg.uf.find(f) != eg.uf.find(g)
        
        # After union, same
        eg.uf.union(f, g)
        assert eg.uf.find(f) == eg.uf.find(g)


class TestSaturation:
    """Test equality saturation."""
    
    def test_saturate_empty_rules(self, signature):
        """Saturation with no rules completes."""
        eg = EGraph(signature)
        cid = eg.add_expr(Seq(Box("f"), Box("g")))
        eg.root = cid
        
        # Should complete without error
        saturate(eg, [], iters=5)
    
    def test_saturate_with_trace(self, signature):
        """Saturation can record trace."""
        from tensorgraph.egraph.trace import Trace
        
        eg = EGraph(signature)
        eg.root = eg.add_expr(Box("f"))
        
        trace = Trace()
        saturate(eg, [], iters=1, trace=trace)
        
        # Trace should exist (may be empty with no rules)
        assert trace is not None


class TestExtraction:
    """Test cost-based extraction."""
    
    def test_extract_simple(self, signature):
        """Simple expression can be extracted."""
        eg = EGraph(signature)
        f = Box("f")
        cid = eg.add_expr(f)
        eg.root = cid
        
        ex = Extractor(eg)
        ex.solve(cid)
        result = ex.extract(cid)
        
        assert result is not None
        assert isinstance(result, Box)
    
    def test_extract_seq(self, signature):
        """Sequential expression can be extracted."""
        eg = EGraph(signature)
        fg = Seq(Box("f"), Box("g"))
        cid = eg.add_expr(fg)
        eg.root = cid
        
        ex = Extractor(eg)
        ex.solve(cid)
        result = ex.extract(cid)
        
        assert result is not None


class TestTrace:
    """Test trace infrastructure (FR-7)."""
    
    def test_trace_creation(self):
        """Trace can be created."""
        from tensorgraph.egraph.trace import Trace
        
        trace = Trace()
        assert len(trace) == 0
    
    def test_trace_record(self):
        """Trace can record entries."""
        from tensorgraph.egraph.trace import Trace
        
        trace = Trace()
        trace.record(
            rule_name="TestRule",
            root_eclass=0,
            rhs_eclass=1,
            merged_from=0,
            merged_to=1,
            expr_env={"x": 1},
            obj_env={}
        )
        
        assert len(trace) == 1
        assert trace.entries[0].rule_name == "TestRule"
    
    def test_trace_summary(self):
        """Trace summary works."""
        from tensorgraph.egraph.trace import Trace
        
        trace = Trace()
        for i in range(3):
            trace.record("RuleA", i, i+1, i, i+1)
        
        summary = trace.summary()
        assert "RuleA" in summary
        assert summary["RuleA"] == 3
    
    def test_trace_dump(self):
        """Trace dump produces string."""
        from tensorgraph.egraph.trace import Trace
        
        trace = Trace()
        trace.record("RuleX", 0, 1, 0, 1, {"v": 1}, {})
        
        dump = trace.dump()
        assert "RuleX" in dump
        assert "entries" in dump
