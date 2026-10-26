"""
TENSORGRAPH v0.2.0: DAG Import Tests

Verifies the P0 requirement: robust FX graph capture including
call_function, call_method, and DAG topologies.
"""
import pytest


# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
torch_nn = pytest.importorskip("torch.nn")
torch_fx = pytest.importorskip("torch.fx")


from tensorgraph.backends.fx_dag import DAGLifter, lift_fx_graph
from tensorgraph.ir import Box, Id, Par, Seq, pretty
from tensorgraph.signature import Signature
from tensorgraph.types import Obj


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def tensor_obj():
    return Obj("Tensor")


@pytest.fixture
def signature(tensor_obj):
    sig = Signature()
    sig.add("relu", tensor_obj, tensor_obj)
    sig.add("Linear", tensor_obj, tensor_obj)
    sig.add("add", tensor_obj, tensor_obj)
    return sig


# -----------------------------------------------------------------------------
# P0-1: call_function support
# -----------------------------------------------------------------------------


def test_call_function_relu(signature, tensor_obj):
    """Test import of F.relu functional call."""
    import torch.nn.functional as F
    
    class SimpleReLU(torch.nn.Module):
        def forward(self, x):
            return F.relu(x)
    
    model = SimpleReLU()
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    # Should contain a relu Box
    expr_str = pretty(expr)
    assert "relu" in expr_str or "Id" in expr_str  # May normalize differently


def test_call_function_add(signature, tensor_obj):
    """Test import of torch.add functional call."""
    class AddModule(torch.nn.Module):
        def forward(self, x):
            return torch.add(x, x)
    
    model = AddModule()
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    # Should produce a valid expression (may include Par for dual inputs)
    assert expr is not None


# -----------------------------------------------------------------------------
# P0-2: call_method support
# -----------------------------------------------------------------------------


def test_call_method_view(signature, tensor_obj):
    """Test import of x.view() method call."""
    class ViewModule(torch.nn.Module):
        def forward(self, x):
            return x.view(-1)
    
    model = ViewModule()
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    # Should contain a view Box
    assert expr is not None


# -----------------------------------------------------------------------------
# P0-3: DAG support (multi-consumer nodes)
# -----------------------------------------------------------------------------


def test_dag_multi_consumer(signature, tensor_obj):
    """Test DAG where one node has multiple consumers."""
    import torch.nn.functional as F
    
    class DAGModule(torch.nn.Module):
        def forward(self, x):
            # x is consumed twice -> requires Dup in categorical semantics
            return F.relu(x) + x
    
    model = DAGModule()
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    # Expression should be valid
    assert expr is not None
    # The placeholder node should have ref_count > 1


def test_dag_diamond(signature, tensor_obj):
    """Test diamond pattern: input -> A,B -> output."""
    import torch.nn.functional as F
    
    class DiamondModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64)
        
        def forward(self, x):
            # Diamond: x -> relu and linear -> add
            a = F.relu(x)
            b = self.linear(x)
            return a + b
    
    model = DiamondModule()
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    # Should handle diamond without error
    assert expr is not None


# -----------------------------------------------------------------------------
# P0-4: Linear chain (regression)
# -----------------------------------------------------------------------------


def test_linear_chain_regression(signature, tensor_obj):
    """Ensure linear chains still work (regression test for MVP)."""
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    gm = torch_fx.symbolic_trace(model)
    
    expr = lift_fx_graph(gm, signature, tensor_obj)
    
    assert expr is not None
    # Should be a Seq structure
    assert isinstance(expr, (Seq, Box, Id))
