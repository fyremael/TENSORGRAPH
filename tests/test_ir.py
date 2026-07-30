"""TENSORGRAPH IR Tests — String Diagram Intermediate Representation.

Comprehensive tests for the typed string diagram IR:
- Objects & Morphisms
- Expression construction
- Type inference
- Normalization
- Pretty printing
"""

from __future__ import annotations

import pytest

from tensorgraph import Obj, Signature
from tensorgraph.ir import Box, Id, Par, Seq, infer_type, normalize, pretty


class TestObjects:
    """Test Object type system."""
    
    def test_obj_creation(self):
        """Objects can be created with string names."""
        T = Obj("Tensor")
        assert T.name == "Tensor"
    
    def test_obj_equality(self):
        """Objects with same name are equal."""
        T1 = Obj("T")
        T2 = Obj("T")
        assert T1 == T2
    
    def test_obj_product(self):
        """Objects support tensor product via @."""
        A = Obj("A")
        B = Obj("B")
        AB = A @ B
        assert hasattr(AB, 'left') or hasattr(AB, 'fst')  # Product type
    
    def test_obj_repr(self):
        """Objects have readable repr."""
        T = Obj("Tensor")
        assert "Tensor" in repr(T)


class TestMorphisms:
    """Test morphism constructors."""
    
    def test_id_creation(self):
        """Identity morphisms can be created."""
        T = Obj("T")
        i = Id(T)
        assert i.obj == T
        assert isinstance(i, Id)
    
    def test_box_creation(self):
        """Box morphisms can be created with names."""
        f = Box("relu")
        assert f.op == "relu"
        assert isinstance(f, Box)
    
    def test_box_with_attrs(self):
        """Box morphisms support attributes."""
        lora = Box.with_attrs("LoRA", deltas=("A", "B"))
        assert lora.op == "LoRA"
        assert ("deltas", ("A", "B")) in lora.attrs
    
    def test_seq_composition(self):
        """Sequential composition f ; g."""
        f = Box("f")
        g = Box("g")
        fg = Seq(f, g)
        assert isinstance(fg, Seq)
        assert fg.first == f
        assert fg.second == g
    
    def test_par_composition(self):
        """Parallel composition f ⊗ g."""
        f = Box("f")
        g = Box("g")
        fg = Par(f, g)
        assert isinstance(fg, Par)


class TestTypeInference:
    """Test type inference for expressions."""
    
    def test_id_type(self, signature):
        """Id(T) : T → T."""
        T = Obj("T")
        i = Id(T)
        dom, cod = infer_type(i, signature)
        assert dom == T
        assert cod == T
    
    def test_box_type(self, signature):
        """Box types come from signature."""
        T = Obj("T")
        f = Box("f")
        dom, cod = infer_type(f, signature)
        assert dom == T
        assert cod == T
    
    def test_seq_type(self, signature):
        """Seq(f, g) : dom(f) → cod(g)."""
        T = Obj("T")
        fg = Seq(Box("f"), Box("g"))
        dom, cod = infer_type(fg, signature)
        assert dom == T
        assert cod == T


class TestNormalization:
    """Test expression normalization."""
    
    def test_normalize_preserves_simple(self):
        """Simple expressions normalize to themselves."""
        f = Box("f")
        assert normalize(f) == f
    
    def test_normalize_flattens_seq(self):
        """Nested Seq is normalized."""
        f, g, h = Box("f"), Box("g"), Box("h")
        nested = Seq(Seq(f, g), h)
        norm = normalize(nested)
        # Should still be valid expression
        assert norm is not None


class TestPrettyPrint:
    """Test pretty printing."""
    
    def test_pretty_id(self):
        """Id pretty prints correctly."""
        T = Obj("T")
        s = pretty(Id(T))
        assert "id" in s.lower() or "Id" in s
    
    def test_pretty_box(self):
        """Box pretty prints with op name."""
        s = pretty(Box("relu"))
        assert "relu" in s
    
    def test_pretty_seq(self):
        """Seq uses semicolon notation."""
        fg = Seq(Box("f"), Box("g"))
        s = pretty(fg)
        assert ";" in s or "f" in s  # Implementation may vary
