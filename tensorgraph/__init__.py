"""TENSORGRAPH: a typed diagram rewriting compiler with 2D semantics."""

from .adjunction import Adjunction
from .egraph import EGraph, ENode, Trace, TraceEntry, UnionFind
from .egraph.extract import Extractor, make_host_aware_cost_function
from .egraph.saturation import saturate
from .hardware import HardwareCapabilities, get_hardware_capabilities
from .engine import HybridEngine
from .ir import Box, Expr, Id, Par, Seq, infer_type, normalize, pretty
from .rewrite import (
    Pattern,
    PBox,
    PId,
    PPar,
    PSeq,
    PVar,
    Rewrite,
    ematch,
)
from .signature import OpDef, Signature
from .types import Obj, ObjLike, ObjVar, Sort

__all__ = [
    "Obj",
    "ObjVar",
    "ObjLike",
    "Sort",
    "Signature",
    "OpDef",
    "Expr",
    "Id",
    "Box",
    "Seq",
    "Par",
    "pretty",
    "normalize",
    "infer_type",
    "Pattern",
    "PVar",
    "PId",
    "PBox",
    "PSeq",
    "PPar",
    "Rewrite",
    "ematch",
    "EGraph",
    "ENode",
    "UnionFind",
    "Trace",
    "TraceEntry",
    "saturate",
    "Extractor",
    "make_host_aware_cost_function",
    "Adjunction",
    "HardwareCapabilities",
    "get_hardware_capabilities",
    "HybridEngine",
]

