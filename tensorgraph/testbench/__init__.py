"""
TENSORGRAPH Validation Testbench Suite.
========================================
Comprehensive benchmark, verification, and evaluation suite for diagrammatic rewriting.
"""

from .workloads import Workload, WorkloadSuite, get_all_workloads
from .evaluator import Evaluator, EvaluationResult
from .runner import TestbenchRunner, BenchmarkReport
from .fx_roundtrip import FXRoundtripOptimizer, FXOptimizationReport

__all__ = [
    "Workload",
    "WorkloadSuite",
    "get_all_workloads",
    "Evaluator",
    "EvaluationResult",
    "TestbenchRunner",
    "BenchmarkReport",
    "FXRoundtripOptimizer",
    "FXOptimizationReport",
]
