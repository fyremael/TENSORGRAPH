from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "bench_six_baseline_elementwise.py"
    spec = importlib.util.spec_from_file_location("tensorgraph_six_baseline_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_six_baseline_matrix_is_complete_and_ordered() -> None:
    module = _load_benchmark_module()

    assert module.BASELINE_ORDER == (
        "pytorch_eager_source",
        "pytorch_eager_optimized",
        "torch_compile_source",
        "torch_compile_optimized",
        "tensorgraph_generated",
        "direct_triton_reference",
    )
    assert len(set(module.BASELINE_ORDER)) == 6


def test_direct_triton_reference_is_independent_and_normalized() -> None:
    module = _load_benchmark_module()
    source = module.DIRECT_TRITON_SOURCE

    assert source.count("tl.where") == 1
    assert source.count("value = -value") == 1
    assert "tensorgraph.pipeline" not in source
    assert "tl.sigmoid" not in source


def test_source_and_optimized_graphs_are_semantically_equal() -> None:
    module = _load_benchmark_module()
    x = torch.tensor([-3.0, -0.0, 0.0, 1.5, 8.0])

    source = module.SourceGraph()(x)
    optimized = module.OptimizedGraph()(x)

    torch.testing.assert_close(source, optimized, rtol=0.0, atol=0.0)


def test_tensorgraph_rewrite_matches_matrix_graph_contract() -> None:
    module = _load_benchmark_module()
    from tensorgraph.pipeline import compile_fx_elementwise

    artifact = compile_fx_elementwise(module.SourceGraph())

    assert artifact.source_pretty == "((ReLU ; ReLU) ; Neg)"
    assert artifact.optimized_pretty == "(ReLU ; Neg)"
    assert artifact.rewrite_summary["relu_idempotence"] >= 1
