from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _validator() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "validate_wp03_evidence.py"
    spec = importlib.util.spec_from_file_location("validate_wp03_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timing() -> dict[str, object]:
    return {
        "samples_ms": [0.01, 0.02, 0.03],
        "median_ms": 0.02,
        "minimum_ms": 0.01,
        "maximum_ms": 0.03,
    }


def _document() -> dict[str, object]:
    source = "__global__ void kernel() {}\n"
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    baselines = {
        name: {"status": "passed", "timing": _timing()}
        for name in (
            "pytorch_eager",
            "torch_compile",
            "tensorgraph_triton",
            "direct_triton",
            "tensorgraph_native_cuda",
        )
    }
    cell = {
        "key": "relu__float32__moderate__ordinary",
        "graph": "relu",
        "operations": ["ReLU"],
        "dtype": "float32",
        "regime": "moderate",
        "launch_mode": "ordinary",
        "disposition": "passed",
        "generated_source": source,
        "generated_source_sha256": source_sha,
        "abi": {"layout": "contiguous"},
        "compiler": {"status": "compiled"},
        "numerical": {"passed": True},
        "timings": {"ordinary": _timing()},
        "baselines": baselines,
    }
    return {
        "schema": "tensorgraph.evidence.native-cuda-inference.v1",
        "package": "TG-GPU-WP03",
        "repository": "fyremael/TENSORGRAPH",
        "timestamp_utc": "2026-07-31T12:00:00+00:00",
        "commit_sha": "a" * 40,
        "dirty_worktree": False,
        "command": ["python", "benchmarks/bench_native_cuda_inference.py"],
        "stack_id": "test-stack",
        "seed": 1,
        "size": 8,
        "warmup": 1,
        "repetitions": 3,
        "request": {
            "graphs": ["relu"],
            "dtypes": ["float32"],
            "regimes": ["moderate"],
            "launch_modes": ["ordinary"],
        },
        "torch_compile_isolation": {
            "mode": "torch._dynamo.reset_per_graph_family",
            "graph_families": ["relu"],
        },
        "versions": {},
        "hardware": {
            "gpu_index": 0,
            "gpu_name": "test",
            "gpu_compute_capability": [8, 0],
            "gpu_total_memory_bytes": 1,
        },
        "cells": [cell],
        "matrix_summary": {
            "requested_cells": 1,
            "recorded_cells": 1,
            "passed_cells": 1,
            "unsupported_cells": 0,
            "failed_cells": 0,
            "unsupported_keys": [],
            "failed_keys": [],
        },
        "evidence_complete_for_requested_matrix": True,
        "promotion_claim": False,
        "claim_boundary": "bounded raw artifact only",
    }


def test_wp03_validator_accepts_complete_bounded_artifact() -> None:
    _validator().validate(_document())


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty",
        "promotion",
        "source_hash",
        "missing_cell",
        "missing_baseline",
        "derived_timing",
        "numerical_failure",
        "missing_compile_isolation",
        "mismatched_compile_isolation",
    ],
)
def test_wp03_validator_rejects_adversarial_mutations(mutation: str) -> None:
    document = copy.deepcopy(_document())
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)

    if mutation == "dirty":
        document["dirty_worktree"] = True
    elif mutation == "promotion":
        document["promotion_claim"] = True
    elif mutation == "source_hash":
        cell["generated_source_sha256"] = "0" * 64
    elif mutation == "missing_cell":
        document["cells"] = []
    elif mutation == "missing_baseline":
        baselines = cell["baselines"]
        assert isinstance(baselines, dict)
        del baselines["direct_triton"]
    elif mutation == "derived_timing":
        timings = cell["timings"]
        assert isinstance(timings, dict)
        ordinary = timings["ordinary"]
        assert isinstance(ordinary, dict)
        ordinary["samples_ms"] = []
    elif mutation == "numerical_failure":
        numerical = cell["numerical"]
        assert isinstance(numerical, dict)
        numerical["passed"] = False
    elif mutation == "missing_compile_isolation":
        del document["torch_compile_isolation"]
    elif mutation == "mismatched_compile_isolation":
        isolation = document["torch_compile_isolation"]
        assert isinstance(isolation, dict)
        isolation["graph_families"] = ["tanh"]
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        _validator().validate(document)
