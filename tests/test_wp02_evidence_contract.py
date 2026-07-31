from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load(relative: str, name: str) -> ModuleType:
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compiler_record(operation: str) -> dict[str, Any]:
    if operation == "sigmoid":
        forward_source = "value = 1.0 / (1.0 + tl.exp(-value))\n"
    else:
        forward_source = "value = 2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0\n"
    backward_source = "grad_input = grad_output * derivative\n"
    return {
        "status": "compiled",
        "source_expr": operation,
        "optimized_expr": operation,
        "optimized_ops": [operation.title()],
        "rewrite_summary": {},
        "forward_generated_source": forward_source,
        "forward_generated_source_sha256": hashlib.sha256(
            forward_source.encode("utf-8")
        ).hexdigest(),
        "backward_generated_source": backward_source,
        "backward_generated_source_sha256": hashlib.sha256(
            backward_source.encode("utf-8")
        ).hexdigest(),
        "forward_phase_ns": {
            "fx_capture": 1,
            "ir_construction": 1,
            "saturation": 1,
            "extraction": 1,
            "source_generation": 1,
        },
        "backward_source_generation_ns": 1,
        "compile_total_ns": 6,
        "forward_source_load_ns": 1,
        "backward_source_load_ns": 1,
    }


def _numerical() -> dict[str, Any]:
    return {
        "pass": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "finite_count": 4,
        "actual_nan_count": 0,
        "expected_nan_count": 0,
        "actual_posinf_count": 0,
        "expected_posinf_count": 0,
        "actual_neginf_count": 0,
        "expected_neginf_count": 0,
        "special_values_match": True,
    }


def _evidence(*, gpu_name: str = "Tesla T4", capability: list[int] | None = None) -> dict[str, Any]:
    capability = capability or [7, 5]
    repetitions = 2
    operations = ["sigmoid"]
    dtypes = ["float32"]
    regimes = ["moderate"]
    directions = ["forward", "forward_backward"]
    cells = []
    for direction in directions:
        cells.append(
            {
                "key": f"float32/sigmoid/moderate/{direction}",
                "operation": "sigmoid",
                "dtype": "float32",
                "regime": "moderate",
                "direction": direction,
                "size": 4,
                "shape": [4],
                "seed": 1,
                "disposition": "passed",
                "first_forward_execution_ns": 1,
                "first_forward_includes_jit": direction == "forward",
                "first_backward_execution_ns": 1 if direction == "forward_backward" else None,
                "first_backward_includes_jit": direction == "forward_backward",
                "forward_numerical": _numerical(),
                "gradient_numerical": _numerical() if direction == "forward_backward" else None,
                "reference_timing": {
                    "raw_ms": [1.0, 1.1],
                    "summary": {"mean_ms": 1.05},
                },
                "tensorgraph_timing": {
                    "raw_ms": [0.5, 0.6],
                    "summary": {"mean_ms": 0.55},
                },
            }
        )
    return {
        "schema": "tensorgraph.evidence.portability-training.v1",
        "package": "TG-GPU-WP02",
        "repository": "fyremael/TENSORGRAPH",
        "timestamp_utc": "2026-07-31T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "dirty_worktree": False,
        "command": ["python", "bench.py"],
        "stack_id": "torch-test-triton-test",
        "seed": 1,
        "size": 4,
        "warmup": 1,
        "repetitions": repetitions,
        "block_size": 256,
        "request": {
            "operations": operations,
            "dtypes": dtypes,
            "regimes": regimes,
            "directions": directions,
        },
        "versions": {
            "python": "3.10",
            "platform": "test",
            "torch": "test",
            "triton": "test",
            "cuda_runtime": "test",
            "cudnn": None,
            "driver": "test",
            "pid": 1,
        },
        "hardware": {
            "gpu_index": 0,
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": 1,
            "gpu_compute_capability": capability,
            "cpu": "test",
        },
        "tolerances": {},
        "compiler": {"sigmoid": _compiler_record("sigmoid")},
        "cells": cells,
        "matrix_summary": {
            "requested_cells": 2,
            "recorded_cells": 2,
            "passed_cells": 2,
            "unsupported_cells": 0,
            "failed_cells": 0,
            "unsupported_keys": [],
            "failed_keys": [],
        },
        "evidence_complete_for_requested_matrix": True,
        "promotion_claim": False,
        "claim_boundary": "bounded",
    }


def test_runner_declares_complete_governed_dimensions() -> None:
    module = _load(
        "benchmarks/bench_portability_training.py",
        "tensorgraph_portability_training_benchmark",
    )

    assert module.OPERATIONS == ("sigmoid", "tanh")
    assert module.DTYPES == ("float16", "bfloat16", "float32")
    assert module.REGIMES == (
        "moderate",
        "positive_saturation",
        "negative_saturation",
        "near_zero",
        "mixed_edge",
    )
    assert module.DIRECTIONS == ("forward", "forward_backward")
    assert len(
        module.expected_matrix_keys(
            module.DTYPES,
            module.OPERATIONS,
            module.REGIMES,
            module.DIRECTIONS,
        )
    ) == 60


def test_validator_accepts_complete_individual_artifact() -> None:
    validator = _load(
        "scripts/validate_wp02_evidence.py",
        "tensorgraph_wp02_evidence_validator",
    )

    assert validator.validate_evidence(_evidence()) == []


def test_validator_rejects_omission_and_source_identity_mutation() -> None:
    validator = _load(
        "scripts/validate_wp02_evidence.py",
        "tensorgraph_wp02_evidence_validator_mutation",
    )
    evidence = _evidence()
    evidence["cells"].pop()
    evidence["compiler"]["sigmoid"]["forward_generated_source"] += "# mutation\n"

    failures = validator.validate_evidence(evidence)

    assert any("source SHA-256 mismatch" in failure for failure in failures)
    assert any("missing requested keys" in failure for failure in failures)


def test_schema_declares_fail_closed_contract() -> None:
    import json

    path = Path(__file__).parents[1] / "schemas" / "tg_gpu_wp02_evidence.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema"]["const"] == (
        "tensorgraph.evidence.portability-training.v1"
    )
    assert schema["properties"]["dirty_worktree"]["const"] is False
    assert schema["properties"]["promotion_claim"]["const"] is False
    assert set(schema["properties"]["cells"]["items"]["$ref"].split("/")) >= {"cell"}


def test_promotion_validator_requires_t4_ampere_and_two_stacks() -> None:
    validator = _load(
        "scripts/validate_wp02_evidence.py",
        "tensorgraph_wp02_promotion_validator",
    )
    t4 = _evidence()
    ampere = _evidence(gpu_name="NVIDIA A100", capability=[8, 0])
    ampere["versions"]["torch"] = "other"
    ampere["versions"]["triton"] = "other"

    failures = validator.validate_promotion_bundle([t4, ampere])

    assert any("full operations matrix" in failure for failure in failures)
    assert not any("Tesla T4" in failure for failure in failures)
    assert not any("Ampere-or-newer" in failure for failure in failures)
    assert not any("two exact PyTorch/Triton" in failure for failure in failures)
