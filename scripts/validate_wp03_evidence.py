#!/usr/bin/env python3
"""Fail-closed validator for TG-GPU-WP03 raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

_SCHEMA = "tensorgraph.evidence.native-cuda-inference.v1"
_PACKAGE = "TG-GPU-WP03"
_LAUNCH_MODES = {"ordinary", "graph_replay", "copy_plus_graph_replay"}
_REQUIRED_BASELINES = {
    "pytorch_eager",
    "torch_compile",
    "tensorgraph_triton",
    "direct_triton",
    "tensorgraph_native_cuda",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_timing(record: Any, context: str) -> None:
    _require(isinstance(record, dict), f"{context} must be an object")
    samples = record.get("samples_ms")
    _require(isinstance(samples, list) and samples, f"{context}.samples_ms must be non-empty")
    _require(
        all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in samples),
        f"{context}.samples_ms contains invalid values",
    )
    for key in ("median_ms", "minimum_ms", "maximum_ms"):
        value = record.get(key)
        _require(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            f"{context}.{key} must be finite and non-negative",
        )
    _require(abs(min(samples) - record["minimum_ms"]) <= 1e-9, f"{context} minimum mismatch")
    _require(abs(max(samples) - record["maximum_ms"]) <= 1e-9, f"{context} maximum mismatch")


def validate(document: Any) -> None:
    _require(isinstance(document, dict), "evidence root must be an object")
    _require(document.get("schema") == _SCHEMA, "unexpected evidence schema")
    _require(document.get("package") == _PACKAGE, "unexpected package")
    _require(document.get("repository") == "fyremael/TENSORGRAPH", "unexpected repository")
    _require(document.get("dirty_worktree") is False, "dirty worktree evidence is inadmissible")
    _require(document.get("promotion_claim") is False, "raw WP03 evidence cannot claim promotion")
    _require(isinstance(document.get("claim_boundary"), str), "claim boundary is required")

    commit_sha = document.get("commit_sha")
    _require(
        isinstance(commit_sha, str)
        and len(commit_sha) == 40
        and all(character in "0123456789abcdef" for character in commit_sha),
        "commit_sha must be a lowercase 40-character SHA",
    )

    request = document.get("request")
    _require(isinstance(request, dict), "request must be an object")
    graphs = request.get("graphs")
    dtypes = request.get("dtypes")
    regimes = request.get("regimes")
    launch_modes = request.get("launch_modes")
    for value, name in (
        (graphs, "graphs"),
        (dtypes, "dtypes"),
        (regimes, "regimes"),
        (launch_modes, "launch_modes"),
    ):
        _require(isinstance(value, list) and value, f"request.{name} must be non-empty")
        _require(len(value) == len(set(value)), f"request.{name} contains duplicates")
    _require(set(launch_modes) <= _LAUNCH_MODES, "request contains an unknown launch mode")

    isolation = document.get("torch_compile_isolation")
    _require(isinstance(isolation, dict), "torch_compile_isolation must be an object")
    _require(
        isolation.get("mode") == "torch._dynamo.reset_per_graph_family",
        "torch.compile graph-family isolation mode is missing or unsupported",
    )
    _require(
        isolation.get("graph_families") == graphs,
        "torch.compile isolation graph families do not match the request",
    )

    expected_keys = {
        f"{graph}__{dtype}__{regime}__{launch_mode}"
        for graph, dtype, regime, launch_mode in itertools.product(
            graphs, dtypes, regimes, launch_modes
        )
    }
    cells = document.get("cells")
    _require(isinstance(cells, list), "cells must be an array")
    actual_keys: set[str] = set()
    passed = unsupported = failed = 0

    for index, cell in enumerate(cells):
        context = f"cells[{index}]"
        _require(isinstance(cell, dict), f"{context} must be an object")
        key = cell.get("key")
        _require(isinstance(key, str) and key, f"{context}.key is required")
        _require(key not in actual_keys, f"duplicate cell key: {key}")
        actual_keys.add(key)
        expected_key = (
            f"{cell.get('graph')}__{cell.get('dtype')}__{cell.get('regime')}__"
            f"{cell.get('launch_mode')}"
        )
        _require(key == expected_key, f"cell key fields disagree: {key}")
        _require(key in expected_keys, f"unrequested cell: {key}")

        disposition = cell.get("disposition")
        _require(disposition in {"passed", "unsupported", "failed"}, f"invalid disposition: {key}")
        source = cell.get("generated_source")
        source_sha = cell.get("generated_source_sha256")
        _require(isinstance(source, str), f"{key} generated_source must be a string")
        _require(_is_sha256(source_sha), f"{key} has invalid generated source SHA")
        if source:
            actual_source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
            _require(actual_source_sha == source_sha, f"{key} generated source SHA mismatch")

        if disposition == "passed":
            passed += 1
            _require(bool(source), f"passed cell {key} must retain generated source")
            compiler = cell.get("compiler")
            _require(isinstance(compiler, dict), f"{key} compiler record missing")
            _require(compiler.get("status") == "compiled", f"{key} was not compiled")
            numerical = cell.get("numerical")
            _require(isinstance(numerical, dict), f"{key} numerical record missing")
            _require(numerical.get("passed") is True, f"{key} numerical gate did not pass")
            timings = cell.get("timings")
            _require(isinstance(timings, dict) and timings, f"{key} timings missing")
            for timing_name, timing_record in timings.items():
                _validate_timing(timing_record, f"{key}.timings.{timing_name}")
            baselines = cell.get("baselines")
            _require(isinstance(baselines, dict), f"{key} baselines missing")
            _require(
                _REQUIRED_BASELINES <= set(baselines),
                f"{key} is missing required independent baselines",
            )
        elif disposition == "unsupported":
            unsupported += 1
            _require(
                isinstance(cell.get("unsupported_reason"), str)
                and bool(cell["unsupported_reason"]),
                f"unsupported cell {key} requires a reason",
            )
        else:
            failed += 1
            _require(
                isinstance(cell.get("failure_reason"), str) and bool(cell["failure_reason"]),
                f"failed cell {key} requires a reason",
            )

    _require(actual_keys == expected_keys, "recorded cells do not equal the requested matrix")
    summary = document.get("matrix_summary")
    _require(isinstance(summary, dict), "matrix_summary must be an object")
    _require(summary.get("requested_cells") == len(expected_keys), "requested cell count mismatch")
    _require(summary.get("recorded_cells") == len(cells), "recorded cell count mismatch")
    _require(summary.get("passed_cells") == passed, "passed cell count mismatch")
    _require(summary.get("unsupported_cells") == unsupported, "unsupported count mismatch")
    _require(summary.get("failed_cells") == failed, "failed count mismatch")
    _require(
        document.get("evidence_complete_for_requested_matrix") is True,
        "evidence must be complete for its requested matrix",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate(document)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"WP03 evidence rejected: {exc}", file=sys.stderr)
        return 1
    print("WP03 evidence accepted for its bounded raw-artifact contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
