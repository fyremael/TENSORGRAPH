"""Validate TG-GPU-WP02 raw evidence and optional promotion bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "tensorgraph.evidence.portability-training.v1"
PACKAGE = "TG-GPU-WP02"
OPERATIONS = {"sigmoid", "tanh"}
DTYPES = {"float16", "bfloat16", "float32"}
REGIMES = {
    "moderate",
    "positive_saturation",
    "negative_saturation",
    "near_zero",
    "mixed_edge",
}
DIRECTIONS = {"forward", "forward_backward"}
DISPOSITIONS = {"passed", "unsupported", "failed"}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _matrix_key(dtype: str, operation: str, regime: str, direction: str) -> str:
    return "/".join((dtype, operation, regime, direction))


def _list_of_strings(
    failures: list[str],
    request: dict[str, Any],
    name: str,
    allowed: set[str],
) -> list[str]:
    value = request.get(name)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        failures.append(f"request.{name} must be a non-empty string list")
        return []
    if len(value) != len(set(value)):
        failures.append(f"request.{name} must not contain duplicates")
    unknown = set(value) - allowed
    if unknown:
        failures.append(f"request.{name} contains unsupported values: {sorted(unknown)!r}")
    return value


def _validate_numerical(
    failures: list[str],
    numerical: Any,
    *,
    label: str,
    required_pass: bool,
) -> None:
    if not isinstance(numerical, dict):
        failures.append(f"{label} must be an object")
        return
    for key in ("pass", "rtol", "atol", "max_abs_error", "max_rel_error"):
        if key not in numerical:
            failures.append(f"{label} is missing {key!r}")
    if required_pass and numerical.get("pass") is not True:
        failures.append(f"{label} must record pass=true")
    if numerical.get("special_values_match") is not True:
        failures.append(f"{label} must preserve NaN and infinity classifications")


def _validate_timing(
    failures: list[str],
    timing: Any,
    *,
    label: str,
    repetitions: int,
    required: bool,
) -> None:
    if not isinstance(timing, dict):
        failures.append(f"{label} must be an object")
        return
    raw = timing.get("raw_ms")
    if not isinstance(raw, list):
        failures.append(f"{label}.raw_ms must be a list")
        return
    if required and len(raw) != repetitions:
        failures.append(f"{label}.raw_ms must contain {repetitions} samples")
    if not required and raw:
        failures.append(f"{label}.raw_ms must be empty for a non-passed cell")
    if required and not isinstance(timing.get("summary"), dict):
        failures.append(f"{label}.summary must be present for a passed cell")


def validate_evidence(document: dict[str, Any]) -> list[str]:
    """Return all evidence-contract violations for one raw artifact."""

    failures: list[str] = []
    if document.get("schema") != SCHEMA:
        failures.append(f"schema must equal {SCHEMA!r}")
    if document.get("package") != PACKAGE:
        failures.append(f"package must equal {PACKAGE!r}")
    if document.get("repository") != "fyremael/TENSORGRAPH":
        failures.append("repository must equal 'fyremael/TENSORGRAPH'")
    if not _is_commit(document.get("commit_sha")):
        failures.append("commit_sha must be an exact 40-character hexadecimal commit")
    if document.get("dirty_worktree") is not False:
        failures.append("dirty_worktree must be false")
    if document.get("promotion_claim") is not False:
        failures.append("raw WP02 evidence must not make a promotion claim")

    stack_id = document.get("stack_id")
    if not isinstance(stack_id, str) or not stack_id.strip():
        failures.append("stack_id must be a non-empty string")
    repetitions = document.get("repetitions")
    if not isinstance(repetitions, int) or repetitions <= 0:
        failures.append("repetitions must be a positive integer")
        repetitions = 0

    versions = document.get("versions")
    if not isinstance(versions, dict):
        failures.append("versions must be an object")
    else:
        for key in ("torch", "triton", "cuda_runtime", "driver"):
            if key not in versions:
                failures.append(f"versions is missing {key!r}")

    hardware = document.get("hardware")
    if not isinstance(hardware, dict):
        failures.append("hardware must be an object")
    else:
        for key in (
            "gpu_index",
            "gpu_name",
            "gpu_total_memory_bytes",
            "gpu_compute_capability",
        ):
            if key not in hardware:
                failures.append(f"hardware is missing {key!r}")
        capability = hardware.get("gpu_compute_capability")
        if not (
            isinstance(capability, list)
            and len(capability) == 2
            and all(isinstance(item, int) for item in capability)
        ):
            failures.append("hardware.gpu_compute_capability must be [major, minor]")

    request = document.get("request")
    if not isinstance(request, dict):
        failures.append("request must be an object")
        request = {}
    operations = _list_of_strings(failures, request, "operations", OPERATIONS)
    dtypes = _list_of_strings(failures, request, "dtypes", DTYPES)
    regimes = _list_of_strings(failures, request, "regimes", REGIMES)
    directions = _list_of_strings(failures, request, "directions", DIRECTIONS)

    compiler = document.get("compiler")
    if not isinstance(compiler, dict):
        failures.append("compiler must be an object")
        compiler = {}
    for operation in operations:
        record = compiler.get(operation)
        if not isinstance(record, dict):
            failures.append(f"compiler is missing operation {operation!r}")
            continue
        if record.get("status") == "failed":
            if not record.get("failure_reason"):
                failures.append(f"compiler.{operation} failed without a reason")
            continue
        forward_source = record.get("forward_generated_source")
        backward_source = record.get("backward_generated_source")
        forward_sha = record.get("forward_generated_source_sha256")
        backward_sha = record.get("backward_generated_source_sha256")
        if not isinstance(forward_source, str) or not forward_source:
            failures.append(f"compiler.{operation} is missing exact forward source")
        elif hashlib.sha256(forward_source.encode("utf-8")).hexdigest() != forward_sha:
            failures.append(f"compiler.{operation} forward source SHA-256 mismatch")
        if not isinstance(backward_source, str) or not backward_source:
            failures.append(f"compiler.{operation} is missing exact backward source")
        elif hashlib.sha256(backward_source.encode("utf-8")).hexdigest() != backward_sha:
            failures.append(f"compiler.{operation} backward source SHA-256 mismatch")
        if not _is_sha256(forward_sha):
            failures.append(f"compiler.{operation} forward SHA-256 is malformed")
        if not _is_sha256(backward_sha):
            failures.append(f"compiler.{operation} backward SHA-256 is malformed")
        if isinstance(forward_source, str):
            if "tl.sigmoid" in forward_source:
                failures.append(f"compiler.{operation} reintroduced tl.sigmoid")
            if "tl.exp" not in forward_source:
                failures.append(f"compiler.{operation} forward source omits tl.exp")
        if isinstance(backward_source, str) and "grad_output" not in backward_source:
            failures.append(f"compiler.{operation} backward source omits grad_output")
        for timing_key in (
            "forward_phase_ns",
            "backward_source_generation_ns",
            "compile_total_ns",
            "forward_source_load_ns",
            "backward_source_load_ns",
        ):
            if timing_key not in record:
                failures.append(f"compiler.{operation} is missing {timing_key!r}")

    expected_keys = {
        _matrix_key(dtype, operation, regime, direction)
        for dtype in dtypes
        for operation in operations
        for regime in regimes
        for direction in directions
    }
    cells = document.get("cells")
    if not isinstance(cells, list):
        failures.append("cells must be a list")
        cells = []
    actual_keys: list[str] = []
    for index, cell in enumerate(cells):
        label = f"cells[{index}]"
        if not isinstance(cell, dict):
            failures.append(f"{label} must be an object")
            continue
        key = cell.get("key")
        if not isinstance(key, str):
            failures.append(f"{label}.key must be a string")
            continue
        actual_keys.append(key)
        expected_key = _matrix_key(
            str(cell.get("dtype")),
            str(cell.get("operation")),
            str(cell.get("regime")),
            str(cell.get("direction")),
        )
        if key != expected_key:
            failures.append(f"{label}.key does not match its dimensions")
        disposition = cell.get("disposition")
        if disposition not in DISPOSITIONS:
            failures.append(f"{label}.disposition is invalid")
            continue
        if disposition == "passed":
            _validate_numerical(
                failures,
                cell.get("forward_numerical"),
                label=f"{label}.forward_numerical",
                required_pass=True,
            )
            if cell.get("direction") == "forward_backward":
                _validate_numerical(
                    failures,
                    cell.get("gradient_numerical"),
                    label=f"{label}.gradient_numerical",
                    required_pass=True,
                )
            _validate_timing(
                failures,
                cell.get("reference_timing"),
                label=f"{label}.reference_timing",
                repetitions=repetitions,
                required=True,
            )
            _validate_timing(
                failures,
                cell.get("tensorgraph_timing"),
                label=f"{label}.tensorgraph_timing",
                repetitions=repetitions,
                required=True,
            )
            if not isinstance(cell.get("first_forward_execution_ns"), int):
                failures.append(f"{label} is missing first forward execution time")
            if cell.get("direction") == "forward_backward" and not isinstance(
                cell.get("first_backward_execution_ns"), int
            ):
                failures.append(f"{label} is missing first backward execution time")
        elif disposition == "unsupported":
            if not cell.get("unsupported_reason"):
                failures.append(f"{label} is unsupported without a reason")
            _validate_timing(
                failures,
                cell.get("reference_timing"),
                label=f"{label}.reference_timing",
                repetitions=repetitions,
                required=False,
            )
            _validate_timing(
                failures,
                cell.get("tensorgraph_timing"),
                label=f"{label}.tensorgraph_timing",
                repetitions=repetitions,
                required=False,
            )
        else:
            if not cell.get("failure_stage") or not cell.get("failure_reason"):
                failures.append(f"{label} failed without stage and reason")
            _validate_timing(
                failures,
                cell.get("reference_timing"),
                label=f"{label}.reference_timing",
                repetitions=repetitions,
                required=False,
            )
            _validate_timing(
                failures,
                cell.get("tensorgraph_timing"),
                label=f"{label}.tensorgraph_timing",
                repetitions=repetitions,
                required=False,
            )

    if len(actual_keys) != len(set(actual_keys)):
        failures.append("cells contains duplicate matrix keys")
    missing = expected_keys - set(actual_keys)
    extra = set(actual_keys) - expected_keys
    if missing:
        failures.append(f"cells is missing requested keys: {sorted(missing)!r}")
    if extra:
        failures.append(f"cells contains unrequested keys: {sorted(extra)!r}")

    summary = document.get("matrix_summary")
    if not isinstance(summary, dict):
        failures.append("matrix_summary must be an object")
    else:
        if summary.get("requested_cells") != len(expected_keys):
            failures.append("matrix_summary.requested_cells is inconsistent")
        if summary.get("recorded_cells") != len(cells):
            failures.append("matrix_summary.recorded_cells is inconsistent")
        counted = {
            disposition: sum(
                isinstance(cell, dict) and cell.get("disposition") == disposition
                for cell in cells
            )
            for disposition in DISPOSITIONS
        }
        for disposition, count in counted.items():
            field = f"{disposition}_cells"
            if summary.get(field) != count:
                failures.append(f"matrix_summary.{field} is inconsistent")

    return failures


def validate_promotion_bundle(documents: list[dict[str, Any]]) -> list[str]:
    """Validate the minimum cross-artifact promotion coverage."""

    failures: list[str] = []
    if len(documents) < 2:
        failures.append("promotion bundle requires at least two raw evidence artifacts")
        return failures

    for index, document in enumerate(documents):
        for failure in validate_evidence(document):
            failures.append(f"artifact[{index}]: {failure}")

    commits = {document.get("commit_sha") for document in documents}
    if len(commits) != 1:
        failures.append("all promotion artifacts must evaluate the same exact commit")

    stacks = {
        (
            document.get("versions", {}).get("torch"),
            document.get("versions", {}).get("triton"),
        )
        for document in documents
        if isinstance(document.get("versions"), dict)
    }
    if len(stacks) < 2:
        failures.append("promotion bundle requires two exact PyTorch/Triton stacks")

    capabilities: list[tuple[int, int]] = []
    gpu_names: list[str] = []
    for document in documents:
        hardware = document.get("hardware")
        if not isinstance(hardware, dict):
            continue
        name = hardware.get("gpu_name")
        capability = hardware.get("gpu_compute_capability")
        if isinstance(name, str):
            gpu_names.append(name)
        if (
            isinstance(capability, list)
            and len(capability) == 2
            and all(isinstance(item, int) for item in capability)
        ):
            capabilities.append((capability[0], capability[1]))
    if not any("T4" in name for name in gpu_names):
        failures.append("promotion bundle requires an explicit Tesla T4 artifact")
    if not any(major >= 8 for major, _minor in capabilities):
        failures.append("promotion bundle requires an Ampere-or-newer artifact")

    for index, document in enumerate(documents):
        request = document.get("request")
        if not isinstance(request, dict):
            continue
        for name, required in (
            ("operations", OPERATIONS),
            ("dtypes", DTYPES),
            ("regimes", REGIMES),
            ("directions", DIRECTIONS),
        ):
            values = request.get(name)
            if not isinstance(values, list) or set(values) != required:
                failures.append(f"artifact[{index}] does not request the full {name} matrix")

    if any(
        isinstance(cell, dict) and cell.get("disposition") == "failed"
        for document in documents
        for cell in document.get("cells", [])
        if isinstance(document.get("cells"), list)
    ):
        failures.append("promotion bundle contains failed matrix cells")

    return failures


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--promotion-bundle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = [_load(path) for path in args.evidence]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"EVIDENCE INVALID: {exc}", file=sys.stderr)
        return 1

    if args.promotion_bundle:
        failures = validate_promotion_bundle(documents)
    else:
        failures = []
        for path, document in zip(args.evidence, documents, strict=True):
            failures.extend(f"{path}: {failure}" for failure in validate_evidence(document))

    if failures:
        print("TG-GPU-WP02 evidence violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    for path in args.evidence:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"PASS {path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
