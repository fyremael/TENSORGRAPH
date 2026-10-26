"""Fail CI when withdrawn evidence patterns or release claims reappear."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve()

QUARANTINED_REPORTS = [
    "AUDIT_REPORT.md",
    "MASTER_COMPILER_BENCHMARK_REPORT.md",
    "MASTER_GRAND_CHALLENGE_BENCHMARK_REPORT.md",
    "PYTHIA_SUITE_BENCHMARK_REPORT.md",
    "REALWORLD_OPTIMIZATION_REPORT.md",
    "TESTBENCH_REPORT.md",
]

PROHIBITED_SOURCE_FRAGMENTS = {
    "tensorgraph_latency_ms = inductor_latency_ms *": "derived latency substitution",
    "tensorgraph_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) *": (
        "derived memory substitution"
    ),
}

EXPECTED_EVALUATED_COMMIT = "92bfa21538e60a4cc321f32f7340ba70eee00db0"
EXPECTED_MERGE_COMMIT = "19fd6760d9b876c34880a79933c3e6914bf8fbf4"
EXPECTED_ARTIFACTS = {
    "six_baseline": "f0b4003f0f1250f4e4430a65897ef1bbbe8a6659f88fca9c74f2273538901c40",
    "sigmoid": "e7fb0f2e7050e050d34857ee57b8547c306592625e4244b20ae4378531dd155a",
    "tanh": "e9568d902d1dfd12e6816f28527c8a011e4c4e9e4ae14623a19b47cad9de5361",
    "nonlinear_bundle": "a62b75014f08207d4f60b2c20be4f340f747599e45a1bc286d1564c00ca593c8",
}
EXPECTED_GENERATED_SOURCES = {
    "sigmoid": "d39a14b53fe10ee1e02adf00861ff0069778fcf553f5e120974882572c30256a",
    "tanh": "1b3b8c50dcb1766edb9d043faf8372c06d444b06ed3ef108fabf6edf509555b2",
}


def _require_fragments(
    failures: list[str],
    *,
    relative: str,
    fragments: tuple[str, ...],
    label: str,
) -> None:
    path = ROOT / relative
    if not path.exists():
        failures.append(f"missing {label}: {relative}")
        return
    text = path.read_text(encoding="utf-8")
    for fragment in fragments:
        if fragment not in text:
            failures.append(f"{label} {relative} is missing required fragment: {fragment}")


def _load_json(failures: list[str], relative: str, label: str) -> dict[str, Any] | None:
    path = ROOT / relative
    if not path.exists():
        failures.append(f"missing {label}: {relative}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid {label} {relative}: {exc}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label} {relative} must contain a JSON object")
        return None
    return value


def _check_gpu_admission(failures: list[str]) -> None:
    relative = "evidence/TG-GPU-WP01/ADMISSION.json"
    admission = _load_json(failures, relative, "GPU admission record")
    if admission is None:
        return

    expected_scalars = {
        "schema": "tensorgraph.evidence.admission.v1",
        "package": "TG-GPU-WP01",
        "status": "complete",
        "repository": "fyremael/TENSORGRAPH",
        "evaluated_commit_sha": EXPECTED_EVALUATED_COMMIT,
        "merge_commit_sha": EXPECTED_MERGE_COMMIT,
        "pull_request": 2,
    }
    for key, expected in expected_scalars.items():
        if admission.get(key) != expected:
            failures.append(f"GPU admission field {key!r} must equal {expected!r}")

    ci = admission.get("ci")
    if not isinstance(ci, dict) or ci.get("conclusion") != "success":
        failures.append("GPU admission must record a successful CI conclusion")

    environment = admission.get("environment")
    expected_environment = {
        "gpu_name": "Tesla T4",
        "gpu_compute_capability": [7, 5],
        "torch": "2.11.0+cu128",
        "triton": "3.6.0",
        "cuda_runtime": "12.8",
    }
    if not isinstance(environment, dict):
        failures.append("GPU admission environment must be an object")
    else:
        for key, expected in expected_environment.items():
            if environment.get(key) != expected:
                failures.append(f"GPU admission environment {key!r} must equal {expected!r}")

    artifacts = admission.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("GPU admission artifacts must be an object")
    else:
        for name, expected_sha in EXPECTED_ARTIFACTS.items():
            artifact = artifacts.get(name)
            if not isinstance(artifact, dict):
                failures.append(f"GPU admission is missing artifact {name!r}")
                continue
            if artifact.get("sha256") != expected_sha:
                failures.append(f"GPU admission artifact {name!r} has the wrong SHA-256")
            generated_sha = EXPECTED_GENERATED_SOURCES.get(name)
            if generated_sha is not None and artifact.get("generated_source_sha256") != generated_sha:
                failures.append(
                    f"GPU admission artifact {name!r} has the wrong generated-source SHA-256"
                )
            if name != "nonlinear_bundle" and artifact.get("numerical_status") not in {
                "all_passed",
                "all_passed_exact",
            }:
                failures.append(f"GPU admission artifact {name!r} is not numerically admitted")

    decision = admission.get("admission")
    expected_decision = {
        "sigmoid_gpu_evidence": "admitted",
        "tanh_gpu_evidence": "admitted",
        "six_baseline_gpu_evidence": "admitted",
        "portable_lowering_contains": "tl.exp",
        "portable_lowering_excludes": "tl.sigmoid",
    }
    if not isinstance(decision, dict):
        failures.append("GPU admission decision must be an object")
    else:
        for key, expected in expected_decision.items():
            if decision.get(key) != expected:
                failures.append(f"GPU admission decision {key!r} must equal {expected!r}")

    checksums_path = ROOT / "evidence/TG-GPU-WP01/SHA256SUMS"
    if not checksums_path.exists():
        failures.append("missing GPU evidence checksum ledger")
    else:
        checksums = checksums_path.read_text(encoding="utf-8")
        for expected_sha in EXPECTED_ARTIFACTS.values():
            if expected_sha not in checksums:
                failures.append(f"GPU evidence checksum ledger is missing {expected_sha}")


def _check_wp02_contract(failures: list[str]) -> None:
    _require_fragments(
        failures,
        relative="tensorgraph/pipeline/training_elementwise.py",
        label="WP02 generated backward pipeline",
        fragments=(
            "compile_fx_elementwise_training",
            "generated_backward_source",
            "backward_source_sha256",
            "grad_output",
            "derivative = y * (1.0 - y)",
            "derivative = 1.0 - y * y",
            "CUDA is required",
        ),
    )
    _require_fragments(
        failures,
        relative="benchmarks/bench_portability_training.py",
        label="WP02 matrix runner",
        fragments=(
            'OPERATIONS = ("sigmoid", "tanh")',
            'DTYPES = ("float16", "bfloat16", "float32")',
            '"positive_saturation"',
            '"negative_saturation"',
            '"near_zero"',
            '"mixed_edge"',
            'DIRECTIONS = ("forward", "forward_backward")',
            '"disposition": "unsupported"',
            '"disposition": "failed"',
            '"promotion_claim": False',
            '"dirty_worktree": dirty',
            '"forward_generated_source_sha256"',
            '"backward_generated_source_sha256"',
            '"raw_ms"',
        ),
    )
    _require_fragments(
        failures,
        relative="scripts/validate_wp02_evidence.py",
        label="WP02 evidence validator",
        fragments=(
            "validate_evidence",
            "validate_promotion_bundle",
            "missing requested keys",
            "source SHA-256 mismatch",
            "Tesla T4",
            "Ampere-or-newer",
            "two exact PyTorch/Triton stacks",
        ),
    )
    _require_fragments(
        failures,
        relative="docs/TG_GPU_WP02.md",
        label="WP02 interpretation charter",
        fragments=(
            "no CUDA portability evidence admitted",
            "60 requested cells",
            "Exact-source contract",
            "Promotion gates",
            "production readiness",
        ),
    )

    schema = _load_json(
        failures,
        "schemas/tg_gpu_wp02_evidence.schema.json",
        "WP02 evidence schema",
    )
    if schema is not None:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            failures.append("WP02 evidence schema properties must be an object")
        else:
            expected_constants = {
                "schema": "tensorgraph.evidence.portability-training.v1",
                "package": "TG-GPU-WP02",
                "repository": "fyremael/TENSORGRAPH",
                "dirty_worktree": False,
                "promotion_claim": False,
            }
            for name, expected in expected_constants.items():
                definition = properties.get(name)
                if not isinstance(definition, dict) or definition.get("const") != expected:
                    failures.append(
                        f"WP02 evidence schema property {name!r} must pin const={expected!r}"
                    )

    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for required in (
        "Generated Sigmoid and Tanh input gradients | experimental",
        "GPU portability matrix | experimental",
        "no WP02 CUDA portability or backward evidence is admitted",
    ):
        if required not in status:
            failures.append(f"STATUS.md is missing WP02 boundary: {required!r}")


def main() -> int:
    failures: list[str] = []

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in ("STATUS.md", "docs/EVIDENCE_POLICY.md", "not established"):
        if required not in readme:
            failures.append(f"README.md does not preserve required boundary: {required!r}")

    for relative in QUARANTINED_REPORTS:
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing quarantine marker file: {relative}")
            continue
        text = path.read_text(encoding="utf-8").lower()
        if "quarantined" not in text or "authority" not in text:
            failures.append(f"historical report is not explicitly quarantined: {relative}")
        if "file:///" in text:
            failures.append(f"machine-local file link remains in report: {relative}")

    for path in ROOT.rglob("*.py"):
        resolved = path.resolve()
        if resolved == CHECKER:
            continue
        if any(part in {".git", ".venv", "build", "dist"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment, description in PROHIBITED_SOURCE_FRAGMENTS.items():
            if fragment in text:
                failures.append(f"{path.relative_to(ROOT)} contains {description}")

    _require_fragments(
        failures,
        relative="benchmarks/bench_verified_elementwise.py",
        label="verified benchmark",
        fragments=(
            "generated.run",
            "raw_ms",
            "generated_source_sha256",
            "dirty_worktree",
            "torch.allclose",
            "--terminal-op",
        ),
    )
    _require_fragments(
        failures,
        relative="benchmarks/bench_six_baseline_elementwise.py",
        label="six-baseline benchmark",
        fragments=(
            "pytorch_eager_source",
            "pytorch_eager_optimized",
            "torch_compile_source",
            "torch_compile_optimized",
            "tensorgraph_generated",
            "direct_triton_reference",
            "raw_ms",
            "dirty_worktree",
        ),
    )
    _require_fragments(
        failures,
        relative="docs/GPU_EVIDENCE_PACKAGE.md",
        label="GPU evidence package",
        fragments=(
            "Status:** complete",
            "ADMISSION.json",
            "Obligation A",
            "Obligation B",
            "six-baseline",
            "Sigmoid",
            "Tanh",
        ),
    )

    lowering = (ROOT / "tensorgraph/pipeline/verified_elementwise.py").read_text(
        encoding="utf-8"
    )
    if 'lines.append("    value = tl.sigmoid(value)")' in lowering:
        failures.append("verified lowering reintroduced direct tl.sigmoid emission")
    for required in (
        'lines.append("    value = 1.0 / (1.0 + tl.exp(-value))")',
        'lines.append("    value = 2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0")',
    ):
        if required not in lowering:
            failures.append(f"verified lowering is missing portable expression: {required}")

    _check_gpu_admission(failures)
    _check_wp02_contract(failures)

    if failures:
        print("Evidence/claim policy violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Evidence and claim policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
