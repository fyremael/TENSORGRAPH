"""Fail CI when withdrawn evidence patterns or release claims reappear."""

from __future__ import annotations

from pathlib import Path

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

    if failures:
        print("Evidence/claim policy violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Evidence and claim policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
