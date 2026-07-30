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

    benchmark = (ROOT / "benchmarks/bench_verified_elementwise.py").read_text(encoding="utf-8")
    for required in (
        "generated.run",
        "raw_ms",
        "generated_source_sha256",
        "dirty_worktree",
        "torch.allclose",
    ):
        if required not in benchmark:
            failures.append(f"verified benchmark is missing required evidence field: {required}")

    if failures:
        print("Evidence/claim policy violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Evidence and claim policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
