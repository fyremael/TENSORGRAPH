"""
TENSORGRAPH Testbench Runner & Report Generator.
=================================================
Orchestrates testbench workload suites, renders Chrome Metropolis styled terminal
output, and generates JSON and Markdown validation report artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

from ..cli import style as S
from .evaluator import Evaluator, EvaluationResult
from .workloads import Workload, get_all_workloads


@dataclass
class BenchmarkReport:
    """Aggregated summary report of the testbench suite execution."""
    timestamp: str
    total_workloads: int
    passed_correctness_count: int
    avg_cost_reduction_pct: float
    avg_node_reduction_pct: float
    avg_saturation_time_ms: float
    results: list[EvaluationResult]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_workloads": self.total_workloads,
            "passed_correctness_count": self.passed_correctness_count,
            "avg_cost_reduction_pct": self.avg_cost_reduction_pct,
            "avg_node_reduction_pct": self.avg_node_reduction_pct,
            "avg_saturation_time_ms": self.avg_saturation_time_ms,
            "results": [asdict(r) for r in self.results],
        }


class TestbenchRunner:
    """Main execution engine for the TENSORGRAPH testbench suite."""
    __test__ = False

    def __init__(
        self,
        verify_correctness: bool = True,
        iterations: int = 10,
        output_dir: Optional[Path] = None,
    ):
        self.evaluator = Evaluator(verify_correctness=verify_correctness, iterations=iterations)
        self.output_dir = output_dir or Path(".")

    def run(self, category_filter: Optional[str] = None) -> BenchmarkReport:
        """Run selected testbench workloads and generate execution metrics."""
        workloads = get_all_workloads()
        if category_filter:
            cat_lower = category_filter.lower()
            workloads = [w for w in workloads if cat_lower in w.category.lower() or cat_lower in w.name.lower()]

        print(S.header("TENSORGRAPH TESTBENCH SUITE", "FRONTIER VALIDATION"))
        print(S.metric("WORKLOADS LOADED", str(len(workloads)), S.cyan))
        print(S.metric("VERIFY CORRECTNESS", str(self.evaluator.verify_correctness), S.amber))
        print(S.metric("MAX SATURATION ITERS", str(self.evaluator.iterations), S.chrome))
        print(S.divider())

        results: list[EvaluationResult] = []
        for idx, wl in enumerate(workloads, 1):
            print(f"{S.bold(f'[{idx}/{len(workloads)}]')} {S.cyan(wl.name)} ({S.dim(wl.category)})")
            res = self.evaluator.evaluate_workload(wl)
            results.append(res)

            status_str = S.green("PASS") if res.correctness_passed else S.red("FAIL")
            cost_change = f"{res.cost_before:.1f} → {res.cost_after:.1f} ({res.cost_reduction_pct:.1f}% reduction)"
            print(f"  {S.dim('Cost:')} {S.bold(cost_change)}")
            print(f"  {S.dim('Nodes:')} {res.nodes_before} → {res.nodes_after} ({res.node_reduction_pct:.1f}% reduction, peak: {res.peak_nodes})")
            print(f"  {S.dim('Latency:')} {res.saturation_time_ms:.2f} ms ({res.iterations} iters, {res.trace_entries_count} proof trace entries)")
            print(f"  {S.dim('Correctness:')} [{status_str}] max_diff={res.correctness_max_diff:.2e}")
            if res.extracted_expr_pretty:
                print(f"  {S.dim('Extracted:')} {S.chrome(res.extracted_expr_pretty)}")
            print()

        # Compute aggregates
        total = len(results)
        passed_correctness = sum(1 for r in results if r.correctness_passed)
        avg_cost_red = sum(r.cost_reduction_pct for r in results) / max(1, total)
        avg_node_red = sum(r.node_reduction_pct for r in results) / max(1, total)
        avg_time = sum(r.saturation_time_ms for r in results) / max(1, total)

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        report = BenchmarkReport(
            timestamp=timestamp_str,
            total_workloads=total,
            passed_correctness_count=passed_correctness,
            avg_cost_reduction_pct=avg_cost_red,
            avg_node_reduction_pct=avg_node_red,
            avg_saturation_time_ms=avg_time,
            results=results,
        )

        # Print Summary Card
        print(S.section("SUMMARY AGGREGATES"))
        print(S.metric("WORKLOADS EVALUATED", f"{total}", S.chrome))
        print(S.metric("NUMERICAL VERIFICATION", f"{passed_correctness}/{total} PASSED", S.green if passed_correctness == total else S.red))
        print(S.metric("AVG COST REDUCTION", f"{avg_cost_red:.2f}%", S.green))
        print(S.metric("AVG NODE REDUCTION", f"{avg_node_red:.2f}%", S.cyan))
        print(S.metric("AVG SATURATION LATENCY", f"{avg_time:.2f} ms", S.amber))
        print(S.divider())

        # Save artifacts
        self._export_artifacts(report)
        return report

    def _export_artifacts(self, report: BenchmarkReport) -> None:
        """Export JSON data and Markdown report artifacts."""
        json_path = self.output_dir / "testbench_results.json"
        md_path = self.output_dir / "TESTBENCH_REPORT.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_content = self.generate_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(S.metric("JSON ARTIFACT", str(json_path.resolve()), S.dim))
        print(S.metric("MARKDOWN REPORT", str(md_path.resolve()), S.dim))

    @staticmethod
    def generate_markdown_report(report: BenchmarkReport) -> str:
        """Generate formatted Markdown report for the testbench suite."""
        lines = []
        lines.append("# TENSORGRAPH Compiler Validation & Benchmark Report")
        lines.append("")
        lines.append(f"**Execution Timestamp:** `{report.timestamp}`  ")
        lines.append(f"**Status:** {'✅ **FULL COMPLIANCE**' if report.passed_correctness_count == report.total_workloads else '⚠️ **PARTIAL VERIFICATION**'}  ")
        lines.append(f"**Total Workloads Evaluated:** `{report.total_workloads}`  ")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Numerical Equivalence Pass Rate:** `{report.passed_correctness_count} / {report.total_workloads}` ({ (report.passed_correctness_count / max(1, report.total_workloads))*100:.1f}%)")
        lines.append(f"- **Average Program Cost Reduction:** `{report.avg_cost_reduction_pct:.2f}%`")
        lines.append(f"- **Average AST / Diagram Node Reduction:** `{report.avg_node_reduction_pct:.2f}%`")
        lines.append(f"- **Average Saturation Engine Latency:** `{report.avg_saturation_time_ms:.2f} ms`")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Workload Breakdown")
        lines.append("")
        lines.append("| Workload Name | Category | Initial Cost | Extracted Cost | Cost Reduction | Saturation Time | Correctness |")
        lines.append("|---|---|---|---|---|---|---|")

        for r in report.results:
            status_icon = "✅ PASS" if r.correctness_passed else "❌ FAIL"
            lines.append(
                f"| `{r.workload_name}` | {r.category} | {r.cost_before:.1f} | {r.cost_after:.1f} | **{r.cost_reduction_pct:.1f}%** | {r.saturation_time_ms:.2f} ms | {status_icon} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Detailed Workload Specifications & Optimizations")
        lines.append("")

        for r in report.results:
            lines.append(f"### `{r.workload_name}` ({r.category})")
            lines.append(f"*{r.description}*")
            lines.append("")
            lines.append(f"- **AST Nodes:** `{r.nodes_before}` → `{r.nodes_after}` (Peak E-Nodes in E-Graph: `{r.peak_nodes}`)")
            lines.append(f"- **Iterations to Saturation:** `{r.iterations}`")
            lines.append(f"- **Recorded Proof Trace Entries:** `{r.trace_entries_count}`")
            lines.append(f"- **Extracted Program (Canonical IR):**")
            lines.append(f"```python")
            lines.append(f"{r.extracted_expr_pretty}")
            lines.append(f"```")
            lines.append("")

        lines.append("---")
        lines.append("*Grand Challenge Technologies — TENSORGRAPH Rewriting Compiler Verification Suite*")
        return "\n".join(lines)
