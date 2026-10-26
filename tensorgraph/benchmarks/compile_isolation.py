"""Isolation helpers for compiler comparison benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence


def isolated_graph_dtype_pairs(
    graphs: Sequence[str],
    dtypes: Sequence[str],
    reset_compiler_state: Callable[[], None],
) -> Iterator[tuple[str, str]]:
    """Yield graph/dtype pairs after resetting state at each graph boundary.

    ``torch.compile`` keeps process-global cache and recompile counters. A
    comparison matrix intentionally compiles structurally different graph
    families, so allowing those counters to leak between families can make a
    later baseline fail solely because earlier families consumed the budget.
    """
    for graph in graphs:
        reset_compiler_state()
        for dtype_name in dtypes:
            yield graph, dtype_name
