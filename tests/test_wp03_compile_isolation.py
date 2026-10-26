from __future__ import annotations

from tensorgraph.benchmarks.compile_isolation import isolated_graph_dtype_pairs


def test_torch_compile_state_is_reset_once_per_graph_family() -> None:
    events: list[str] = []

    def reset() -> None:
        events.append("reset")

    pairs = []
    for graph, dtype_name in isolated_graph_dtype_pairs(
        ["relu", "tanh", "exp"],
        ["float16", "float32"],
        reset,
    ):
        events.append(f"{graph}:{dtype_name}")
        pairs.append((graph, dtype_name))

    assert pairs == [
        ("relu", "float16"),
        ("relu", "float32"),
        ("tanh", "float16"),
        ("tanh", "float32"),
        ("exp", "float16"),
        ("exp", "float32"),
    ]
    assert events == [
        "reset",
        "relu:float16",
        "relu:float32",
        "reset",
        "tanh:float16",
        "tanh:float32",
        "reset",
        "exp:float16",
        "exp:float32",
    ]
