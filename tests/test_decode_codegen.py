import hashlib
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "decode_codegen", Path(__file__).parents[1] / "tensorgraph/codegen/decode.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
import pytest


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("hidden", [1024, 4096, 8192])
def test_artifact_is_deterministic_typed_and_hash_bound(dtype, hidden):
    graph = m.DecodeGraph(dtype, hidden)
    a = graph.emit()
    assert a == graph.emit()
    assert hashlib.sha256(a["source"].encode()).hexdigest() == a["source_sha256"]
    assert a["contract"]["outputs"] == [3, 8]
    assert graph.nodes[5].op == "mean" and graph.nodes[5].axis == "row"
    assert graph.nodes[3].args == (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 0},
        {"hidden": 16},
        {"hidden": 1024, "dtype": "int8"},
        {"hidden": 1024, "epsilon": 0},
        {"hidden": 1024, "epsilon": float("nan")},
    ],
)
def test_reject_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        m.DecodeGraph(**({"dtype": "float32"} | kwargs))


def test_modified_source_rejected_before_compiler():
    a = m.DecodeGraph("float32", 1024).emit()
    a["source"] += "alteration"
    # Validation occurs before load_inline; torch is optional for CPU suite.
    with pytest.raises(ValueError, match="hash"):
        m.compile_decode(a, ".")
