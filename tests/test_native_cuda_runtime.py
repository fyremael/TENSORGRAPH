from __future__ import annotations

import hashlib

import pytest

from tensorgraph.codegen.native_cuda import NativeCUDAEmitter
from tensorgraph.ir import Box, Par, Seq
from tensorgraph.signature import Signature
from tensorgraph.types import Obj


def _signature() -> Signature:
    tensor = Obj("Tensor")
    signature = Signature()
    for operation in ("ReLU", "Neg", "Sigmoid", "Tanh", "Exp", "Log"):
        signature.add(operation, tensor, tensor, traits={"elementwise"})
    return signature


@pytest.mark.parametrize(
    ("dtype", "storage_fragment", "load_fragment", "store_fragment"),
    [
        ("float32", "using tensorgraph_storage_t = float", "float value = x[idx]", "y[idx] = value"),
        (
            "float16",
            "using tensorgraph_storage_t = __half",
            "__half2float(x[idx])",
            "__float2half_rn(value)",
        ),
        (
            "bfloat16",
            "using tensorgraph_storage_t = __nv_bfloat16",
            "__bfloat162float(x[idx])",
            "__float2bfloat16_rn(value)",
        ),
    ],
)
def test_native_cuda_emitter_produces_typed_exact_source(
    dtype: str,
    storage_fragment: str,
    load_fragment: str,
    store_fragment: str,
) -> None:
    emitter = NativeCUDAEmitter(_signature())
    expression = Seq(Box("ReLU"), Seq(Box("Neg"), Box("Sigmoid")))
    artifact = emitter.emit_artifact(expression, dtype=dtype)  # type: ignore[arg-type]

    assert artifact.operations == ("ReLU", "Neg", "Sigmoid")
    assert storage_fragment in artifact.generated_source
    assert load_fragment in artifact.generated_source
    assert store_fragment in artifact.generated_source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in artifact.generated_source
    assert "getCurrentCUDAStream" in artifact.generated_source
    assert "isnan(value) ? value" in artifact.generated_source
    assert artifact.source_sha256 == hashlib.sha256(
        artifact.generated_source.encode("utf-8")
    ).hexdigest()


def test_native_cuda_emitter_fails_closed_for_parallel_or_unknown_graphs() -> None:
    emitter = NativeCUDAEmitter(_signature())
    with pytest.raises(ValueError, match="only Id, Box, and Seq"):
        emitter.emit_artifact(Par(Box("ReLU"), Box("Neg")))
    with pytest.raises(ValueError, match="unsupported native CUDA unary operation"):
        emitter.emit_artifact(Box("Softmax"))


def test_native_cuda_log_requires_provable_strict_positive_domain() -> None:
    emitter = NativeCUDAEmitter(_signature())

    with pytest.raises(ValueError, match="Log requires"):
        emitter.emit_artifact(Box("Log"))
    with pytest.raises(ValueError, match="Log requires"):
        emitter.emit_artifact(Seq(Box("ReLU"), Box("Log")))
    with pytest.raises(ValueError, match="Log requires"):
        emitter.emit_artifact(Seq(Box("Neg"), Box("Log")), log_domain="strict_positive")

    direct = emitter.emit_artifact(Box("Log"), log_domain="strict_positive")
    generated_positive = emitter.emit_artifact(Seq(Box("Sigmoid"), Box("Log")))
    assert direct.log_domain == "strict_positive"
    assert generated_positive.operations == ("Sigmoid", "Log")


def test_native_cuda_compile_fails_closed_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from tensorgraph.runtime.native_cuda import compile_native_cuda

    artifact = NativeCUDAEmitter(_signature()).emit_artifact(Box("ReLU"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        compile_native_cuda(artifact)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_native_cuda_ordinary_and_graph_replay_match_changed_inputs(dtype_name: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if getattr(torch.utils.cpp_extension, "CUDA_HOME", None) is None:
        pytest.skip("CUDA toolkit is unavailable")

    from tensorgraph.runtime.native_cuda import compile_native_cuda

    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    expression = Seq(Box("ReLU"), Seq(Box("Neg"), Box("Sigmoid")))
    artifact = NativeCUDAEmitter(_signature()).emit_artifact(
        expression,
        dtype=dtype_name,  # type: ignore[arg-type]
    )
    executable = compile_native_cuda(artifact)

    x0 = torch.randn(65_537, device="cuda", dtype=dtype).contiguous()
    expected0 = torch.sigmoid(-torch.relu(x0))
    actual0 = executable.run(x0)
    torch.testing.assert_close(actual0, expected0, rtol=5e-3, atol=5e-4, equal_nan=True)

    captured = executable.capture(x0)
    replay0 = captured.replay(x0, clone_output=True)
    torch.testing.assert_close(replay0, expected0, rtol=5e-3, atol=5e-4, equal_nan=True)

    x1 = (torch.randn_like(x0) * 4.0).contiguous()
    expected1 = torch.sigmoid(-torch.relu(x1))
    replay1 = captured.replay(x1, clone_output=True)
    torch.testing.assert_close(replay1, expected1, rtol=5e-3, atol=5e-4, equal_nan=True)
    assert not torch.equal(replay0, replay1)


def test_native_cuda_run_out_rejects_alias_before_extension_call() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from tensorgraph.runtime.native_cuda import NativeCUDAExecutable

    artifact = NativeCUDAEmitter(_signature()).emit_artifact(Box("ReLU"))

    class NeverCalled:
        def tensorgraph_native_run_out(self, _x: object, _y: object) -> None:
            raise AssertionError("extension must not be called for invalid aliasing")

    executable = NativeCUDAExecutable(
        artifact=artifact,
        module=NeverCalled(),
        phase_ns={},
        compiler_identity={},
    )
    x = torch.randn(8, device="cuda")
    with pytest.raises(ValueError, match="must not alias"):
        executable.run_out(x, x)
