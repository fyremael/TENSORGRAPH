import hashlib

import pytest

torch = pytest.importorskip("torch")

from tensorgraph.pipeline import compile_fx_elementwise, load_generated_kernel


def test_fx_ir_rewrite_extract_and_codegen_are_connected() -> None:
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.ReLU(),
        torch.nn.Sigmoid(),
    )

    artifact = compile_fx_elementwise(model)

    assert artifact.source_pretty.count("ReLU") == 2
    assert artifact.optimized_pretty.count("ReLU") == 1
    assert artifact.optimized_pretty.count("Sigmoid") == 1
    assert artifact.rewrite_summary["relu_idempotence"] >= 1
    assert artifact.generated_source.count("tl.where") == 1
    assert "tl.sigmoid" in artifact.generated_source
    assert artifact.source_sha256 == hashlib.sha256(
        artifact.generated_source.encode("utf-8")
    ).hexdigest()
    assert set(artifact.phase_ns) == {
        "fx_capture",
        "ir_construction",
        "saturation",
        "extraction",
        "source_generation",
    }
    assert all(value >= 0 for value in artifact.phase_ns.values())


def test_pipeline_rejects_parameterized_or_branching_graphs() -> None:
    with pytest.raises(ValueError, match="outside the verified pipeline|unsupported FX module"):
        compile_fx_elementwise(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()))

    class Branching(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + x

    with pytest.raises(ValueError, match="linear unary|unsupported FX function|unary operations"):
        compile_fx_elementwise(Branching())


def test_generated_execution_fails_closed_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU fail-closed behavior is only applicable on a non-CUDA host")
    artifact = compile_fx_elementwise(torch.nn.Sequential(torch.nn.ReLU()))
    with pytest.raises(RuntimeError, match="CUDA is required"):
        load_generated_kernel(artifact)


@pytest.mark.gpu
def test_generated_source_executes_and_matches_pytorch_on_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")

    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.ReLU(),
        torch.nn.Sigmoid(),
    ).cuda()
    artifact = compile_fx_elementwise(model)
    generated = load_generated_kernel(artifact)
    x = torch.randn(131_071, device="cuda", dtype=torch.float32)

    expected = model(x)
    actual = generated.run(x)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
