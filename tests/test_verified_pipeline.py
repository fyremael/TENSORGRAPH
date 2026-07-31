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
    assert "tl.sigmoid" not in artifact.generated_source
    assert "1.0 / (1.0 + tl.exp(-value))" in artifact.generated_source
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


@pytest.mark.parametrize(
    ("module", "expected_fragment"),
    [
        (torch.nn.Sigmoid(), "1.0 / (1.0 + tl.exp(-value))"),
        (torch.nn.Tanh(), "2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0"),
    ],
)
def test_sigmoid_and_tanh_use_version_portable_exp_lowering(
    module: torch.nn.Module,
    expected_fragment: str,
) -> None:
    artifact = compile_fx_elementwise(torch.nn.Sequential(module))

    assert expected_fragment in artifact.generated_source
    assert "tl.sigmoid" not in artifact.generated_source
    assert artifact.generated_source.count("tl.exp") == 1


def test_pipeline_rejects_parameterized_or_branching_graphs() -> None:
    with pytest.raises(ValueError, match="outside the verified pipeline|unsupported FX module"):
        compile_fx_elementwise(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()))

    class Branching(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + x

    with pytest.raises(
        ValueError,
        match="linear unary|unsupported FX function|unary operations|branching nodes",
    ):
        compile_fx_elementwise(Branching())


def test_generated_execution_fails_closed_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU fail-closed behavior is only applicable on a non-CUDA host")
    artifact = compile_fx_elementwise(torch.nn.Sequential(torch.nn.ReLU()))
    with pytest.raises(RuntimeError, match="CUDA is required"):
        load_generated_kernel(artifact)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("module", "rtol", "atol"),
    [
        (torch.nn.Sigmoid(), 1e-5, 1e-6),
        (torch.nn.Tanh(), 2e-5, 2e-6),
    ],
)
def test_portable_transcendental_lowerings_execute_on_gpu(
    module: torch.nn.Module,
    rtol: float,
    atol: float,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")

    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.ReLU(),
        module,
    ).cuda()
    artifact = compile_fx_elementwise(model)
    generated = load_generated_kernel(artifact)
    random_values = torch.randn(131_059, device="cuda", dtype=torch.float32) * 8.0
    edge_values = torch.tensor(
        [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0],
        device="cuda",
        dtype=torch.float32,
    )
    x = torch.cat((edge_values, random_values))

    expected = model(x)
    actual = generated.run(x)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
