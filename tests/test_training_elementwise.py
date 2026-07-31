from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from tensorgraph.pipeline import (
    compile_fx_elementwise_training,
    load_generated_backward_kernel,
    load_generated_kernel,
)


@pytest.mark.parametrize(
    ("module", "terminal", "derivative"),
    [
        (torch.nn.Sigmoid(), "Sigmoid", "derivative = y * (1.0 - y)"),
        (torch.nn.Tanh(), "Tanh", "derivative = 1.0 - y * y"),
    ],
)
def test_training_codegen_preserves_forward_and_generates_input_gradient(
    module: torch.nn.Module,
    terminal: str,
    derivative: str,
) -> None:
    artifact = compile_fx_elementwise_training(torch.nn.Sequential(module))

    assert artifact.optimized_ops == (terminal,)
    assert artifact.terminal_op == terminal
    assert "tl.exp" in artifact.forward.generated_source
    assert "tl.sigmoid" not in artifact.forward.generated_source
    assert derivative in artifact.generated_backward_source
    assert "grad_output" in artifact.generated_backward_source
    assert "tl.sigmoid" not in artifact.generated_backward_source
    assert artifact.backward_source_sha256 == hashlib.sha256(
        artifact.generated_backward_source.encode("utf-8")
    ).hexdigest()
    assert artifact.backward_generation_ns >= 0


@pytest.mark.parametrize("module", [torch.nn.Sigmoid(), torch.nn.Tanh()])
def test_training_codegen_accepts_one_optimized_relu_prefix(module: torch.nn.Module) -> None:
    artifact = compile_fx_elementwise_training(
        torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ReLU(), module)
    )

    assert artifact.optimized_ops[0] == "ReLU"
    assert artifact.optimized_ops.count("ReLU") == 1
    assert "tl.where(x > 0.0, derivative, 0.0)" in artifact.generated_backward_source


def test_training_codegen_rejects_unsupported_compositions() -> None:
    with pytest.raises(ValueError, match="bounded training lowering"):
        compile_fx_elementwise_training(
            torch.nn.Sequential(torch.nn.Sigmoid(), torch.nn.Tanh())
        )


def test_generated_backward_fails_closed_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU fail-closed behavior applies only without CUDA")
    artifact = compile_fx_elementwise_training(torch.nn.Sequential(torch.nn.Sigmoid()))
    with pytest.raises(RuntimeError, match="CUDA is required"):
        load_generated_backward_kernel(artifact)


@pytest.mark.gpu
@pytest.mark.parametrize("module", [torch.nn.Sigmoid(), torch.nn.Tanh()])
@pytest.mark.parametrize("with_relu", [False, True])
def test_generated_forward_backward_matches_pytorch_on_gpu(
    module: torch.nn.Module,
    with_relu: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")

    modules: list[torch.nn.Module] = []
    if with_relu:
        modules.extend((torch.nn.ReLU(), torch.nn.ReLU()))
    modules.append(module)
    model = torch.nn.Sequential(*modules).cuda().eval()
    artifact = compile_fx_elementwise_training(model)
    generated_forward = load_generated_kernel(artifact.forward)
    generated_backward = load_generated_backward_kernel(artifact)

    edge = torch.tensor(
        [
            -float("inf"),
            -30.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            30.0,
            float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=torch.float32,
    )
    random = torch.randn(65_537, device="cuda", dtype=torch.float32) * 8.0
    x = torch.cat((edge, random)).contiguous()
    grad_output = torch.randn_like(x)

    candidate_y = generated_forward.run(x)
    candidate_grad = generated_backward.run(x, candidate_y, grad_output)

    reference_x = x.detach().clone().requires_grad_(True)
    reference_y = model(reference_x)
    reference_y.backward(grad_output)
    assert reference_x.grad is not None
    torch.cuda.synchronize()

    torch.testing.assert_close(candidate_y, reference_y, rtol=2e-5, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(
        candidate_grad,
        reference_x.grad,
        rtol=5e-5,
        atol=5e-6,
        equal_nan=True,
    )
