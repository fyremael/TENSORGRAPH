# GPU recovery package: portable nonlinear lowering and six-baseline evidence

**Package identifier:** `TG-GPU-WP01`  
**Parent recovery:** P0-P3 repository recovery  
**Status:** complete for the admitted commit and recorded Tesla T4 environment

This package contains two parallel obligations. Neither obligation promotes general compiler maturity.

## Admission record

The implementation was evaluated at commit `92bfa21538e60a4cc321f32f7340ba70eee00db0` and merged through PR #2 at merge commit `19fd6760d9b876c34880a79933c3e6914bf8fbf4`.

The authoritative machine-readable decision is `evidence/TG-GPU-WP01/ADMISSION.json`. Immutable artifact identities are recorded in `evidence/TG-GPU-WP01/SHA256SUMS`.

Admitted evidence:

- six-baseline JSON: `f0b4003f0f1250f4e4430a65897ef1bbbe8a6659f88fca9c74f2273538901c40`
- Sigmoid JSON: `e7fb0f2e7050e050d34857ee57b8547c306592625e4244b20ae4378531dd155a`
- Tanh JSON: `e9568d902d1dfd12e6816f28527c8a011e4c4e9e4ae14623a19b47cad9de5361`
- nonlinear evidence ZIP: `a62b75014f08207d4f60b2c20be4f340f747599e45a1bc286d1564c00ca593c8`

The raw evidence objects must remain byte-identical to these hashes. The source tree stores the admission record and identities; it does not treat a reformatted or summarized copy as equivalent evidence.

## Obligation A — version-portable Sigmoid and Tanh lowering

### Defect

The first Colab GPU replay reached the generated Triton kernel but Triton 3.6.0 rejected the generated `tl.sigmoid(value)` expression during JIT compilation. Tanh used the same intrinsic through the identity `2 * sigmoid(2x) - 1` and inherited the same blocker.

### Repair

The bounded Triton emitter now uses only `tl.exp` arithmetic:

```text
Sigmoid(x) = 1 / (1 + exp(-x))
Tanh(x)    = 2 / (1 + exp(-2x)) - 1
```

The generated source remains the exact source that is hashed, loaded, JIT compiled, and executed. The benchmark does not patch generated text or replace it with a handwritten substitute.

### Admission outcome

Both operations executed on the exact evaluated commit with a clean worktree, Tesla T4, PyTorch `2.11.0+cu128`, Triton `3.6.0`, and CUDA runtime `12.8`.

- Sigmoid: four workloads passed; maximum absolute error `1.7881393432617188e-07`.
- Tanh: four workloads passed; maximum absolute error `2.384185791015625e-07`.
- Both generated sources contain `tl.exp` and exclude `tl.sigmoid`.
- Each workload retains 100 raw timing samples for eager and generated execution.

## Obligation B — six-baseline comparative matrix

The earlier benchmark compared the unoptimized eager source graph against the rewritten and fused TENSORGRAPH graph. That result combined rewrite, fusion, launch-count, and backend effects. The expanded matrix separates those effects.

| Lane | Implementation | Graph |
|---|---|---|
| A | PyTorch eager | `ReLU -> ReLU -> Neg` |
| B | PyTorch eager | `ReLU -> Neg` |
| C | `torch.compile` | `ReLU -> ReLU -> Neg` |
| D | `torch.compile` | `ReLU -> Neg` |
| E | TENSORGRAPH generated Triton | extracted `ReLU -> Neg` |
| F | independent direct Triton reference | `ReLU -> Neg` |

The admitted matrix contains eight workloads: four sizes crossed with float32 and float16. All 48 lane-workload numerical gates passed exactly. The TENSORGRAPH-to-direct-Triton median latency ratio remained between `0.9937` and `1.0242`, establishing backend parity for this bounded kernel on the recorded environment.

## Interpretation contract

The matrix supports distinct statements:

- `A / B` estimates the benefit of removing the redundant ReLU in eager mode.
- `A / C` measures what `torch.compile` does with the original source graph.
- `B / D` measures `torch.compile` on the manually normalized graph.
- `A / E` is the end-to-end TENSORGRAPH compiler result.
- `B / E` compares TENSORGRAPH against an eager graph with the same semantics.
- `E / F` compares TENSORGRAPH code generation with an independent direct Triton implementation of the same normalized graph.

No single ratio establishes general compiler superiority. Results apply only to the recorded graph, dtype, tensor sizes, software stack, and GPU.

## Closed and open gates

Closed for `TG-GPU-WP01`:

- portable Sigmoid GPU execution
- portable Tanh GPU execution
- complete six-baseline execution and numerical agreement
- exact commit, clean-worktree, generated-source identity, environment, and raw-sample capture

Still open:

- general FX DAG compilation
- backward and training correctness
- cross-version and cross-hardware portability
- distributed production execution
- production readiness
