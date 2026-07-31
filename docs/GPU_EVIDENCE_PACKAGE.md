# GPU recovery package: portable nonlinear lowering and six-baseline evidence

**Package identifier:** `TG-GPU-WP01`  
**Parent recovery:** P0-P3 repository recovery  
**Status:** executable; promotion requires recorded CUDA evidence on the exact reviewed commit

This package contains two parallel obligations. Neither obligation promotes general
compiler maturity.

## Obligation A — version-portable Sigmoid and Tanh lowering

### Defect

The first Colab GPU replay reached the generated Triton kernel but Triton 3.6.0
rejected the generated `tl.sigmoid(value)` expression during JIT compilation.
Tanh used the same intrinsic through the identity
`2 * sigmoid(2x) - 1` and inherited the same blocker.

### Repair

The bounded Triton emitter now uses only `tl.exp` arithmetic:

```text
Sigmoid(x) = 1 / (1 + exp(-x))
Tanh(x)    = 2 / (1 + exp(-2x)) - 1
```

The generated source remains the exact source that is hashed, loaded, JIT
compiled, and executed. The benchmark does not patch generated text or replace it
with a handwritten substitute.

### Required checks

1. Unit tests must prove that generated Sigmoid and Tanh source contains the
   portable `tl.exp` lowering and does not contain `tl.sigmoid`.
2. Optional GPU tests must run random and edge-valued float32 inputs for both
   operations and compare against PyTorch.
3. GPU evidence must record the exact commit, clean worktree, generated-source
   SHA-256, environment, raw samples, and numerical error.
4. A failure in either operation is a failed obligation. Evidence from Neg does
   not substitute for Sigmoid or Tanh evidence.

Run the two evidence replays on a clean CUDA/Triton host:

```bash
python benchmarks/bench_verified_elementwise.py \
  --terminal-op sigmoid \
  --output artifacts/gpu/sigmoid.json

python benchmarks/bench_verified_elementwise.py \
  --terminal-op tanh \
  --output artifacts/gpu/tanh.json
```

## Obligation B — six-baseline comparative matrix

The earlier benchmark compared the unoptimized eager source graph against the
rewritten and fused TENSORGRAPH graph. That result combined rewrite, fusion,
launch-count, and backend effects. The expanded matrix separates those effects.

| Lane | Implementation | Graph |
|---|---|---|
| A | PyTorch eager | `ReLU -> ReLU -> Neg` |
| B | PyTorch eager | `ReLU -> Neg` |
| C | `torch.compile` | `ReLU -> ReLU -> Neg` |
| D | `torch.compile` | `ReLU -> Neg` |
| E | TENSORGRAPH generated Triton | extracted `ReLU -> Neg` |
| F | independent direct Triton reference | `ReLU -> Neg` |

The benchmark records all six outputs, numerical gates, first-execution times,
raw steady-state CUDA-event samples, summaries, and source identities. Timing
order is shuffled for every repetition to reduce monotonic drift bias.

Run:

```bash
python benchmarks/bench_six_baseline_elementwise.py \
  --sizes 1024 65536 1048576 4194304 \
  --dtypes float32 float16 \
  --warmup 25 \
  --repetitions 100 \
  --block-size 256 \
  --output artifacts/gpu/six_baseline.json
```

A CSV summary is written next to the JSON evidence unless
`--summary-output` is supplied.

## Interpretation contract

The matrix supports distinct statements:

- `A / B` estimates the benefit of removing the redundant ReLU in eager mode.
- `A / C` measures what `torch.compile` does with the original source graph.
- `B / D` measures `torch.compile` on the manually normalized graph.
- `A / E` is the end-to-end TENSORGRAPH compiler result.
- `B / E` compares TENSORGRAPH against an eager graph with the same semantics.
- `E / F` compares TENSORGRAPH code generation with an independent direct Triton
  implementation of the same normalized graph.

No single ratio establishes general compiler superiority. Results apply only to
the recorded graph, dtype, tensor sizes, software stack, and GPU.

## Promotion gates

Obligation A may be promoted when both Sigmoid and Tanh evidence files pass on
the same reviewed commit.

Obligation B may be promoted when all six lanes execute and agree numerically,
the worktree is clean, raw samples are retained, and no lane is silently omitted
or replaced.

Production readiness, general FX DAG compilation, backward correctness,
distributed saturation, and performance portability remain outside this package.
