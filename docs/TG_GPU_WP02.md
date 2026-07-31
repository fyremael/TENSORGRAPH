# TG-GPU-WP02 — Portability and Training Semantics

**Status:** implementation package; no CUDA portability evidence admitted  
**Parent:** TG-GPU-WP01 closure at `61bec69cf248d96058ab5364a86de98b4e97b47e`  
**Issue:** `#4`

## Purpose

TG-GPU-WP02 determines whether the bounded generated Sigmoid and Tanh path remains numerically correct across hardware generations, floating-point formats, software stacks, numerical regimes, and input-gradient execution.

The package supplies an evidence-producing implementation. It does not itself establish portability. Promotion requires independent admission of raw CUDA artifacts from the complete governed matrix.

## Executable surface

The package adds:

- `tensorgraph.pipeline.compile_fx_elementwise_training`;
- exact generated forward execution through the existing verified pipeline;
- generated input-gradient kernels for Sigmoid and Tanh, optionally preceded by one optimized ReLU;
- `benchmarks/bench_portability_training.py`;
- `schemas/tg_gpu_wp02_evidence.schema.json`;
- `scripts/validate_wp02_evidence.py`;
- CPU contract tests and optional CUDA differential tests.

The training lowering is deliberately bounded. Accepted optimized operation sequences are:

```text
Sigmoid
Tanh
ReLU -> Sigmoid
ReLU -> Tanh
```

All other sequences fail closed.

## Exact-source contract

The forward source is the exact source produced, hashed, loaded, JIT-compiled, and executed by the verified elementwise pipeline. The backward source is generated from the optimized expression and is also hashed before loading.

The backward kernel consumes:

- the original input;
- the exact generated forward output;
- the upstream gradient.

It produces the input gradient. The evidence runner does not patch generated source and does not substitute a handwritten CUDA or Triton implementation.

## Evidence matrix

One raw artifact covers one exact hardware and software-stack environment. Its requested matrix is:

```text
dtype × operation × numerical_regime × direction
```

The full required values are:

- dtype: `float16`, `bfloat16`, `float32`;
- operation: `sigmoid`, `tanh`;
- numerical regime: `moderate`, `positive_saturation`, `negative_saturation`, `near_zero`, `mixed_edge`;
- direction: `forward`, `forward_backward`.

This is 60 requested cells per environment. Every cell must have one disposition:

- `passed`;
- `unsupported` with an explicit reason;
- `failed` with an explicit stage and reason.

The runner does not silently remove unsupported or failed cells. On pre-Ampere hardware, the default contract records `bfloat16` cells as unsupported rather than substituting another dtype.

## Numerical regimes

`moderate`
: Random values at ordinary scale.

`positive_saturation`
: Large positive finite values.

`negative_saturation`
: Large negative finite values.

`near_zero`
: Values scaled by the selected dtype epsilon.

`mixed_edge`
: Signed zero, finite extrema capped at a safe evaluation magnitude, tiny values, representative moderate and saturation values, positive and negative infinity, and NaN.

NaN and infinity classifications are compared explicitly. They are not discarded before summary construction. Relative error is reported, but absolute error remains an independent gate, especially near zero.

## Compilation and timing contract

The artifact separates:

- FX capture;
- IR construction;
- equality saturation;
- extraction;
- forward source generation;
- backward source generation;
- forward source load;
- backward source load;
- first forward execution and JIT specialization;
- first backward execution and JIT specialization;
- steady-state PyTorch timing;
- steady-state generated timing.

First execution and compilation costs must not be represented as steady-state kernel latency. Raw CUDA-event samples are retained for passed cells.

## Running one environment

Use an exact clean commit and an explicit stack identifier:

```bash
python benchmarks/bench_portability_training.py \
  --stack-id torch-2.11.0-cu128__triton-3.6.0 \
  --operations sigmoid tanh \
  --dtypes float16 bfloat16 float32 \
  --regimes moderate positive_saturation negative_saturation near_zero mixed_edge \
  --directions forward forward_backward \
  --size 65536 \
  --warmup 10 \
  --repetitions 50 \
  --block-size 256 \
  --output artifacts/gpu/wp02-t4-stack-a.json
```

The process writes the complete artifact before returning exit code `2` for rejected cells. `--allow-failed-cells` can be used for characterization runs where a nonzero result would interrupt orchestration. It does not convert failed cells into passed evidence.

Validate one or more artifacts:

```bash
python scripts/validate_wp02_evidence.py artifacts/gpu/wp02-t4-stack-a.json
```

Validate a proposed promotion bundle:

```bash
python scripts/validate_wp02_evidence.py \
  --promotion-bundle \
  artifacts/gpu/wp02-t4-stack-a.json \
  artifacts/gpu/wp02-ampere-stack-b.json
```

## Promotion gates

A promotion bundle must contain:

1. an explicit Tesla T4 artifact;
2. an Ampere-or-newer artifact;
3. at least two distinct exact PyTorch/Triton version pairs;
4. the full 60-cell request in each artifact;
5. no failed cells in the promoted bundle;
6. explicit unsupported dispositions where applicable;
7. exact generated forward and backward source identities;
8. raw timing samples and separate first-execution costs;
9. an independently reviewed machine-readable admission record.

Unsupported cells limit the resulting claim. For example, an unsupported T4 `bfloat16` cell does not block a bounded statement about supported T4 dtypes, but it blocks a claim that T4 `bfloat16` was validated.

## Interpretation contract

The evidence can support statements about numerical correctness and measured timing only for the exact admitted matrix cells. Cross-hardware comparisons must distinguish hardware, dtype, stack, operation, regime, and direction.

The package does not authorize claims of:

- general FX DAG compilation;
- arbitrary operator differentiation;
- parameter, optimizer, or end-to-end model-training correctness;
- distributed execution;
- non-NVIDIA portability;
- production readiness;
- performance portability outside the exact admitted matrix.
