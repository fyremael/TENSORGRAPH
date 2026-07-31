# TG-GPU-WP03 — Native CUDA low-latency inference lane

## Purpose

TG-GPU-WP03 restores the pre-remediation TENSORGRAPH-to-native-CUDA inference
path as an executable and evidence-governed backend. It is independent from
TG-GPU-WP02, which covers generated Triton forward and backward semantics.

The bounded executable sequence is:

```text
TENSORGRAPH unary expression
→ NativeCUDAEmitter
→ exact CUDA extension source and SHA-256
→ compile and load
→ ordinary launch
→ CUDA Graph capture and replay
→ numerical differential checks
→ raw latency evidence
```

The package does not establish complete transformer next-token generation.

## First-target graph contract

The admitted source language is a single contiguous tensor passed through a
linear sequence of these pure unary operations:

- `ReLU`
- `Neg`
- `Sigmoid`
- `Tanh`
- `Exp`
- `Log`

`Par`, branching, parameters, mutation, reductions, multiple inputs, and
unsupported boxes fail closed. `Log` requires a strict-positive input contract
or a preceding operation such as `Sigmoid` or `Exp` that proves strict
positivity at the Log site.

The first dtype set is `float16`, `bfloat16`, and `float32`. Computation is
performed in scalar `float`; typed storage conversion occurs at the kernel
boundary. Pre-Ampere `bfloat16` is explicitly unsupported.

## ABI contract

Every executable artifact fixes:

- one input and one output;
- identical shapes and dtypes;
- contiguous layout;
- no input/output aliasing;
- the current PyTorch CUDA stream;
- a 256-thread launch block;
- dynamic extent equal to tensor `numel`;
- stable input and output addresses after graph capture.

The runtime rejects CPU tensors, non-contiguous tensors, dtype or device
mismatch, shape mismatch, aliasing, missing CUDA toolchains, stale source hashes,
and unsupported hardware.

## Exact-source rule

`NativeCUDAArtifact.generated_source` is the complete CUDA translation unit
compiled by `torch.utils.cpp_extension.load_inline`. The source SHA-256 is
recomputed before compilation. No handwritten replacement kernel, source patch,
or eager fallback is permitted.

The PyTorch C++ declaration is runtime scaffolding only. The kernel body and its
operation sequence are generated from the TENSORGRAPH expression by the
`NativeCUDAEmitter`, which specializes the historical `CUDAEmitter` lineage.

## CUDA Graph contract

`NativeCUDAGraph` allocates static input and output tensors, warms the loaded
native kernel on a non-default stream, captures one `run_out` launch, and retains
the graph object and stable buffers.

Replay permits changed input contents. It does not permit changed shape, dtype,
device, layout, or captured addresses. The evidence runner verifies a second,
different input against an independent PyTorch result before admitting a cell.

## Timing interpretation

The runner records these categories separately:

- native compile and module load;
- CUDA Graph capture and instantiation;
- warmed ordinary native launch;
- warmed graph replay without input copy;
- input copy plus graph replay;
- eager PyTorch;
- `torch.compile`/Inductor where supported;
- TENSORGRAPH-generated Triton;
- independent direct Triton.

Every steady-state timing retains raw CUDA-event samples. Compilation, first use,
and capture costs must not be presented as steady-state launch latency.

The independent direct-native baseline remains a required follow-up before any
strong emitter-quality or backend-optimality claim. Its absence does not permit a
substitute or estimated value.

## Evidence dispositions

Each requested cell has one disposition:

- `passed`: exact generated native CUDA compiled, executed, matched the numerical
  reference, passed changed-input graph replay, and retained raw timing samples;
- `unsupported`: the exact hardware, dtype, compiler, or baseline combination is
  outside the bounded contract and has an explicit reason;
- `failed`: compilation, loading, launch, capture, replay, numerical, or evidence
  validation failed.

A raw artifact always has `promotion_claim: false`.

## Promotion boundary

TG-GPU-WP03 first-target promotion requires independently reviewed evidence on a
T4-class GPU and an Ampere-or-newer GPU, explicit dtype dispositions, ordinary
and graph-replay correctness, changed-input replay, raw timing samples, exact
source identities, and independently executed baselines.

Promotion establishes only the bounded unary native-CUDA inference subgraph.
GEMV, normalization, RoPE, KV-cache mutation, attention, logits processing,
sampling, and integrated transformer-block replay remain separate follow-on
lanes.
