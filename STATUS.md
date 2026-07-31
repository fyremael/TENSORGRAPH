# TENSORGRAPH capability ledger

**Ledger version:** 4  
**Target package version:** `0.6.0.dev0`  
**Authority:** this file controls capability and maturity language used elsewhere in the repository.

Status values:

- `supported-research`: implemented and exercised by committed non-optional tests.
- `bounded-experimental`: implemented for a stated subset; broader behavior is not claimed.
- `experimental`: implementation exists but evidence is incomplete.
- `scaffold`: interface or partial implementation exists; execution is incomplete.
- `quarantined`: retained historical material that is not admissible evidence.
- `not-established`: no positive claim is permitted.

| Capability | Status | Scope | Promotion gate |
|---|---|---|---|
| Immutable typed expression IR | supported-research | `Id`, `Box`, `Seq`, `Par`, `Dup`, `Del`, `Swap`, `Case`, bounded `Iter` | Keep type-admission and normalization tests green. |
| Typed e-class merge discipline | supported-research | E-classes with identical `(domain, codomain)` sorts | Invariant checker and adversarial merge tests. |
| Equality saturation | supported-research | Bounded iterations and applications | Termination caps, deterministic tests, replayable trace. |
| Cost extraction | supported-research | Existing extractor cost model | Differential extraction tests. |
| Pure Cartesian structural naturality | bounded-experimental | `Dup` and `Del` only for morphisms not marked `effectful` | Effect-aware adversarial suite and formal semantic note. |
| Symmetric swap naturality | bounded-experimental | Typed `Par` followed by compatible `Swap` | Coherence tests. |
| FX unary elementwise import | bounded-experimental | Linear chains of supported unary modules/functions | Shape/dtype/device metadata and unsupported-node fail-closed tests. |
| Generated Triton unary execution | bounded-experimental | Contiguous floating CUDA tensors; bounded ReLU, Neg, portable-exp Sigmoid and Tanh paths | Preserve admitted same-commit evidence; extend across hardware, dtypes, graph forms, and backward execution. |
| Portable Sigmoid and Tanh lowering | bounded-experimental | Forward-only unary chains lowered through `tl.exp`; no `tl.sigmoid` dependency | TG-GPU-WP01 gate satisfied on the evaluated Tesla T4 stack; cross-version and cross-hardware evidence remains open. |
| Six-baseline GPU comparison | bounded-experimental | Source/optimized eager, source/optimized `torch.compile`, TENSORGRAPH, and direct Triton for `ReLU -> ReLU -> Neg` | TG-GPU-WP01 gate satisfied on the evaluated Tesla T4 stack; performance portability remains open. |
| Generated Sigmoid and Tanh input gradients | experimental | Terminal Sigmoid or Tanh, optionally preceded by one optimized ReLU; input gradient only | Admit TG-GPU-WP02 forward/backward evidence across the governed hardware, stack, dtype, regime, and direction matrix. |
| GPU portability matrix | experimental | TG-GPU-WP02 evidence runner and validator; no portability evidence admitted yet | T4 plus Ampere-or-newer, two exact PyTorch/Triton stacks, complete dtype dispositions, raw evidence, and independent admission. |
| General FX DAG import | experimental | Research representation only | Tensor metadata, aliasing, mutation, parameters, and round-trip execution. |
| Reduction code generation | experimental | Source-generation experiments | Correct neutral elements, multi-block algorithms, compiled differential tests. |
| CUDA C++ generation | experimental | Source-generation experiments | Compile, launch, sanitizer, and numerical tests. |
| Host-aware engine | experimental | Routing prototype | Route must invoke distinct compiled implementations and measure them. |
| Distributed saturation | scaffold | Local interfaces and partial transports | Complete workers, synchronization, failure handling, and multi-process tests. |
| Neural rule scheduling | experimental | Research scheduler | Controlled comparison against deterministic scheduling. |
| Historical benchmark reports | quarantined | Pre-recovery reports | Re-run under `docs/EVIDENCE_POLICY.md`; do not edit old numbers into compliance. |
| Production readiness | not-established | Entire repository | Independent review, supported release contract, reproducible CI and broader GPU evidence. |

## TG-GPU-WP01 evidence admission

The package is complete for evaluated commit `92bfa21538e60a4cc321f32f7340ba70eee00db0` on the recorded Tesla T4 environment. The code was merged through commit `19fd6760d9b876c34880a79933c3e6914bf8fbf4`.

The authoritative closure record is `evidence/TG-GPU-WP01/ADMISSION.json`. Artifact identities are pinned in `evidence/TG-GPU-WP01/SHA256SUMS`. The admission covers Sigmoid, Tanh, and the six-baseline comparison. It remains bounded to the recorded forward-only graphs, hardware, and software stack.

## TG-GPU-WP02 implementation status

The implementation package provides exact generated forward execution, generated Sigmoid and Tanh input-gradient kernels, a 60-cell per-environment matrix runner, a machine-readable evidence schema, a fail-closed validator, CPU contract tests, optional CUDA differential tests, and an interpretation charter.

No TG-GPU-WP02 CUDA portability or backward evidence is admitted by implementation alone. Promotion remains gated on raw same-commit artifacts from the governed hardware and software matrix and independent review.

## Claim boundary

No README, report, package metadata, generated dashboard, or release note may claim broader maturity than this ledger. A benchmark result does not promote a component by itself. Promotion requires code, tests, raw evidence, and review on the same commit.
