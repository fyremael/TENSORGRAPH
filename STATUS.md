# TENSORGRAPH capability ledger

**Ledger version:** 1  
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
| Generated Triton unary execution | bounded-experimental | Contiguous floating tensors on CUDA; supported unary chains | GPU CI evidence, numerical tolerance, generated-source identity. |
| General FX DAG import | experimental | Research representation only | Tensor metadata, aliasing, mutation, parameters, and round-trip execution. |
| Reduction code generation | experimental | Source-generation experiments | Correct neutral elements, multi-block algorithms, compiled differential tests. |
| CUDA C++ generation | experimental | Source-generation experiments | Compile, launch, sanitizer, and numerical tests. |
| Host-aware engine | experimental | Routing prototype | Route must invoke distinct compiled implementations and measure them. |
| Distributed saturation | scaffold | Local interfaces and partial transports | Complete workers, synchronization, failure handling, and multi-process tests. |
| Neural rule scheduling | experimental | Research scheduler | Controlled comparison against deterministic scheduling. |
| Historical benchmark reports | quarantined | Pre-recovery reports | Re-run under `docs/EVIDENCE_POLICY.md`; do not edit old numbers into compliance. |
| Production readiness | not-established | Entire repository | Independent review, supported release contract, reproducible CI and GPU evidence. |

## Claim boundary

No README, report, package metadata, generated dashboard, or release note may claim broader maturity than this ledger. A benchmark result does not promote a component by itself. Promotion requires code, tests, raw evidence, and review on the same commit.
