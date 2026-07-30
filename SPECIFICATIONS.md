# TENSORGRAPH Specification — The Blueprint of Elegance

> *"Precision is the courtesy of the builder; elegance is the hallmark of the architect."*

This document serves as the formal specification for TENSORGRAPH—a toolkit designed to harmonize abstract category theory with industrial-grade compute. We provide here the rigorous blueprints necessary for the implementation and verification of our frontier engineering goals.

---

## 0. Executive Summary

TENSORGRAPH is a **diagrammatic rewriting compiler** in which:

1. **Systems** are **objects/types**.
2. **Processes between systems** are **typed diagrams** (1-morphisms).
3. **Transformations between processes** are **typed rewrite steps** (2-morphisms).
4. **Coherent compositional reasoning** is enforced by **normalization + canonical forms**.
5. **Equivalence “up to coherent isomorphism”** is implemented operationally as **rewrite theories + equality saturation**.
6. **Translation across interfaces** is implemented as **adjunctions + mates**, i.e., compiler passes that transport rewrite laws between IR layers.

The core optimization engine is an **e-graph**. It compactly represents many equivalent programs and can extract the lowest-cost equivalent program under a configurable cost model.

---

## 1. Goals and Non-Goals

### 1.1 Goals

TENSORGRAPH SHALL provide:

1. A typed intermediate representation (IR) for programs expressed as **string diagrams** in at least a strict monoidal setting.
2. A rewrite rule system representing **2-morphisms** as typed equalities between diagrams.
3. An e-graph engine supporting **equality saturation** over the diagram IR.
4. A cost model and extraction engine that selects the **best equivalent program**.
5. Deterministic normalization/canonicalization rules (“coherence discipline”) to keep search stable and reduce bureaucratic variance.
6. A backend integration starting with `torch.fx` to demonstrate real graph optimization and round-tripping.
7. Proof/audit trace capability sufficient to replay or inspect the transformation path producing the optimized output.

### 1.2 Non-Goals (Initial Version)

TENSORGRAPH WILL NOT initially aim to:

1. Provide a full general-purpose theorem prover for higher category theory.
2. Provide complete support for arbitrary cyclic graphs, recursion, or control-flow (this may be phased).
3. Optimize arbitrary `torch.fx` graphs beyond the declared supported subset without explicit roadmap implementation.
4. Provide verified correctness proofs in a proof assistant as a hard requirement (though traceability is required).

---

## 2. Glossary and Terms

- **Object (Obj):** A type representing a system interface (e.g., `Tensor`, `KeyValueCache`, `AudioState`).
- **1-morphism (Diagram / Expr):** A typed process `A → B`.
- **2-morphism (Rewrite):** A typed equivalence witness transforming one diagram into another.
- **Interface:** A tuple/list of objects `(A1, A2, …)` representing multiple ports.
- **Generator / Box:** A primitive operation with declared domain and codomain interfaces.
- **Sequential composition:** `f ; g` connecting codomain of `f` to domain of `g`.
- **Parallel composition:** `f ⊗ g` representing independent side-by-side execution.
- **Coherence discipline:** Normalization rules ensuring diagrams have canonical forms to reduce redundant variations.
- **E-graph:** A data structure representing equivalence classes of expressions with efficient unioning and rewrite application.
- **Saturation:** The process of repeatedly applying rewrites until a fixed point (or iteration limit) is reached.
- **Extraction:** Selecting a single concrete expression from an e-class according to a cost function.
- **Adjunction:** A structured interface translation pair `(f ⊣ g)` used to transport rewrites (“mates”).

---

## 3. System Requirements

### 3.1 Functional Requirements

**FR-1 (Typed IR):**  
The system SHALL represent diagrams with explicit domain and codomain types and reject ill-typed compositions.

**FR-2 (Composition):**  
The system SHALL support at minimum:
- Identity morphisms.
- Sequential composition.
- Parallel (tensor) composition.

**FR-3 (Rewrite Rules):**  
The system SHALL support user-defined rewrite rules with:
- Typed patterns on the left-hand side (LHS).
- Typed replacements on the right-hand side (RHS).
- Optional side conditions.
- Bidirectional equalities.

**FR-4 (Equality Saturation):**  
The system SHALL implement an e-graph engine with:
- Hashconsing of nodes.
- Union-find equivalence management.
- Rebuild/repair after unions.
- Iterative rewrite scheduling.

**FR-5 (Cost-based Extraction):**  
The system SHALL extract best expressions based on a user-defined cost function, defaulting to a stable baseline.

**FR-6 (Backend Integration):**  
The system SHALL include a backend integration with `torch.fx` demonstrating:
- Import of a supported FX subset into TENSORGRAPH IR.
- Optimization via saturation.
- Export of optimized result back to a runnable PyTorch module.

**FR-7 (Proof/Audit Trace):**  
The system SHALL optionally record rewrite applications and unions.

**FR-8 (Extensibility):**  
The system SHALL support external rewrite libraries as importable modules without editing core engine code.

### 3.2 Non-Functional Requirements

**NFR-1 (Determinism):** deterministic results under fixed scheduling and cost model.

**NFR-2 (Robustness):** bounded iteration limits and typed guards.

**NFR-3 (Performance):** interactive-time behavior for typical IR sizes.

**NFR-4 (Ergonomics):** clear diagnostics.

---

## 4. Formal Model

### 4.1 Object Language

#### 4.1.1 Objects

An **object** is either:
1. An atomic object `Obj(name)`, or
2. A tensor product `ObjTensor(left, right)`.

Object equality SHALL be structural.

#### 4.1.2 Interfaces

An **interface** is a finite ordered tuple of objects:  
`I = (A1, A2, ..., An)`

---

### 4.2 Diagram Language (1-Morphisms)

A diagram expression `Expr` denotes a typed morphism `dom(Expr) → cod(Expr)`.

#### 4.2.1 Core Constructors

TENSORGRAPH SHALL implement:

1. **Identity**: `Id(A)` with type `A → A`.
2. **Generator**: `Box(op, attrs)` with type `dom(op) → cod(op)`.
3. **Sequential composition**: `Seq(f, g)` with type `dom(f) → cod(g)` if `cod(f) = dom(g)`.
4. **Parallel composition**: `Par(f, g)` with type `dom(f) ⊗ dom(g) → cod(f) ⊗ cod(g)`.

#### 4.2.2 Typing Rules

Let `Γ` be a signature mapping ops to their domains/codomains.

- `Γ ⊢ Id(A) : A → A`
- `Γ ⊢ Box(op) : dom(op) → cod(op)`
- If `Γ ⊢ f : A → B` and `Γ ⊢ g : B → C`, then `Γ ⊢ Seq(f,g) : A → C`
- If `Γ ⊢ f : A → B` and `Γ ⊢ g : C → D`, then `Γ ⊢ Par(f,g) : (A ⊗ C) → (B ⊗ D)`

Typechecking SHALL be strict.

---

### 4.3 Rewrite Language (2-Morphisms)

A rewrite rule is a typed equality between diagrams:
`Rule: LHS ≡ RHS`

#### 4.3.1 Pattern Syntax

Patterns mirror expressions but allow variables:

- `PVar(name, dom?, cod?)`
- `PBox(op, attrs?)`
- `PSeq(p1, p2)`
- `PPar(p1, p2)`
- `PId(ObjVar)`

#### 4.3.2 Match Environment

A successful match yields:
- `env_expr: Dict[str, EClassId]`
- `env_obj: Dict[str, Obj]`

#### 4.3.3 Side Conditions

Rules MAY include guards over envs and attrs.

---

## 5. Architecture and Module Layout

The package SHALL follow:

```
tensorgraph/
  __init__.py
  types.py
  signature.py
  ir/
    expr.py
    normalize.py
    pretty.py
  rewrite/
    pattern.py
    rule.py
    ematch.py
  egraph/
    enode.py
    unionfind.py
    egraph.py
    saturation.py
    extract.py
    trace.py
  backends/
    fx_import.py
    fx_export.py
  cli/
    optimize_fx.py
  tests/
```

---

## 6. Core Data Structures

Normative definitions for `Obj`, `Signature`, `Expr`, `ENode`, `EClass`, `UnionFind`, and invariants MUST be implemented.

---

## 7. Algorithms

### 7.1 Normalization

Normalization SHALL:
- remove identity compositions
- right-associate sequential composition

### 7.2 E-Graph Operations

`add_expr`, `merge`, and `rebuild` MUST maintain congruence and type sorts.

### 7.3 Saturation

`saturate(egraph, rewrites, iters)` SHALL apply rewrites until fixed point or limits.

### 7.4 Extraction

Extraction SHALL compute best representatives by a cost model.

---

## 8. Proof / Audit Tracing

Tracing SHALL record:
- rule name
- match env
- merged class ids

---

## 9. torch.fx Backend

Initial supported subset:
- `placeholder → call_module* → output`

Import/export SHALL preserve numerical behavior under the rewrite laws applied.

---

## 10. Adjunctions and Mates

Given a commuting equality of the form:

```
f ; u  ≡  v ; f
```

The operational mate rewrite SHALL be synthesized as:

```
u  ≡  f ; v ; g
```

(where `f : A→B` and `g : B→A`).

---

## 11. Acceptance Criteria

A build is complete when:
- unit tests pass
- FX demo round-trips and validates outputs
- saturation reduces cost in the LoRA fusion case

---
---

## 12. v0.4.0 Additions (Kernel Release)

### 12.1 Dynamic Control Flow
- **Pattern Matching**: `PIter` added with support for capturing data attributes (e.g., `count`).
- **Rewrite Rules**: `peel_iter`, `split`, `iter_fusion`, `iter_product` implemented in `tensorgraph.library.control_flow`.
- **Algebra**: `right_unit`, `left_unit`, `assoc` rules added to ensure convergence with normalization.

### 12.2 Heterogeneous Sharding
- **Architecture**: `Shard` class implemented with `Partition` (Owned/Ghost nodes).
- **Fabric**: `MockFabric` implemented to simulate inter-shard communication.
- **Protocol**: `on_merge` callback in `EGraph` hooks into `Shard.receive_merge` to propagate equality updates.

### 12.3 Automated Kernel Fusion
- **Codegen**: `TritonEmitter` class implemented in `tensorgraph.codegen.triton`.
- **Traits**: `OpDef` extended to support traits (`elementwise`, `reduction`).
- **Composition**: `Seq` and `Par` supported for implementation of fused kernels.

---

## 13. v0.5.0 Additions (Level Up Release)

### 13.1 Performance Regression Suite
- **Harness**: `tensorgraph.benchmarks.regression` implemented with automated timing and stats collection.
- **Baselines**: Support for saving and loading `baseline.json` for historical comparison.
- **CI Mode**: `--ci` flag enforced for automated regression detection (default >10% threshold).

### 13.2 Reduction Kernels
- **Operators**: `Sum`, `Mean`, `Max`, `Min` implemented in `TritonEmitter`.
- **Composites**: Numerically stable `Softmax` generator using max-shift technique.
- **Analysis**: `_has_reduction` detection for switching between elementwise and reduction kernel templates.

### 13.3 Production Fabric
- **Protocol**: Formal `Fabric` protocol defined in `tensorgraph.dist.fabric`.
- **Async Implementation**: `AsyncFabric` with background dispatcher and worker threads, message batching, and statistics.
- **Transport**: Abstracted message passing supporting both in-process and multi-process (TcpFabric) models.

### 13.4 CUDA Backend
- **Codegen**: `CUDAEmitter` implemented for raw CUDA C++ generation.
- **Feature Parity**: Full parity with TritonEmitter for elementwise and reduction operations.
- **Host Launchers**: Automatic generation of C++ host code and `cudaStream_t` compliant launchers.
