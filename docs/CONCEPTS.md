# TENSORGRAPH Concepts — The Mental Model

> *"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."*  
> — Edsger W. Dijkstra

Welcome. This guide is designed to cultivate the mental model required to master TENSORGRAPH. We respect your time and intellectual curiosity—offering a path that is simultaneously rigorous, elegant, and profoundly practical.

---

## Table of Contents

1. [The Core Insight](#the-core-insight)
2. [The Art of the Cathedral Builder](#intuition-pump-the-cathedral-builder)
3. [Objects: The Language of Systems](#objects-the-types-of-systems)
4. [Morphisms: The Fluidity of Process](#morphisms-processes-between-systems)
5. [The E-Graph: A Symphony of Programs](#the-e-graph-many-programs-one-structure)
6. [Saturation & The Extraction of Best](#saturation--extraction)
7. [The Philosophy of Why](#why-this-matters)

---

## The Core Insight

**Programs are living diagrams. Optimization is the discovery of their most harmonious form.**

Traditional compilers perceive programs as static trees of instruction. TENSORGRAPH evolves this perspective, representing computation as **typed string diagrams**—a powerful categorical formalism where composition is explicit and optimization becomes an algebraic dance.

```
Traditional:  optimize(parse(source)) → new_tree
TENSORGRAPH:     saturate(diagram) → extract_best(equivalent_forms)
```

The breakthrough lies in a fundamental shift: instead of a step-by-step transformation, we **visualize all equivalent programs simultaneously** within an e-graph, then **distill the most elegant implementation**.

---

## The Art of the Cathedral Builder

Imagine you are the architect of a grand cathedral. Your vision is composed of:

- **Materials** (stone, glass, wood) — these are our **Objects**.
- **Craftsmanship** (carve, assemble, install) — these are our **Boxes**.
- **Sequence** (the flow of work from foundation to spire) — this is **Seq**.
- **Parallelism** (crews working in unison across different wings) — this is **Par**.

In the course of your work, you might discover that a specific sequence of "carve then polish" achieves the same structural integrity as "polish then carve" for a particular stone. This realization is a **Rewrite Rule**.

An **e-graph** is like possessing every possible blueprint for your cathedral at once, layered with perfect transparency. Each layer is a different path to the same magnificent result. When an equivalence is discovered, you don't discard the old design; you merge the blueprints, recognizing they lead to the same destination.

**Saturation** is the exhaustive exploration of these possibilities. **Extraction** is the final, pragmatic choice—selecting the blueprint that balances beauty with cost-effective materials.

---

## Objects: The Language of Systems

An **Object** in TENSORGRAPH defines the interface—the essential signature of what flows between our processes.

```python
from tensorgraph import Obj

# Atomic objects
Tensor = Obj("Tensor")
Cache = Obj("Cache")
Audio = Obj("Audio")

# Tensor products (multiple simultaneous interfaces)
Combined = Tensor @ Cache  # Tensor ⊗ Cache
```

Think of objects as the **types of wires** in a circuit diagram. A wire carrying audio is different from a wire carrying video, and you can bundle them together.

```
┌─────────┐
│  Tensor │ ⊗ ┌───────┐ = ┌─────────────────┐
└─────────┘   │ Cache │   │ Tensor ⊗ Cache  │
              └───────┘   └─────────────────┘
```

---

## Morphisms: Processes Between Systems

A **morphism** (or **diagram expression**) represents a typed process from input objects to output objects.

### The Four Constructors

```python
from tensorgraph import Id, Box, Seq, Par, Obj

T = Obj("Tensor")

# 1. Identity: do nothing, pass through
identity = Id(T)  # T → T

# 2. Box: a primitive operation
linear = Box("Linear")  # Declared type via Signature

# 3. Seq: sequential composition (;)
pipeline = Seq(box1, box2)  # "first box1, then box2"

# 4. Par: parallel composition (⊗)
parallel = Par(left, right)  # "left and right, side by side"
```

### Visual Representation

String diagrams flow **left to right**. Boxes are operations, wires are data:

```
Sequential: f ; g

    ┌───┐   ┌───┐
───▶│ f │──▶│ g │──▶
    └───┘   └───┘


Parallel: f ⊗ g

    ┌───┐
───▶│ f │──▶
    └───┘
    ┌───┐
───▶│ g │──▶
    └───┘
```

### Type Checking

TENSORGRAPH enforces that compositions are well-typed:

```python
from tensorgraph import Signature, infer_type

sig = Signature()
sig.add("Encode", Obj("Audio"), Obj("Latent"))
sig.add("Decode", Obj("Latent"), Obj("Audio"))

pipeline = Seq(Box("Encode"), Box("Decode"))
print(infer_type(pipeline, sig))  # (Audio, Audio) ✓

# This would fail:
# Seq(Box("Decode"), Box("Encode"))  # Latent ≠ Audio → TypeError
```

---

## The E-Graph: Many Programs, One Structure

The **e-graph** (equivalence graph) is TENSORGRAPH's secret weapon. It compactly represents **many equivalent programs** in a single data structure.

### The Key Idea

Instead of:
```
Program A ─transform→ Program B ─transform→ Program C
```

We have:
```
┌─────────────────────────────────────┐
│         Equivalence Class           │
│  ┌─────┐   ┌─────┐   ┌─────┐       │
│  │  A  │ ≡ │  B  │ ≡ │  C  │       │
│  └─────┘   └─────┘   └─────┘       │
└─────────────────────────────────────┘
```

All three are recognized as equivalent simultaneously.

### Visual: E-Graph Structure

```
E-Graph after applying rewrites:

┌─ E-Class 1 ─────────────────────────────────┐
│  • (f ; g) ; h                              │
│  • f ; (g ; h)    [associativity applied]   │
└─────────────────────────────────────────────┘
         │
         ▼ (references)
┌─ E-Class 2 ─────────────────────────────────┐
│  • g ; h                                    │
└─────────────────────────────────────────────┘
```

Each **e-class** contains equivalent expressions. E-classes reference other e-classes (not expressions), enabling compact sharing.

### Code Example

```python
from tensorgraph import EGraph, Signature, Obj, Box, Seq

sig = Signature()
T = Obj("T")
sig.add("f", T, T)
sig.add("g", T, T)

# Add expression to e-graph
eg = EGraph(sig)
expr = Seq(Box("f"), Box("g"))
root = eg.add_expr(expr)

# Multiple expressions, same e-class after merging
print(len(eg.nodes[root]))  # Number of representations in the class
```

---

## Saturation & Extraction

### The Two-Phase Dance

**Phase 1: Saturation** — Apply all rewrite rules exhaustively until no new equivalences are discovered.

**Phase 2: Extraction** — Select the lowest-cost expression from the equivalence class.

```
                    ┌──────────────┐
  Input Program ───▶│  Saturation  │───▶ E-Graph (many equivalents)
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Extraction  │───▶ Optimized Program
                    └──────────────┘
```

### Rewrite Rules

A rewrite rule declares an equivalence between patterns:

```python
from tensorgraph import Rewrite, PSeq, PVar

# Pattern: ?x ; ?y can match any sequential composition
# This rule says: (?a ; ?b) ; ?c ≡ ?a ; (?b ; ?c)
associativity = Rewrite(
    name="Associativity",
    lhs=PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
    rhs=PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
)
```

### Cost-Based Extraction

The **Extractor** assigns costs and selects the cheapest:

```python
from tensorgraph import Extractor

extractor = Extractor(eg)
extractor.solve(root)
best = extractor.extract(root)

# Default: each Box costs 1, minimize total boxes
```

### Full Workflow

```python
from tensorgraph import (
    EGraph, Signature, Obj, Box, Seq,
    Rewrite, PSeq, PVar, saturate, Extractor
)

# 1. Define types and operations
sig = Signature()
T = Obj("T")
sig.add("expensive", T, T)
sig.add("cheap", T, T)

# 2. Build input program
prog = Seq(Box("expensive"), Seq(Box("expensive"), Box("cheap")))

# 3. Add to e-graph
eg = EGraph(sig)
root = eg.add_expr(prog)

# 4. Define rewrites (e.g., expensive;expensive ≡ expensive)
fuse_rule = Rewrite(
    name="FuseExpensive",
    lhs=PSeq(PBox("expensive"), PBox("expensive")),
    rhs=PBox("expensive"),
)

# 5. Saturate
saturate(eg, [fuse_rule], iters=10)

# 6. Extract best
ex = Extractor(eg)
ex.solve(root)
best = ex.extract(root)

# best now has fewer boxes!
```

---

## Why This Matters

### The Problem with Traditional Optimization

Traditional compilers apply transformations **one at a time**, making **irrevocable decisions** at each step. This leads to:

1. **Phase ordering problems** — The order of optimizations matters
2. **Local minima** — Greedy choices miss global optima
3. **Brittleness** — Small changes break optimization patterns

### The E-Graph Solution

E-graphs solve this by **deferring decisions**:

1. **Explore all equivalences first** — No premature commitment
2. **Global optimization** — See all options before choosing
3. **Composable rules** — Add rules without phase ordering worries

### Real-World Applications

- **Program Synthesis** — Crafting optimal implementations from abstract specifications.

### Beyond the Single Machine: The Global Fabric

TENSORGRAPH v0.5.0 introduces the **Coordinated Fabric**, a testament to distributed harmony. It allows equality saturation to resonate across multiple machines or shards.

- **Heterogeneous Sharding**: Distributing the cognitive load across a network of shards.
- **Ghost Nodes**: Elegant placeholders maintaining the integrity of distributed state.
- **Async Propagation**: Congruence flowing globally through high-speed, non-blocking channels.

### From Diagrams to Hardware: The Art of Emission

TENSORGRAPH doesn't merely optimize; it **creates**. With the `TritonEmitter` and `CUDAEmitter`, our elegant diagrams are distilled directly into high-performance GPU kernels—fusing multiple refinements into a single, efficient realization.

TENSORGRAPH brings this power to **any domain** that recognizes the beauty of typed string diagrams.

---

## Summary: The TENSORGRAPH Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        TENSORGRAPH FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Source Program                                                │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │  Parse to   │  Objects (types)                              │
│   │  Diagram IR │  Boxes (operations)                           │
│   └─────────────┘  Seq/Par (composition)                        │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │  E-Graph    │  Add expression                               │
│   │  add_expr   │  Track equivalence classes                    │
│   └─────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │ Saturation  │  Apply rewrite rules                          │
│   │   Loop      │  Merge equivalent classes                     │
│   └─────────────┘  Until fixed point                            │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │ Extraction  │  Compute costs                                │
│   │   Engine    │  Select minimum                               │
│   └─────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│   Optimized Program                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

- **[Tutorial](TUTORIAL.md)** — Build your first optimizer hands-on
- **[API Reference](API.md)** — Every function and class documented
- **[Architecture](ARCHITECTURE.md)** — Deep dive into internals

---

*Documentation by Grand Challenge Technologies Ltd. — Frontier Engineering*
