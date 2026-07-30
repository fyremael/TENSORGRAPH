# TENSORGRAPH: Diagrammatic Rewriting Compiler

<div align="center">

![TENSORGRAPH](docs/assets/tensorgraph_architecture.png)

</div>

<div align="center">

[![GCT Frontier Engineering](https://img.shields.io/badge/GCT-Frontier%20Engineering-050505?style=for-the-badge&labelColor=0a0a0a&color=00F0FF)](https://www.grandchallenge.io)
[![Status](https://img.shields.io/badge/Status-Operational-00ff66?style=for-the-badge)](https://github.com/gct/tensorgraph)
[![License](https://img.shields.io/badge/License-MIT-A0A0A0?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

*Transform programs through equality saturation on string diagrams*


</div>

---

## What is TENSORGRAPH?

TENSORGRAPH represents programs as **typed string diagrams** and optimizes them through **equality saturation** on an **e-graph**. Instead of applying transformations one-by-one, it explores *all equivalent programs simultaneously* and extracts the cheapest one.

```mermaid
graph LR
    style P fill:#0a0a0a,stroke:#333,stroke-width:2px,color:#fff
    style S fill:#0f1a20,stroke:#00f0ff,stroke-width:2px,color:#00f0ff
    style E fill:#1a1005,stroke:#ff9900,stroke-width:2px,color:#ff9900

    P[Program<br/>(Diagram)] -->|Add| S{Saturate<br/>(E-Graph)}
    S -- Rules --> S
    S -->|Extract| E[Optimal<br/>(Term)]
```

**Key concepts:**
- **Objects** — Types representing system interfaces
- **1-Morphisms** — Typed diagrams (`Id`, `Box`, `Seq`, `Par`)
- **2-Morphisms** — Rewrite rules defining equivalences
- **E-Graph** — Compact representation of all equivalent programs
- **Extraction** — Cost-based selection of optimal program

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/TENSORGRAPH.git
cd TENSORGRAPH

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -m tensorgraph.examples.demo_core
```

**Requirements:** Python ≥ 3.10

**Optional:** PyTorch ≥ 2.0 for the FX backend (`pip install -e ".[fx]"`)

---

## Quick Start

### 30 Seconds: See it Work

```bash
python -m tensorgraph.examples.demo_core
```

**Output:**
```
=== LoRA fusion demo (e-graph equality saturation) ===
Original: ((InjectLoRA(deltas=('A1B1',)) ⊗ Id[X]) ; ((InjectLoRA(deltas=('A2B2',)) ⊗ Id[X]) ; LinearApply))
Boxes: 3
Best:   ((InjectLoRA(deltas=('A1B1', 'A2B2')) ⊗ Id[X]) ; LinearApply)
Boxes:  2
```

Two `InjectLoRA` operations were automatically fused into one!

### 5 Minutes: Build Your First Optimizer

```python
from tensorgraph import (
    Obj, Signature, Box, Seq, pretty,
    Rewrite, PSeq, PBox, EGraph, saturate, Extractor
)

# 1. Define types
T = Obj("T")
sig = Signature()
sig.add("f", T, T)
sig.add("g", T, T)

# 2. Build a program
prog = Seq(Box("f"), Seq(Box("f"), Box("g")))
print("Before:", pretty(prog))  # (f ; (f ; g))

# 3. Define a rewrite: f ; f ≡ f
fuse = Rewrite(
    name="FuseF",
    lhs=PSeq(PBox("f"), PBox("f")),
    rhs=PBox("f"),
)

# 4. Optimize
eg = EGraph(sig)
root = eg.add_expr(prog)
saturate(eg, [fuse], iters=10)

# 5. Extract the best
ex = Extractor(eg)
ex.solve(root)
best = ex.extract(root)
print("After:", pretty(best))  # (f ; g)
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[CONCEPTS.md](docs/CONCEPTS.md)** | Mental model, intuition pumps, visual diagrams |
| **[TUTORIAL.md](docs/TUTORIAL.md)** | Step-by-step hands-on guide |
| **[API.md](docs/API.md)** | Complete reference for every symbol |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Internals deep-dive for contributors |
| **[SPEC.md](SPEC.md)** | Formal specification |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Development workflow |

---

## Core Features

### Typed String Diagrams

Programs are represented as composable, typed diagrams:

```python
from tensorgraph import Obj, Box, Seq, Par, Id

Tensor = Obj("Tensor")
Latent = Obj("Latent")

# Sequential: encode → transform → decode
pipeline = Seq(Box("Encode"), Seq(Box("Transform"), Box("Decode")))

# Parallel: two encoders side-by-side
dual = Par(Box("Encode"), Box("Encode"))

# Type-checked composition
# Seq(Box("Decode"), Box("Encode"))  # TypeError if types don't match!
```

### Pattern-Based Rewriting

Define equivalences with pattern matching:

```python
from tensorgraph import Rewrite, PSeq, PVar, PBox

# Associativity: (a ; b) ; c ≡ a ; (b ; c)
assoc = Rewrite(
    name="Assoc",
    lhs=PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
    rhs=PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
)

# Fusion: expensive ; expensive ≡ expensive
fuse = Rewrite(
    name="Fuse",
    lhs=PSeq(PBox("Expensive"), PBox("Expensive")),
    rhs=PBox("Expensive"),
)
```

### Automated Kernel Fusion

TENSORGRAPH v0.4.0+ automatically fuses sequential and parallel operations into optimized GPU kernels.

```python
from tensorgraph.codegen.triton import TritonEmitter
from tensorgraph.codegen.cuda import CUDAEmitter

# Fuse Seq(ReLU, Sigmoid) into single kernel
expr = Seq(Box("ReLU"), Box("Sigmoid"))
emitter = TritonEmitter(sig)
kernel_code = emitter.emit(expr)
```

### Distributed Sharding

Scale optimization across nodes with the **AETHER Coordination Fabric**.

```python
from tensorgraph.dist.sharding import Shard
from tensorgraph.dist.fabric import create_fabric

# Create an asynchronous distributed fabric
fabric = create_fabric("async", batch_size=100)
fabric.start()

shard = Shard(shard_id=1, fabric=fabric, sig=sig)
# ... distributed equality propagation is automatic!
```

### Neural Heuristics (`tensorgraph.neural`)

Experimental support for learned rule scheduling:

```python
from tensorgraph.neural import GNNStateEmbedder, PolicyNetwork, NeuralScheduler

# Use learned policy to guide saturation
scheduler = NeuralScheduler(
    policy_net=PolicyNetwork(),
    embedder=GNNStateEmbedder(),
    mode="hybrid"  # Mix of privacy and exploration
)
trace = scheduler.saturate(egraph, rules)
```

### Interactive Explorer

TENSORGRAPH includes a high-fidelity visualization tool for debugging saturation:

```bash
# Start the WebSocket server and saturation demo
python showcase/demo_self_contained.py
```

Then open `showcase/egraph_explorer.html` to see:
- **Optimization Timeline**: Step-by-step replay
- **Visual E-Graph**: Force-directed layout of e-classes
- **Inspector**: Deep dive into equivalent terms

---

## Project Status

| Component | Version | Status |
|-----------|---------|--------|
| Typed IR | v0.1.0 | ✅ Complete |
| E-Graph Engine | v0.2.0 | ✅ Complete |
| Neural Heuristics | v0.3.0 | ✅ Beta |
| Dynamic Control Flow | v0.4.0 | ✅ Complete |
| Heterogeneous Sharding | v0.4.0 | ✅ Complete |
| Triton Codegen | v0.4.0 | ✅ Complete |
| CUDA Backend | v0.5.0 | ✅ Complete |
| Performance Suite | v0.5.0 | ✅ Complete |
| Production Fabric | v0.5.0 | ✅ Complete |

**SPEC.md Compliance:** 100% (13/13 functional sections)

---

## Performance & Scaling

TENSORGRAPH v0.5.0 includes a formal **Performance Regression Suite** to ensure optimization and codegen latency remains within bounded limits.

```bash
# Run the CI-grade regression suite
python -m tensorgraph.benchmarks.regression --ci
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests -v

# Lint & type check
python -m ruff check tensorgraph tests
python -m mypy tensorgraph

# Run demos
python -m tensorgraph.examples.demo_core
python -m tensorgraph.cli.optimize_fx --in-dim 16 --out-dim 8
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

---

## Architecture Overview

```
tensorgraph/
├── types.py          # Obj, Sort
├── signature.py      # Operation registry
├── ir/               # Expression IR (Id, Box, Seq, Par)
├── rewrite/          # Patterns, rules, matching
├── egraph/           # E-graph, saturation, extraction, tracing
├── backends/         # torch.fx integration
└── cli/              # Command-line tools
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full deep-dive.

---

## Why E-Graphs?

Traditional optimizers apply transformations **sequentially**, making irrevocable decisions at each step. This leads to:
- **Phase ordering problems** — Optimization order matters
- **Local minima** — Greedy choices miss global optima
- **Brittleness** — Small changes break patterns

E-graphs solve this by **deferring decisions**:
1. **Explore all equivalences** — No premature commitment
2. **Global view** — See all options before choosing
3. **Composable rules** — Add rules without phase ordering worries

---

## License

MIT — See [LICENSE](LICENSE)

---

## Acknowledgments

TENSORGRAPH builds on foundational work in:
- **E-graphs:** [egg](https://egraphs-good.github.io/) and equality saturation
- **String diagrams:** Categorical compositional semantics
- **torch.fx:** PyTorch's graph capture and transformation framework

---

<div align="center">

*Documentation by Grand Challenge Technologies Ltd. — Frontier Engineering*

</div>
