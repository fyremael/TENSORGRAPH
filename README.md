# TENSORGRAPH

TENSORGRAPH is a research compiler for typed diagram intermediate representations, equality saturation, extraction, and bounded GPU lowering.

The repository is in an evidence-recovery phase. It contains a useful compiler core, but it is not represented as production-ready. Capability claims are limited to behavior covered by committed tests or reproducible evidence records.

See:

- [STATUS.md](STATUS.md) for the machine-oriented capability ledger.
- [docs/SEMANTICS.md](docs/SEMANTICS.md) for the categorical and effect assumptions.
- [docs/EVIDENCE_POLICY.md](docs/EVIDENCE_POLICY.md) for benchmark and audit admission rules.

## Current supported surface

The supported research surface is deliberately narrow:

1. Construct typed expressions from `Id`, `Box`, `Seq`, `Par`, and selected structural primitives.
2. Admit expressions into a typed e-graph.
3. Apply typed rewrites under bounded equality saturation.
4. Extract a lower-cost representative.
5. Lower a bounded unary elementwise chain to generated Triton source.
6. On a CUDA/Triton host, compile and execute that generated kernel and compare it with the PyTorch source model.

Other modules are experimental, partial, or retained for investigation. The status ledger is authoritative.

## Installation

```bash
git clone https://github.com/fyremael/TENSORGRAPH.git
cd TENSORGRAPH
python -m pip install -e ".[dev,fx]"
```

For generated-kernel execution on Linux with a compatible NVIDIA GPU:

```bash
python -m pip install -e ".[dev,fx,gpu]"
```

Python 3.10 through 3.13 is supported by package metadata. The CI matrix is authoritative for versions currently exercised.

## Core example

```python
from tensorgraph import Box, EGraph, Extractor, Obj, PBox, PSeq, Rewrite, Seq, Signature, saturate

T = Obj("Tensor")
sig = Signature()
sig.add("ReLU", T, T, traits={"elementwise"})

expr = Seq(Box("ReLU"), Box("ReLU"))
rule = Rewrite(
    name="relu_idempotence",
    lhs=PSeq(PBox("ReLU"), PBox("ReLU")),
    rhs=PBox("ReLU"),
    origin="torch.relu(torch.relu(x)) == torch.relu(x)",
)

egraph = EGraph(sig)
root = egraph.add_expr(expr)
saturate(egraph, [rule], iters=4)

extractor = Extractor(egraph)
extractor.solve(root)
best = extractor.extract(root)
```

## Verified vertical slice

The recovery path adds a bounded compiler pipeline for unary elementwise PyTorch modules:

```text
PyTorch FX capture
→ typed TENSORGRAPH expression
→ bounded equality saturation
→ extraction
→ generated Triton source
→ CUDA/Triton compilation and execution
→ differential comparison with PyTorch
```

Run the CPU-available structural tests:

```bash
pytest tests/test_recovery_semantics.py tests/test_verified_pipeline.py -v
```

Run the generated-kernel differential benchmark on a CUDA/Triton host:

```bash
python benchmarks/bench_verified_elementwise.py \
  --sizes 1024 65536 1048576 \
  --warmup 25 \
  --repetitions 100 \
  --output artifacts/verified-elementwise.json
```

The benchmark exits unsuccessfully if generated TENSORGRAPH code is not executed, numerical comparison fails, required metadata is missing, or the host lacks CUDA/Triton. It does not substitute estimated values.

## Semantics

TENSORGRAPH distinguishes structural syntax from semantic permission.

- Sequential and parallel composition are typed.
- `Swap` belongs to the symmetric monoidal structure.
- `Dup` and `Del` represent Cartesian copying and discarding.
- Copy/delete naturality is valid only for pure morphisms.
- Operations marked with the `effectful` trait do not receive copy/delete naturality rewrites.
- `Case` branches must have the required unit-domain and common-codomain shape.
- `Iter` is restricted to non-negative, statically known endomorphism counts.

These rules are enforced in the e-graph admission path and documented in [docs/SEMANTICS.md](docs/SEMANTICS.md).

## Evidence rules

A performance or compliance claim is admissible only when the repository contains:

- the exact commit identity;
- the command used;
- dependency and hardware metadata;
- raw samples or complete pass/fail records;
- a clear distinction between capture, saturation, extraction, compilation, and execution time;
- direct execution of generated TENSORGRAPH output;
- numerical error measurements against an identified reference.

Synthetic multipliers, handwritten substitute kernels, self-comparisons, and machine-local image links are prohibited as evidence. Historical reports that do not meet these requirements are non-authoritative.

## Project status

| Component | Status | Admission basis |
|---|---|---|
| Typed expression IR | research-supported | unit tests |
| Typed e-graph and extraction | research-supported | unit tests |
| Bounded equality saturation | research-supported | unit tests and trace records |
| FX unary elementwise capture | bounded experimental | differential tests |
| Generated Triton unary chain | bounded experimental | source checks plus optional GPU execution |
| General FX DAG round trip | experimental | incomplete semantic metadata |
| Reductions | experimental | not admitted for general production use |
| CUDA C++ emitter | experimental | source generation only |
| Distributed saturation | scaffold | incomplete workers and synchronization |
| Neural scheduling | experimental | no production claim |
| Production readiness | not established | prohibited claim until promoted by evidence gates |

The detailed ledger in [STATUS.md](STATUS.md) controls over this summary.

## Development

```bash
python -m ruff check tensorgraph tests benchmarks
python -m mypy tensorgraph
python -m pytest tests -v --cov=tensorgraph
python benchmarks/bench_saturation.py
```

GPU evidence is a separate, explicitly marked job because standard GitHub-hosted Linux runners do not provide the required NVIDIA execution environment.

## Repository policy

Changes that alter semantics, code generation, or benchmark claims require:

1. a falsifiable test;
2. an update to the capability ledger when status changes;
3. raw evidence for any performance statement;
4. review of effect, type, and numerical assumptions;
5. CI success on the committed revision.

## License

MIT. See [LICENSE](LICENSE).
