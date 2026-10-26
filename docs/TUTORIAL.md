# TENSORGRAPH Tutorial — The Craft of Optimization

> *A guided journey from first principles to a high-performance diagram optimizer.*

This tutorial invites you to master the art of custom optimization using TENSORGRAPH. We will journey together through the creation of a refined system that automatically harmonizes redundant operations into their most efficient forms.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation & Verification](#installation--verification)
3. [Step 1: Define Your Types](#step-1-define-your-types)
4. [Step 2: Build an Expression](#step-2-build-an-expression)
5. [Step 3: Create a Rewrite Rule](#step-3-create-a-rewrite-rule)
6. [Step 4: Run Saturation](#step-4-run-saturation)
7. [Step 5: Extract the Best](#step-5-extract-the-best)
8. [Step 6: Inspect with Tracing](#step-6-inspect-with-tracing)
9. [Complete Working Example](#complete-working-example)
10. [Next Steps](#next-steps)

---

## Prerequisites

**You should be comfortable with:**
- Python 3.10+ basics
- The concept of types and composition
- (Optional) Basic category theory intuition helps but isn't required

**Time required:** 15-20 minutes

---

## Installation & Verification

### Install TENSORGRAPH

```bash
# Clone and install
cd TENSORGRAPH
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import tensorgraph; print('✓ TENSORGRAPH installed')"
python -m tensorgraph.examples.demo_core
```

You should see the LoRA fusion demo output. If so, you're ready!

---

## Step 1: Define Your Types

Every TENSORGRAPH program starts by defining the **types** (Objects) that flow between operations.

```python
from tensorgraph import Obj, Signature

# Create atomic type objects
Tensor = Obj("Tensor")
Latent = Obj("Latent")

# You can also create compound types with tensor product
TensorPair = Tensor @ Tensor  # Two tensors bundled together

print(Tensor)        # Tensor
print(TensorPair)    # (Tensor ⊗ Tensor)
```

### Define a Signature

The **Signature** registers operations with their input/output types:

```python
sig = Signature()

# add(name, domain, codomain)
sig.add("Encode", Tensor, Latent)   # Tensor → Latent
sig.add("Decode", Latent, Tensor)   # Latent → Tensor
sig.add("Transform", Latent, Latent)  # Latent → Latent

# Verify
print(sig.get("Encode"))  # OpDef(name='Encode', dom=Tensor, cod=Latent)
```

---

## Step 2: Build an Expression

Now construct a **diagram expression** representing your program:

```python
from tensorgraph import Box, Seq, Id, pretty

# Create boxes (primitive operations)
encode = Box("Encode")
decode = Box("Decode")
transform = Box("Transform")

# Compose sequentially: Encode ; Transform ; Decode
pipeline = Seq(encode, Seq(transform, decode))

# Pretty print
print(pretty(pipeline))
# Output: (Encode ; (Transform ; Decode))
```

### Type Checking

TENSORGRAPH enforces type safety:

```python
from tensorgraph import infer_type

# Check the type of our pipeline
dom, cod = infer_type(pipeline, sig)
print(f"Pipeline: {dom} → {cod}")
# Output: Pipeline: Tensor → Tensor
```

### Using Parallel Composition

For operations that run side-by-side:

```python
from tensorgraph import Par

# Two encoders in parallel
dual_encode = Par(encode, encode)

# Type: (Tensor ⊗ Tensor) → (Latent ⊗ Latent)
print(infer_type(dual_encode, sig))
```

---

## Step 3: Create a Rewrite Rule

Rewrite rules define **equivalences** between diagram patterns.

### Pattern Language

Patterns mirror expressions but allow **variables**:

| Pattern | Matches |
|---------|---------|
| `PVar("x")` | Any expression |
| `PBox("Op")` | A box with that op name |
| `PSeq(a, b)` | Sequential composition |
| `PPar(a, b)` | Parallel composition |
| `PId(obj)` | Identity on an object |

### Example: Fusing Consecutive Transforms

Let's create a rule that fuses adjacent `Transform` operations:

```python
from tensorgraph import Rewrite, PSeq, PBox

# Rule: Transform ; Transform ≡ Transform
# (Two transforms in a row can be replaced by one)
fuse_transforms = Rewrite(
    name="FuseTransforms",
    lhs=PSeq(PBox("Transform"), PBox("Transform")),
    rhs=PBox("Transform"),
)
```

### Pattern Variables

For more flexible rules, use variables:

```python
from tensorgraph import PVar

# Rule: (x ; y) ; z ≡ x ; (y ; z)  [associativity]
associativity = Rewrite(
    name="SeqAssoc",
    lhs=PSeq(PSeq(PVar("x"), PVar("y")), PVar("z")),
    rhs=PSeq(PVar("x"), PSeq(PVar("y"), PVar("z"))),
)
```

---

## Step 4: Run Saturation

**Saturation** applies all rules exhaustively until no new equivalences are found.

```python
from tensorgraph import EGraph, saturate

# Create e-graph from signature
eg = EGraph(sig)

# Add our expression to the e-graph
root = eg.add_expr(pipeline)
eg.root = root

# Define a program with redundant transforms
redundant = Seq(encode, Seq(transform, Seq(transform, decode)))
root = eg.add_expr(redundant)

# Run saturation with our fusion rule
saturate(eg, [fuse_transforms], iters=10)

print(f"E-graph has {len(eg.nodes)} equivalence classes")
```

### What Happens During Saturation?

1. **Match**: Find all places where `Transform ; Transform` appears
2. **Instantiate**: Create the RHS (`Transform`)
3. **Merge**: Mark the original and replacement as equivalent
4. **Rebuild**: Propagate equivalences (congruence closure)
5. **Repeat**: Until nothing changes

---

## Step 5: Extract the Best

After saturation, **extract** the lowest-cost equivalent program:

```python
from tensorgraph import Extractor

# Create extractor with default cost (count boxes)
extractor = Extractor(eg)

# Solve for optimal
extractor.solve(root)

# Extract the best expression
best = extractor.extract(root)

print("Original:", pretty(redundant))
print("Optimized:", pretty(best))
```

**Expected output:**
```
Original: (Encode ; (Transform ; (Transform ; Decode)))
Optimized: (Encode ; (Transform ; Decode))
```

🎉 The redundant `Transform` was eliminated!

---

## Step 6: Inspect with Tracing

TENSORGRAPH can record every rewrite application for debugging:

```python
from tensorgraph import Trace

# Create a trace object
trace = Trace()

# Pass it to saturate
eg2 = EGraph(sig)
root2 = eg2.add_expr(redundant)
saturate(eg2, [fuse_transforms], iters=10, trace=trace)

# Inspect what happened
print(f"Applied {len(trace)} rewrites")
print(trace.summary())  # {'FuseTransforms': N}

# Detailed dump
print(trace.dump(max_entries=5))
```

### Trace Entry Details

Each entry records:
- `rule_name`: Which rule fired
- `expr_env`: Pattern variable bindings (as e-class IDs)
- `obj_env`: Object variable bindings
- `merged_from`, `merged_to`: The merge that occurred

---

## Complete Working Example

Here's everything together in one runnable script:

```python
#!/usr/bin/env python3
"""TENSORGRAPH Tutorial: Complete Working Example"""

from tensorgraph import (
    Obj, Signature,
    Box, Seq, Id, Par,
    pretty, infer_type, normalize,
    Rewrite, PSeq, PBox, PVar,
    EGraph, saturate, Extractor, Trace,
)


def main():
    # ─────────────────────────────────────────────────────────
    # Step 1: Define types and signature
    # ─────────────────────────────────────────────────────────
    Tensor = Obj("Tensor")
    Latent = Obj("Latent")

    sig = Signature()
    sig.add("Encode", Tensor, Latent)
    sig.add("Decode", Latent, Tensor)
    sig.add("Transform", Latent, Latent)

    # ─────────────────────────────────────────────────────────
    # Step 2: Build an expression with redundancy
    # ─────────────────────────────────────────────────────────
    encode = Box("Encode")
    decode = Box("Decode")
    transform = Box("Transform")

    # Encode → Transform → Transform → Transform → Decode
    # (Two consecutive Transforms can be fused)
    redundant = normalize(
        Seq(encode, 
            Seq(transform, 
                Seq(transform, 
                    Seq(transform, decode))))
    )

    print("=" * 60)
    print("TENSORGRAPH Optimization Demo")
    print("=" * 60)
    print(f"\nOriginal program: {pretty(redundant)}")
    print(f"Type: {infer_type(redundant, sig)}")

    # ─────────────────────────────────────────────────────────
    # Step 3: Define rewrite rule
    # ─────────────────────────────────────────────────────────
    fuse_transforms = Rewrite(
        name="FuseTransforms",
        lhs=PSeq(PBox("Transform"), PBox("Transform")),
        rhs=PBox("Transform"),
    )

    print(f"\nRewrite rule: Transform ; Transform ≡ Transform")

    # ─────────────────────────────────────────────────────────
    # Step 4: Run saturation with tracing
    # ─────────────────────────────────────────────────────────
    eg = EGraph(sig)
    root = eg.add_expr(redundant)
    eg.root = root

    trace = Trace()
    saturate(eg, [fuse_transforms], iters=10, trace=trace)

    print(f"\nSaturation complete:")
    print(f"  - Applied {len(trace)} rewrite(s)")
    print(f"  - E-graph has {len(eg.nodes)} equivalence class(es)")

    # ─────────────────────────────────────────────────────────
    # Step 5: Extract the best
    # ─────────────────────────────────────────────────────────
    extractor = Extractor(eg)
    extractor.solve(root)
    best = extractor.extract(root)

    print(f"\nOptimized program: {pretty(best)}")
    print(f"Type: {infer_type(best, sig)}")

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    def count_boxes(e):
        if isinstance(e, Box): return 1
        if isinstance(e, Seq): return count_boxes(e.first) + count_boxes(e.second)
        return 0

    original_boxes = count_boxes(redundant)
    optimized_boxes = count_boxes(best)

    print(f"\n✓ Reduced from {original_boxes} boxes to {optimized_boxes} boxes")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python tutorial_example.py
```

**Expected output:**
```
============================================================
TENSORGRAPH Optimization Demo
============================================================

Original program: (Encode ; (Transform ; (Transform ; (Transform ; Decode))))
Type: (Tensor, Tensor)

Rewrite rule: Transform ; Transform ≡ Transform

Saturation complete:
  - Applied 2 rewrite(s)
  - E-graph has N equivalence class(es)

Optimized program: (Encode ; (Transform ; Decode))
Type: (Tensor, Tensor)

✓ Reduced from 5 boxes to 3 boxes
============================================================
```

---

## Next Steps

You've learned the core TENSORGRAPH workflow:

1. ✅ Define types and signatures
2. ✅ Build diagram expressions
3. ✅ Write rewrite rules
4. ✅ Run saturation
5. ✅ Extract optimized programs
6. ✅ Inspect with tracing

**Where to go next:**

- **[API Reference](API.md)** — Complete documentation of every function
- **[Architecture](ARCHITECTURE.md)** — Understand the internals
- **[torch.fx Backend](../tensorgraph/cli/optimize_fx.py)** — Real-world integration example
- **[SPEC.md](../SPEC.md)** — Formal specification

### Going Further with v0.5.0

TENSORGRAPH v0.5.0 adds powerful industrial-grade features:

- **Distributed Saturation**: Use `AsyncFabric` to scale optimization across shards.
- **GPU Codegen**: Turn your optimized diagrams into Triton or CUDA C++ kernels.
- **Performance Regression**: Keep your optimizer fast with `tensorgraph.benchmarks.regression`.

Explore the **[SPEC.md](../SPEC.md)** for a deep dive into these advanced capabilities.

---

## Common Patterns

### Custom Cost Functions

```python
def my_cost(enode):
    """Custom cost: expensive ops cost more."""
    if enode.tag == "Box":
        op, _ = enode.data
        if op == "ExpensiveOp":
            return 10
        return 1
    return 0

extractor = Extractor(eg, local_cost=my_cost)
```

### Builder Functions for Complex RHS

```python
def complex_rhs_builder(eg, root, env, oenv):
    """Programmatically build the RHS."""
    # Access matched variables
    x_eclass = env["x"]
    
    # Build new expression
    new_expr = Box.with_attrs("Fused", source=x_eclass)
    
    return eg.add_expr(new_expr)

rule = Rewrite(
    name="ComplexRule",
    lhs=PSeq(PVar("x"), PVar("y")),
    rhs=complex_rhs_builder,  # Callable instead of pattern
)
```

### Filtering Trace by Rule

```python
# Find all applications of a specific rule
fuse_entries = trace.filter_by_rule("FuseTransforms")

# Find all merges involving a specific e-class
related = trace.filter_by_eclass(some_eclass_id)
```

---

*Documentation by Grand Challenge Technologies Ltd. — Frontier Engineering*
