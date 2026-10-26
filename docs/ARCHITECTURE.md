# TENSORGRAPH Architecture — Internals Deep Dive

> *For contributors and the deeply curious.*

This document explains **how** TENSORGRAPH works internally. For **what** it does, see [CONCEPTS.md](CONCEPTS.md).

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Module Architecture](#module-architecture)
3. [Data Flow](#data-flow)
4. [Key Invariants](#key-invariants)
5. [Algorithmic Details](#algorithmic-details)
6. [Extension Points](#extension-points)
7. [Design Decisions](#design-decisions)

---

## Design Philosophy

### Core Principles

1. **Types are data.** Everything is immutable, hashable, and structural.
2. **Composition is explicit.** No implicit side effects in IR construction.
3. **Equivalence is first-class.** The e-graph treats equality as structure, not transformation.
4. **Traceability over magic.** Every optimization step can be inspected.

### Non-Goals (by design)

- Not a general theorem prover
- Not a full graph rewriting system (no cycles in MVP)
- Not optimized for million-node e-graphs (correctness first)

---

## Module Architecture

```
tensorgraph/
├── __init__.py          # Package exports
├── types.py             # Obj, ObjVar, Sort
├── signature.py         # Signature, OpDef
├── adjunction.py        # Adjunction mates
│
├── ir/                  # Intermediate Representation
│   ├── __init__.py
│   ├── expr.py          # Id, Box, Seq, Par, pretty
│   └── normalize.py     # normalize, infer_type
│
├── rewrite/             # Pattern Matching & Rules
│   ├── __init__.py
│   ├── pattern.py       # Pattern types, ematch
│   └── rule.py          # Rewrite, instantiate_pattern
│
├── egraph/              # Equality Graph Engine
│   ├── __init__.py
│   ├── enode.py         # ENode dataclass
│   ├── unionfind.py     # UnionFind (disjoint sets)
│   ├── egraph.py        # EGraph core
│   ├── saturation.py    # saturate loop
│   ├── extract.py       # Extractor, default_cost
│   └── trace.py         # Trace, TraceEntry
│
├── codegen/             # Kernel Generation (v0.4.0+)
│   ├── triton.py        # Triton Emitter (elementwise + reduction)
│   └── cuda.py          # CUDA Emitter (fallback path)
│
├── dist/                # Distributed Coordination (v0.5.0+)
│   ├── fabric.py        # Production Fabric (Async/TCP)
│   ├── mock_fabric.py   # Simulation Fabric
│   └── sharding.py      # Shard, Partition, Ghost nodes
│
├── benchmarks/          # Performance Monitoring
│   └── regression.py    # Regression Suite & Baselines
│
├── backends/            # External Integrations
│   └── fx.py            # torch.fx import/export
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        tensorgraph                             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │  types  │          │ signature │         │ adjunction│
   └─────────┘          └──────────┘          └──────────┘
        │                     │                     │
        └──────────┬──────────┘                     │
                   ▼                                │
              ┌─────────┐                           │
              │   ir    │◀──────────────────────────┘
              └─────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌─────────┐          ┌──────────┐
   │ rewrite │          │  egraph  │
   └─────────┘          └──────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
              ┌─────────┐
              │backends │
              └─────────┘
```

---

## Data Flow

### Full Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Program                           │
│              (Python model, FX graph, etc.)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Import                            │
│         fx_chain_to_ops → ops_to_expr                       │
│                                                             │
│  Result: Expr (Id | Box | Seq | Par)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Normalization                            │
│         normalize(expr) → canonical form                    │
│                                                             │
│  • Remove identity compositions                             │
│  • Right-associate Seq                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   E-Graph Addition                          │
│              eg.add_expr(normalized)                        │
│                                                             │
│  • Recursive descent                                        │
│  • Hashcons each ENode                                      │
│  • Assign sort (dom, cod) to each e-class                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Saturation                              │
│           saturate(eg, rewrites, iters)                     │
│                                                             │
│  For each iteration:                                        │
│    For each rule:                                           │
│      matches = ematch(eg, rule.lhs)                         │
│      For each match:                                        │
│        rhs_id = instantiate(rule.rhs, env)                  │
│        eg.merge(match_root, rhs_id)                         │
│    eg.rebuild()  # restore congruence                       │
│                                                             │
│  Until: no new merges OR iteration limit                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Extraction                              │
│    extractor.solve(root) → extractor.extract(root)         │
│                                                             │
│  • Dynamic programming over e-classes                       │
│  • Compute minimum cost for each class                      │
│  • Reconstruct best expression                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Export                            │
│        expr_to_sequential_module(best, ...)                 │
│                                                             │
│  Result: Optimized PyTorch module                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Kernel Generation                         │
│           TritonEmitter | CUDAEmitter                       │
│                                                             │
│  • Infer kernel types (elementwise vs reduction)            │
│  • Fuse Seq/Par chains into single loops                    │
│  • Shared memory parallel reduction (CUDA)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Invariants

### Typing Invariants

1. **E-classes have fixed sorts.** Once created, `(dom, cod)` never changes.
2. **Merges require sort equality.** Cannot merge `A→B` with `C→D`.
3. **ENodes reference e-class IDs.** Children are always canonical reps.

### Congruence Invariant

After `rebuild()`:
> If two e-nodes have identical structure (same tag, data, and children), they belong to the same e-class.

This ensures that structural equality implies semantic equality.

### E-Graph Invariants

1. **No dangling references.** Every child ID exists in `nodes`.
2. **Representative consistency.** `uf.find(id)` always returns a key in `nodes`.
3. **Hashcons completeness.** Every e-node in `nodes` is in `memo`.

---

## Algorithmic Details

### E-Matching (`ematch`)

Pattern matching against e-graph structure:

```
ematch(eg, pattern):
    results = []
    for eclass in eg.nodes:
        for (env, oenv) in match_at(eg, eclass, pattern, {}, {}):
            results.append((eclass, env, oenv))
    return results

match_at(eg, eclass, pattern, env, oenv):
    if pattern is PVar:
        if pattern.name in env:
            return [(env, oenv)] if env[name] == eclass else []
        return [(env | {name: eclass}, oenv)]
    
    if pattern is PBox:
        for enode in eg.nodes[eclass]:
            if enode matches PBox:
                yield (env, oenv)
    
    if pattern is PSeq/PPar:
        for enode in eg.nodes[eclass]:
            if enode.tag matches:
                for sub_match in children:
                    yield combined result
```

**Complexity:** O(|patterns| × |e-nodes| × match_depth)

### Rebuild (Congruence Closure)

```
rebuild(eg):
    changed = True
    while changed:
        changed = False
        
        # Canonicalize all children
        for eclass, enodes in eg.nodes:
            for enode in enodes:
                enode.children = [uf.find(c) for c in enode.children]
        
        # Find congruent pairs
        for enode1, enode2 in all_pairs:
            if structurally_equal(enode1, enode2):
                if eg.merge(class1, class2):
                    changed = True
```

### Extraction (Dynamic Programming)

```
solve(root):
    best_cost = {eclass: ∞ for eclass in eg}
    best_node = {}
    
    changed = True
    while changed:
        changed = False
        
        for eclass in eg.nodes:
            for enode in eg.nodes[eclass]:
                child_cost = sum(best_cost[c] for c in enode.children)
                total = local_cost(enode) + child_cost
                
                if total < best_cost[eclass]:
                    best_cost[eclass] = total
                    best_node[eclass] = enode
                    changed = True

extract(root):
    enode = best_node[root]
    return reconstruct(enode)  # Recursive

### Distributed Coordination (Fabric)

TENSORGRAPH v0.5.0 uses a **Fabric** protocol for distributed equality saturation:

1. **Local Merge:** A merge occurs on Shard A.
2. **Notification:** Shard A broadcasts a `FabricMessage(MERGE)` via the `Fabric`.
3. **Propagation:** Shard B receives the message and performs a `receive_merge` on its local e-graph.
4. **Convergence:** Congruence closure on Shard B propagates the equality through local nodes.
```

---

## Extension Points

### Custom Cost Functions

```python
def my_cost(enode: ENode) -> int:
    """Per-node cost contribution."""
    if enode.tag == "Box":
        op, attrs = enode.data
        # Custom logic based on operation
        return custom_cost_table.get(op, 1)
    return 0

extractor = Extractor(eg, local_cost=my_cost)
```

### Builder-Based RHS

For rules requiring computation:

```python
def my_builder(eg, root, env, oenv):
    """Programmatically construct RHS."""
    # Access matched bindings
    x_class = env["x"]
    obj_val = oenv["T"]
    
    # Build new expression
    new_expr = compute_fused(eg, x_class)
    return eg.add_expr(new_expr)

rule = Rewrite(name="...", lhs=pattern, rhs=my_builder)
```

### New Backends

To add a new backend:

1. **Import:** Parse source format to `Expr`
2. **Export:** Convert `Expr` back to target format
3. **Register operations** in a `Signature`

Template:
```python
def import_from_myformat(source) -> tuple[Expr, Signature]:
    sig = Signature()
    # Parse and register ops
    expr = build_expr_from_source(source, sig)
    return expr, sig

def export_to_myformat(expr: Expr) -> MyFormat:
    # Convert Expr tree back to target
    pass
```

---

## Design Decisions

### Why Immutable Data Structures?

- **Hashability:** Required for e-graph hashconsing
- **Thread safety:** No mutation races
- **Debugging:** Values don't change beneath you
- **Composability:** Easy to test and reason about

### Why Separate Pattern and Expr Types?

Patterns and expressions are conceptually different:
- **Expr:** Concrete programs with specific operations
- **Pattern:** Templates with variables that match many expressions

Keeping them separate:
- Prevents accidental misuse
- Enables pattern-specific features (variables, wildcards)
- Clearer type signatures

### Why Right-Associate Seq?

Normalization right-associates `Seq`:
```
(a ; b) ; c  →  a ; (b ; c)
```

Benefits:
- **Canonical form:** One representation per equivalence class
- **Simpler matching:** Patterns assume right association
- **Reduced e-graph size:** Fewer redundant nodes

### Why Optional Tracing?

Tracing is optional (`trace=None` by default) because:
- **Performance:** Recording every rewrite has overhead
- **Memory:** Large traces consume memory
- **Flexibility:** Not always needed

When enabled, tracing records complete information for debugging and auditing.

---

## Testing Strategy

### Unit Tests (`tests/test_core.py`)

1. **Type checking:** Ill-typed compositions raise `TypeError`
2. **Normalization:** Verifies canonical forms
3. **Saturation:** Applies rules and checks equivalence
4. **Extraction:** Verifies cost optimization
5. **Tracing:** Confirms trace recording (FR-7)

### Running Tests

```bash
python -m pytest tests -v
python -m pytest tests -v --cov=tensorgraph  # With coverage
```

### Demo Validation

```bash
python -m tensorgraph.examples.demo_core      # Core sanity
python -m tensorgraph.cli.optimize_fx ...     # FX round-trip
```

---

## Performance Considerations

### Current Optimizations

1. **Hashconsing:** Avoids duplicate e-nodes
2. **Union-find with path compression:** O(α(n)) operations
3. **Iterative DP extraction:** Avoids stack overflow on deep graphs

### Known Limitations

1. **No e-class analysis:** Could enable abstract interpretation
2. **No parallel saturation:** Single-threaded
3. **No incremental rebuild:** Full rebuild each iteration

### Future Optimization Opportunities

- E-class analyses for domain-specific optimizations
- Parallel e-matching
- Incremental congruence closure
- Better scheduling heuristics

---

*Documentation by Grand Challenge Technologies Ltd. — Frontier Engineering*
