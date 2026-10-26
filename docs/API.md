# TENSORGRAPH API Reference

> *Complete documentation for every public symbol in TENSORGRAPH.*

This reference covers all public functions, classes, and types. For conceptual understanding, see [CONCEPTS.md](CONCEPTS.md). For hands-on learning, see [TUTORIAL.md](TUTORIAL.md).

---

## Table of Contents

1. [Types Module](#types-module)
2. [Signature Module](#signature-module)
3. [IR Module](#ir-module)
4. [Rewrite Module](#rewrite-module)
5. [E-Graph Module](#e-graph-module)
6. [Saturation](#saturation)
7. [Extraction](#extraction)
8. [Tracing](#tracing)
9. [Adjunctions](#adjunctions)
10. [Backends](#backends)

---

## Types Module

**Import:** `from tensorgraph import Obj, ObjVar, ObjLike, Sort`

### `Obj`

```python
@dataclass(frozen=True)
class Obj:
    """A type object representing a system interface."""
    
    name: str
    left: Obj | None = None
    right: Obj | None = None
```

**Purpose:** Represents atomic types or tensor products.

**Methods:**

| Method | Description |
|--------|-------------|
| `Obj.tensor(a, b)` | Create tensor product `a ⊗ b` |
| `a @ b` | Operator shorthand for `Obj.tensor(a, b)` |
| `obj.is_tensor()` | Returns `True` if this is a product type |

**Example:**
```python
Tensor = Obj("Tensor")
Cache = Obj("Cache")
Combined = Tensor @ Cache  # (Tensor ⊗ Cache)
```

---

### `ObjVar`

```python
@dataclass(frozen=True)
class ObjVar:
    """Pattern variable that can match an object."""
    name: str
```

**Purpose:** Used in patterns to match any object type.

**Example:**
```python
from tensorgraph import ObjVar, PId

# Pattern that matches identity on any object
any_identity = PId(ObjVar("X"))
```

---

### `Sort`

```python
Sort = tuple[Obj, Obj]  # (domain, codomain)
```

**Purpose:** Type alias for morphism types. `(A, B)` means "A → B".

---

## Signature Module

**Import:** `from tensorgraph import Signature, OpDef`

### `Signature`

```python
class Signature:
    """Mapping from operation names to their type signatures."""
```

**Purpose:** Registry of all operations with their input/output types.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(name: str, dom: Obj, cod: Obj) → None` | Register an operation |
| `get` | `(name: str) → OpDef` | Retrieve operation definition |
| `__contains__` | `(name: str) → bool` | Check if operation exists |
| `__len__` | `() → int` | Count of registered operations |

**Example:**
```python
sig = Signature()
sig.add("Linear", Obj("Tensor"), Obj("Tensor"))
sig.add("Encode", Obj("Audio"), Obj("Latent"))

print("Linear" in sig)  # True
print(len(sig))         # 2
```

---

### `OpDef`

```python
@dataclass(frozen=True)
class OpDef:
    """Primitive generator declaration."""
    name: str
    dom: Obj
    cod: Obj
```

**Purpose:** Immutable record of an operation's type signature.

---

## IR Module

**Import:** `from tensorgraph import Expr, Id, Box, Seq, Par, pretty, normalize, infer_type`

### `Expr`

```python
@dataclass(frozen=True)
class Expr:
    """Base class for diagram terms (1-morphisms)."""
```

**Purpose:** Abstract base for all expression types.

---

### `Id`

```python
@dataclass(frozen=True)
class Id(Expr):
    """Identity morphism: A → A"""
    obj: Obj
```

**Purpose:** The "do nothing" operation that passes data through unchanged.

**Example:**
```python
identity = Id(Obj("Tensor"))  # Tensor → Tensor
```

---

### `Box`

```python
@dataclass(frozen=True)
class Box(Expr):
    """Generator: a primitive operation."""
    op: str
    attrs: tuple[tuple[str, Any], ...] = ()
```

**Purpose:** Represents a primitive operation (must be registered in Signature).

**Class Methods:**

| Method | Description |
|--------|-------------|
| `Box.with_attrs(op, **attrs)` | Create box with hashable attributes |

**Example:**
```python
# Simple box
linear = Box("Linear")

# Box with attributes
lora = Box.with_attrs("LoRA", rank=16, alpha=1.0)
```

---

### `Seq`

```python
@dataclass(frozen=True)
class Seq(Expr):
    """Sequential composition: first ; second"""
    first: Expr
    second: Expr
```

**Purpose:** Compose operations in sequence. Output of `first` feeds into `second`.

**Type Rule:** If `first: A → B` and `second: B → C`, then `Seq(first, second): A → C`.

**Example:**
```python
pipeline = Seq(Box("Encode"), Box("Decode"))
```

---

### `Par`

```python
@dataclass(frozen=True)
class Par(Expr):
    """Parallel composition: left ⊗ right"""
    left: Expr
    right: Expr
```

**Purpose:** Execute operations in parallel (no data dependency).

**Type Rule:** If `left: A → B` and `right: C → D`, then `Par(left, right): (A ⊗ C) → (B ⊗ D)`.

**Example:**
```python
parallel = Par(Box("EncodeAudio"), Box("EncodeVideo"))
```

---

### `pretty`

```python
def pretty(e: Expr) -> str:
    """Human-friendly string representation."""
```

**Example:**
```python
expr = Seq(Box("f"), Seq(Box("g"), Box("h")))
print(pretty(expr))  # (f ; (g ; h))
```

---

### `normalize`

```python
def normalize(e: Expr) -> Expr:
    """Coherence discipline: canonical form."""
```

**Purpose:** Applies normalization rules:
- Removes identity compositions (`Id ; f` → `f`, `f ; Id` → `f`)
- Right-associates sequential composition

**Example:**
```python
messy = Seq(Seq(Box("a"), Box("b")), Box("c"))
clean = normalize(messy)  # Right-associated form
```

---

### `infer_type`

```python
def infer_type(e: Expr, sig: Signature) -> Sort:
    """Infer the (domain, codomain) of an expression."""
```

**Raises:** `TypeError` if composition is ill-typed.

**Example:**
```python
dom, cod = infer_type(pipeline, sig)
print(f"{dom} → {cod}")
```

---

## Rewrite Module

**Import:** `from tensorgraph import Pattern, PVar, PId, PBox, PSeq, PPar, Rewrite, ematch`

### Pattern Classes

| Class | Purpose |
|-------|---------|
| `PVar(name, sort?)` | Match any expression (optionally with type constraint) |
| `PId(obj)` | Match identity on `obj` (can be `ObjVar`) |
| `PBox(op, attrs?)` | Match box with given op (optionally exact attrs) |
| `PSeq(a, b)` | Match sequential composition |
| `PPar(l, r)` | Match parallel composition |

---

### `Rewrite`

```python
@dataclass(frozen=True)
class Rewrite:
    """A typed rewrite law."""
    name: str
    lhs: Pattern
    rhs: Pattern | Callable[[EGraph, int, Subst, ObjSubst], int]
```

**Purpose:** Defines an equivalence between diagram patterns.

**RHS Options:**
- **Pattern:** Instantiated from matched variables
- **Callable:** Builder function for complex transformations

**Example (Pattern RHS):**
```python
# Associativity: (a;b);c ≡ a;(b;c)
assoc = Rewrite(
    name="Assoc",
    lhs=PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
    rhs=PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
)
```

**Example (Builder RHS):**
```python
def fuse_builder(eg, root, env, oenv):
    # Custom logic to build RHS
    return eg.add_expr(Box("Fused"))

rule = Rewrite(name="Fuse", lhs=PSeq(PVar("x"), PVar("y")), rhs=fuse_builder)
```

---

### `ematch`

```python
def ematch(eg: EGraph, pat: Pattern) -> list[tuple[int, Subst, ObjSubst]]:
    """Find all matches of pattern in e-graph."""
```

**Returns:** List of `(eclass_id, expr_env, obj_env)` tuples.

**Example:**
```python
matches = ematch(eg, PSeq(PVar("x"), PVar("y")))
for eclass, env, oenv in matches:
    print(f"Match at e-class {eclass}: {env}")
```

---

## E-Graph Module

**Import:** `from tensorgraph import EGraph, ENode, UnionFind`

### `EGraph`

```python
class EGraph:
    """A typed e-graph with sort-preserving operations."""
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `sig` | `Signature` | The operation registry |
| `uf` | `UnionFind` | Equivalence class management |
| `sort` | `dict[int, Sort]` | E-class ID → type |
| `nodes` | `dict[int, set[ENode]]` | E-class ID → contained nodes |
| `memo` | `dict[ENode, int]` | Hashcons: node → e-class ID |
| `root` | `int \| None` | Root e-class (optional) |
| `merge_log` | `list[tuple[str, int, int]]` | Merge history |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_expr` | `(e: Expr) → int` | Add expression, return e-class ID |
| `add_enode` | `(en: ENode, sort: Sort) → int` | Add raw e-node |
| `merge` | `(a: int, b: int, reason: str) → int` | Merge e-classes, return new rep |
| `rebuild` | `() → None` | Restore congruence invariants |

**Example:**
```python
eg = EGraph(sig)
root = eg.add_expr(my_expression)
eg.merge(class1, class2, reason="MyRule")
eg.rebuild()
```

---

### `ENode`

```python
@dataclass(frozen=True)
class ENode:
    """A node in the e-graph."""
    tag: str           # "Id", "Box", "Seq", "Par"
    data: tuple        # Tag-specific data
    children: tuple[int, ...]  # E-class IDs of children
```

---

### `UnionFind`

```python
class UnionFind:
    """Disjoint set data structure with path compression."""
```

**Methods:**

| Method | Description |
|--------|-------------|
| `make()` | Create new singleton set, return ID |
| `find(x)` | Find representative of set containing x |
| `union(a, b)` | Merge sets, return new representative |

---

## Saturation

**Import:** `from tensorgraph import saturate`

### `saturate`

```python
def saturate(
    eg: EGraph,
    rewrites: Sequence[Rewrite],
    iters: int = 8,
    max_applications: int = 10_000,
    trace: Trace | None = None,
) -> None:
    """Equality saturation loop."""
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eg` | required | E-graph to saturate |
| `rewrites` | required | Rewrite rules to apply |
| `iters` | `8` | Maximum iterations |
| `max_applications` | `10_000` | Max rewrites per iteration |
| `trace` | `None` | Optional trace for recording |

**Example:**
```python
trace = Trace()
saturate(eg, [rule1, rule2], iters=10, trace=trace)
```

---

## Extraction

**Import:** `from tensorgraph import Extractor`

### `Extractor`

```python
class Extractor:
    """Cost-based extractor for e-graph equivalence classes."""
```

**Constructor:**
```python
def __init__(self, eg: EGraph, local_cost=default_cost):
    """Create extractor with optional custom cost function."""
```

**Methods:**

| Method | Description |
|--------|-------------|
| `solve(root, max_rounds=50)` | Compute optimal costs |
| `extract(root)` | Return lowest-cost expression |

**Custom Cost Functions:**
```python
def my_cost(enode: ENode) -> int:
    """Return non-negative cost for this node."""
    if enode.tag == "Box":
        return 1
    return 0

extractor = Extractor(eg, local_cost=my_cost)
```

---

### `default_cost`

```python
def default_cost(en: ENode) -> int:
    """Default: Box costs 1, others cost 0."""
```

---

## Tracing

**Import:** `from tensorgraph import Trace, TraceEntry`

### `Trace`

```python
@dataclass
class Trace:
    """Collection of trace entries with inspection utilities."""
    entries: list[TraceEntry]
    enabled: bool = True
```

**Methods:**

| Method | Description |
|--------|-------------|
| `record(...)` | Add a trace entry |
| `clear()` | Remove all entries |
| `filter_by_rule(name)` | Entries matching rule name |
| `filter_by_eclass(id)` | Entries involving e-class |
| `summary()` | `{rule_name: count}` dict |
| `dump(max_entries=None)` | Human-readable string |

**Example:**
```python
trace = Trace()
saturate(eg, rules, trace=trace)

print(trace.summary())
print(trace.dump(max_entries=10))
```

---

### `TraceEntry`

```python
@dataclass
class TraceEntry:
    """A single rewrite application record."""
    rule_name: str
    root_eclass: int
    rhs_eclass: int
    merged_from: int
    merged_to: int
    expr_env: dict[str, int]   # Pattern var → e-class
    obj_env: dict[str, Obj]    # Object var → Obj
```

---

## Adjunctions

**Import:** `from tensorgraph import Adjunction`

### `Adjunction`

```python
@dataclass(frozen=True)
class Adjunction:
    """Operational adjunction f ⊣ g for mate synthesis."""
    f_lower: Box
    g_lift: Box
```

**Purpose:** Given a commuting square `f;u ≡ v;f`, synthesize the mate `u ≡ f;v;g`.

**Methods:**

| Method | Description |
|--------|-------------|
| `mate_left_to_right(alpha)` | Synthesize mate from commuting rewrite |

**Example:**
```python
adj = Adjunction(f_lower=Box("Lower"), g_lift=Box("Lift"))

alpha = Rewrite(
    name="Commute",
    lhs=PSeq(PBox("Lower"), PBox("OptA")),
    rhs=PSeq(PBox("OptB"), PBox("Lower")),
)

mate = adj.mate_left_to_right(alpha)
# mate.lhs = PBox("OptA")
# mate.rhs = PSeq(Lower, PSeq(OptB, Lift))
```

---

## 11. Backends

**Import:** `from tensorgraph.backends.fx import ...`
**Import:** `from tensorgraph.codegen.triton import TritonEmitter`
**Import:** `from tensorgraph.codegen.cuda import CUDAEmitter`

### `TritonEmitter`

```python
class TritonEmitter:
    """Generates OpenAI Triton kernels from diagrammatic expressions."""
```

**Methods:**

| Method | Description |
|--------|-------------|
| `emit(expr, kernel_name)` | Generate Triton Python code |

**Features:**
- Automatic fusion of `Seq` and `Par` groups.
- Support for `elementwise` and `reduction` traits.
- Multi-variable input/output handling.

---

### `CUDAEmitter`

```python
class CUDAEmitter:
    """Generates raw CUDA C++ kernels from diagrammatic expressions."""
```

**Purpose:** Provides a fallback codegen path for non-Triton environments.

**Methods:**

| Method | Description |
|--------|-------------|
| `emit(expr, kernel_name)` | Generate C++/CUDA kernel and host launcher |

---

## 12. Distributed Fabric

**Import:** `from tensorgraph.dist.fabric import Fabric, create_fabric, AsyncFabric`
**Import:** `from tensorgraph.dist.sharding import Shard`

### `Fabric` (Protocol)

```python
class Fabric(Protocol):
    """Abstract protocol for inter-shard communication."""
```

### `AsyncFabric`

```python
class AsyncFabric:
    """Production-ready asynchronous fabric implementation."""
```

**Methods:**

| Method | Description |
|--------|-------------|
| `register(shard)` | Attach a shard to the fabric |
| `start()` | Launch background dispatcher/worker threads |
| `stop()` | Graceful shutdown of fabric threads |
| `get_stats()` | Return message throughput and error stats |

---

## 13. Performance Regression

**Import:** `from tensorgraph.benchmarks.regression import benchmark, run_suite`

### `BenchmarkResult`

```python
@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    # ... min/max/iterations
```

### `run_suite`

```python
def run_suite() -> BenchmarkSuite:
    """Execution all registered benchmarks."""
```

---

## Package-Level Exports

All commonly-used symbols are available at the package level:

```python
from tensorgraph import (
    # Types
    Obj, ObjVar, ObjLike, Sort,
    
    # Signature
    Signature, OpDef,
    
    # IR
    Expr, Id, Box, Seq, Par, pretty, normalize, infer_type,
    
    # Patterns
    Pattern, PVar, PId, PBox, PSeq, PPar,
    
    # Rewriting
    Rewrite, ematch,
    
    # E-Graph
    EGraph, ENode, UnionFind,
    
    # Saturation & Extraction
    saturate, Extractor,
    
    # Tracing
    Trace, TraceEntry,
    
    # Adjunctions
    Adjunction,
)
```

---

*Documentation by Grand Challenge Technologies Ltd. — Frontier Engineering*
