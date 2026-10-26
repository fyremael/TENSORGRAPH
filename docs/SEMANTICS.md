# TENSORGRAPH semantics

## Core category

The core IR is interpreted as typed morphism syntax with sequential composition, tensor product, identities, and symmetric braiding.

For objects `A`, `B`, and `C`:

- `Id(A) : A → A`;
- `Seq(f, g) : A → C` when `f : A → B` and `g : B → C`;
- `Par(f, g) : A ⊗ C → B ⊗ D` when `f : A → B` and `g : C → D`;
- `Swap(A, B) : A ⊗ B → B ⊗ A`.

The implementation represents tensor products as binary object trees. It does not silently identify differently associated trees unless a specific normalization or coherence rule does so.

## Cartesian structure

`Dup(A) : A → A ⊗ A` and `Del(A) : A → I` introduce copying and deletion. Their naturality equations are sound for pure deterministic morphisms:

```text
f ; Del(B) = Del(A)
f ; Dup(B) = Dup(A) ; (f ⊗ f)
```

They are not sound for arbitrary stateful, random, I/O, mutating, or otherwise effectful operations. TENSORGRAPH therefore treats a signature operation as pure unless it carries the `effectful` trait. The e-graph does not apply copy/delete naturality across an e-class that may contain an effectful operation.

This purity convention is a bounded research contract. A future effect system may replace it with explicit effect rows or graded morphisms.

## Sum and case

`Case(left, right)` is admitted only when:

- `left : I → B`;
- `right : A → B`.

Its type is `(I + A) → B`. Both the ordinary type inference path and direct e-graph admission enforce the same conditions.

## Iteration

`Iter(body, count)` is admitted only when:

- `count` is an integer and not a Boolean;
- `count ≥ 0`;
- `body : A → A`.

The count is static data in the IR. This primitive does not claim general dynamic-loop semantics.

## Equality and rewriting

A `Rewrite` introduces an equation only within its declared typed pattern scope. Equality saturation records alternatives; extraction selects a representative according to a cost model. Extraction is not a proof that an undeclared semantic equation is valid.

Every domain rule must identify its semantic origin. Rules whose validity depends on numerical restrictions, purity, shapes, dtypes, layouts, or devices must encode or externally gate those restrictions.

## Adjunctions

General adjunction mate constructions require unit and counit data and triangle identities. Convenience transformations that amount to conjugation must not be described as valid for an arbitrary adjunction. The existing adjunction helper remains experimental until its contract is rebuilt around explicit evidence.

## Backend boundary

A backend claim requires execution of code generated from the exact admitted and extracted IR. Capturing a PyTorch graph, emitting a source string, or executing a separate handwritten kernel does not establish backend correctness.
