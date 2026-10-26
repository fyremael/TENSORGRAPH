# Decoder performance direction

## Decision: shelve standalone RMSNorm tuning

User decision, 2026-10-26: retain the correct generated native backend and stop
standalone RMSNorm optimization campaigns. Existing implementation and evidence
are preserved; no further standalone GPU sweeps are planned.

WP03-C executed source 8cbfa070096448519a289dc162cac8c0b47a33dd.
All 27 cases on A100 and 27 on L4 passed numerical execution. Both archives
passed CRC checks, and separate session readback confirmed no active runtimes.
T4 was excluded.

Median across case medians of the amortized 1000-invocation captured batches:

| Backend | A100 microseconds | L4 microseconds |
|---|---:|---:|
| Generated native CUDA | 8.376 | 4.917 |
| Independent direct CUDA | 8.718 | 7.087 |
| Inductor | 3.442 | 2.546 |
| Eager | 32.925 | 22.460 |

These are repeated-fragment captured-batch averages, not single-token latency.
Inductor still allocated one temporary in 18/27 profiled ordinary calls per GPU,
up to 524288 bytes. The explicit workspace change did not establish full ordinary
allocation compliance. Those ordinary comparisons remain excluded.

Raw exports are retained on CHAD at
`/home/jim/gcl-research/wp03c-artifacts/{A100,L4}/export.zip`.
SHA-256: A100 d49e430a367bb0b729655ff582d733cb681c80c361a26ffa9a7dfd1cd2f2b16a;
L4 30c7ec411ce17f2a7f72091c4914aa6b9f9f35e4244c518dba11921370e8f7ab.

## Active performance target: residual + RMSNorm + packed QKV projection

Start with one pre-attention decoder subgraph:

    s = FP32(x) + FP32(residual)
    n = cast_dtype(s * rsqrt(mean(s*s) + 1e-5) * FP32(weight))
    residual_out = cast_dtype(s)
    QKV = n @ Wqkv

`x` and `residual` are [B,H], weight is [H], and the packed projection weight is
[H,3H]. This initial equal-width Q/K/V contract describes multi-head attention;
it does not silently claim grouped-query attention support. Residual output,
normalized storage, and packed QKV output persist for every lane. All lanes must
observe the same normalization storage rounding before projection.

`tensorgraph/benchmarks/decode_projection.py` now supplies the initial persistent
lane setup. The native lane deliberately composes the existing generated native
normalization with a library matrix multiply. It must be labeled mixed backend,
not an all-generated native projection kernel. The eager baseline uses the same
library multiply; Inductor compiles the whole declared subgraph. Numerical GPU validation subsequently passed the bounded A100/L4 matrix; see
[evidence/qkv-validation](../evidence/qkv-validation/README.md). No performance
claim is made for this new setup.

Initial bounded matrix: B={1,4}, H={1024,4096}, float16/bfloat16, A100 and L4.
Keep weights resident, allocate outputs/workspaces before timing, and audit actual
allocations rather than assuming caller buffers eliminate compiler temporaries.
Charge the same two input copies when measuring updated-input execution. Record
ordinary host latency, full-subgraph graph replay, and batched device intervals
separately. Do not conflate batch amortization with interactive token latency.

Before GPU timing, validate all three outputs against a staged reference with the
explicit normalization rounding boundary, test changed inputs and persistent
addresses, and fix projection precision settings (including TF32) across lanes.
Retain compiler source, stack, source hashes, raw samples and failed cases.

Compare normalization-only, projection-only, and whole-subgraph measurements in
that experiment as attribution controls, not a renewed standalone tuning sweep.
Proceed to attention/KV-cache work only after this boundary is understood.
