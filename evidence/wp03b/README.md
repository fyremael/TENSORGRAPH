# WP03-B GPU execution review

The residual update plus weighted RMSNorm implementation ran successfully on A100 and L4. T4 was excluded. All 54 shape/dtype cases passed for all four backends (216 backend checks), including FP64 CPU and staged FP32 references, adversarial distributions, nonfinite/overflow policies, and 100 changing-input replays per backend/case. Native ABI alias rejection also passed. Both runtimes exported their evidence and stopped.

The executed source is `79ae02de0e7da74529c4f93d9029b19e828b176b`, clean on both machines. Later commits add analysis and this review; they are not claimed as the GPU-executed revision.

The normative raw `results/evidence.json` files are retained inside the two archives identified by `review.json`. `summary.json` derives per-case median, p95, p99, synchronized host medians, and allocation counts. `audit.json` records stack identity, distribution error statistics and integrity checks. The raw archives contain 3,240,000 timing samples, generated and compiled sources, extension builds, Inductor caches, and allocator traces.

## Allocation result

The profiled ordinary calls for generated native CUDA, independent direct CUDA, and eager PyTorch had zero additional CUDA allocated bytes and zero positive CUDA allocation events for every case on both GPUs. These three paths satisfy the matched ordinary allocation comparison. All use the same resident inputs/weights and persistent caller-owned output buffers; eager workspaces are preallocated.

Inductor had no timed allocation at H=1024 (nine cases per GPU). At H=4096 and H=8192 it allocated one temporary (8 KiB to 512 KiB depending on dtype/shape). Those 18 ordinary-launch cases per GPU are excluded from matched-allocation performance claims. The unrestricted ordinary Inductor aggregate is retained in JSON for provenance, but must not be advertised as a fair native speedup. Under CUDA graph capture, temporaries reside in captured graph storage and are reused; the two output pointers remain stable across changed-input replay checks.

## Observed graph replay latency

Values below are microseconds: the median across 27 per-case medians; each per-case median uses five rounds of 1,000 samples. They describe the measured single-step event interval. They do not isolate intrinsic GPU computation.

| GPU | Generated native | Direct CUDA | Eager | Inductor |
|---|---:|---:|---:|---:|
| NVIDIA A100-SXM4-40GB | 23.296 | 23.456 | 43.200 | 23.200 |
| NVIDIA L4 | 23.488 | 23.456 | 29.728 | 23.456 |

Native is approximately tied with direct CUDA and Inductor in the measured graph replay path. Differences around one percent are not evidence of kernel superiority. Native/eager paired median speedups are 1.866× on A100 and 1.274× on L4. With the same two input copies before replay, these reduce to 1.298× and 1.090×. Copies and submission overhead substantially reduce the benefit visible to the caller.

## Matched ordinary launches

| GPU | Native event interval (µs) | Direct CUDA (µs) | Eager (µs) | Paired native/eager speedup |
|---|---:|---:|---:|---:|
| NVIDIA A100-SXM4-40GB | 34.496 | 32.416 | 186.112 | 5.400× |
| NVIDIA L4 | 33.984 | 32.512 | 187.680 | 5.543× |

Ordinary launch results include the cost of each backend’s dispatch path. Native performs public ABI validation; the attribution-only direct CUDA wrapper does not perform the same validation. The ordinary timing gap cannot be assigned wholly to GPU computation.

## Interpretation and limits

The experiment establishes correct useful decoder-fragment execution with persistent outputs and bounded multi-input/reduction code generation. The fused candidate avoids eager’s sequence of individual operation launches, while graph replay largely removes the observed dispatch advantage over other fused implementations. This is a concrete improvement over the earlier unary-fragment test, but it does not establish full-model or next-token speedup.

Each sample records CUDA events around a single submitted call and then synchronizes. GPU idle time between host submissions can therefore enter the event interval. Similar A100/L4 medians and convergence of fused paths are consistent with a substantial submission floor; they do not establish equal intrinsic hardware performance. Device-only throughput or kernel-duration claims require an additional batched or internally instrumented graph measurement. No such measurement is claimed here.

The stack was Torch 2.11.0+cu130, CUDA 13.0, NVCC 13.0.88, Triton 3.6.0, driver 580.82.07 and Python 3.13.16. CPU metadata records architecture rather than processor model. All archive members passed CRC checks; native source hashes match the sources retained in the compiled translation units. This is operator-run evidence, not independently certified evidence or a capability promotion.

The implementation remains experimental and separate from the general diagram IR/equality-saturation path. Remaining acceptance work is explicit workspace control for the wider ordinary Inductor baseline, broader negative ABI execution tests, and a measurement that separates device work from host submission gaps. Existing raw results remain unchanged.
