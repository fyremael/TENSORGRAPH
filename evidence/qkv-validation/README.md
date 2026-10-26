# Residual / RMSNorm / QKV GPU validation

Executed clean workload commit: `acb3d732e11cc08779a6200685f04325a8c848f8`.

A100 and L4 each passed the full eight-case matrix: B={1,4}, H={1024,4096}, float16/bfloat16. Eager-plus-library, generated-native-normalization-plus-library, and Inductor paths passed all three output checks. T4 was excluded.

There were 48 path checks in total, each including five ordinary distributions and 100 changed-input CUDA graph replays: 240 distribution checks and 4800 replay checks. Inputs, projection weights and buffer addresses remained unchanged by execution. Both archives passed CRC and generated-to-compiled source checks. Archive paths, SHA-256 digests, per-case errors, frozen tolerances and stack metadata are retained in results.json.

The reference uses CPU FP64 normalization, an explicit normalization storage-dtype cast, and CUDA FP32 projection with TF32 disabled. FP16/BF16 reduced-precision reductions are disabled. Projection weights are scaled by 1/sqrt(H). Cases cover normal, zero, near-zero, cancellation and large finite inputs; this report does not claim nonfinite QKV validation.

The native path is mixed backend: generated native normalization followed by a library matrix multiply. This validates the declared larger subgraph numerically; it is not an all-generated GEMM, an allocation-free guarantee, or a performance claim. Ordinary timing and capture allocation audits remain separate work.

| GPU | Cases | Path checks | Changed-input replays |
|---|---:|---:|---:|
| NVIDIA A100-SXM4-40GB | 8 | 24 | 2400 |
| NVIDIA L4 | 8 | 24 | 2400 |
