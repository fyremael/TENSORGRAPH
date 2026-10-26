# WP03-B residual and weighted RMSNorm experiment

Standalone performance tuning is shelved. The correct native backend is retained.
See [Decoder performance direction](DECODER_PERFORMANCE_DIRECTION.md) for the
WP03-C outcome and the active residual/RMSNorm/QKV projection target.

Experimental bounded typed DAG lowering, tracked in issue #11. This adds a
multi-input, row-reduction path alongside the existing unary backend. It does
not yet feed the general diagram IR or equality-saturation engine.

For each row, all arithmetic is FP32:

    s = float(x) + float(residual)
    inverse = rsqrt(mean(s * s) + 1e-5)
    residual_out = cast(s)
    normalized_out = cast((s * inverse) * float(weight))

Supported contracts: contiguous B×H inputs and two disjoint outputs, contiguous
H weight, B positive, H in {1024,4096,8192}, float32/float16/bfloat16.
All five storage byte ranges must be disjoint, on the same CUDA device and dtype.
The public extension validates this ABI and launches on the current stream.
NaN/Inf propagate by IEEE arithmetic. FP32 square overflow yields inverse RMS
zero; finite large values then normalize to zero. Inputs are immutable.

The emitter retains the typed nodes, graph hash, exact generated translation unit
and source hash. Compilation checks the source hash and disables FMA contraction.
No fallback is allowed. Source generation and compilation are timed separately;
saturation is explicitly unavailable and extraction is identity.

Run `python -m tensorgraph.benchmarks.decode --gpu A100 --output /outside/repo/results`
on the declared Torch 2.11.0+cu130/CUDA 13.0/NVCC 13.0.88/Triton 3.6.0 stack.
The orchestration script `scripts/run_decode_colab.py` executes A100 then L4;
T4 is excluded. It exports process status and evidence before releasing sessions.

The matrix is B={1,4,16}, H={1024,4096,8192}, three dtypes. Each backend uses the
same resident input/weight and caller-owned persistent output addresses. Eager
also preallocates its FP32 workspaces. Inductor receives explicit output buffers;
its internal allocations are recorded with a memory profiler trace and peak
allocator statistics. An independent handwritten CUB/CUDA baseline is labeled
separately from generated TENSORGRAPH. All four paths undergo the same numerical
and changed-input graph replay checks.

There are three distinct timing contracts: resident ordinary launches, whole
fragment graph replay, and two device-to-device input copies plus graph replay.
Weights remain resident. Each has five randomized backend rounds of 1000 raw
samples. Per-sample device events and synchronized host latency are retained.
These synchronized samples characterize single decode-step latency, not saturated
throughput. Output publication is identical (resident outputs) for every path.

Numerical tolerances are frozen in the benchmark before execution: F32 2e-5,
F16 .002, BF16 .02 (absolute and relative). Both FP64 oracle and staged FP32
references are checked, including zero, nearzero, cancellation, large finite,
nonfinite and overflow cases. Each graph checks 100 changed-input replays with
stable pointers and unchanged inputs/weights. Native ABI alias rejection includes
partially overlapping contiguous views. Failed numerical values are retained.

No result is a full decoder or next-token throughput claim. Comparisons requiring
allocation-free internals must consult the allocation traces before admitting
Inductor. GPU execution and review are required before any capability promotion.
