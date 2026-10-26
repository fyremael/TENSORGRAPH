# Historical master compiler benchmark — quarantined

**Evidence status:** `quarantined`  
**Authority:** none for performance, numerical parity, or compiler comparisons.

The prior report presented TENSORGRAPH latency and memory values that were derived from PyTorch Inductor measurements rather than measured by executing generated TENSORGRAPH code. Its numerical check compared PyTorch paths and its chart used a machine-local file reference.

Those results are withdrawn. Use `benchmarks/bench_verified_elementwise.py` to produce a clean-commit JSON record under [docs/EVIDENCE_POLICY.md](docs/EVIDENCE_POLICY.md).
