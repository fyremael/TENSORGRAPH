# Evidence admission policy

This policy governs benchmark, compliance, audit, and maturity claims in TENSORGRAPH.

## Required evidence record

Every admitted run must record:

1. repository and exact commit SHA;
2. dirty-worktree state;
3. UTC execution time;
4. complete command line;
5. Python, PyTorch, Triton, CUDA, driver, and operating-system versions where applicable;
6. CPU and GPU identity;
7. workload definition and tensor metadata;
8. random seed;
9. warm-up and repetition counts;
10. raw timing samples;
11. separately measured capture, saturation, extraction, source generation, compilation, and execution phases;
12. generated source SHA-256;
13. reference and candidate numerical outputs or sufficient error statistics;
14. pass/fail thresholds declared before result interpretation.

JSON is the normative evidence format. Markdown summaries are derived views and must identify the JSON source.

## Prohibited substitutions

The following are not evidence of TENSORGRAPH performance or correctness:

- assigning candidate latency as a multiplier of another compiler's latency;
- assigning candidate memory as a multiplier of another runtime's memory;
- comparing PyTorch eager with PyTorch FX or Inductor while labeling the result TENSORGRAPH;
- executing a handwritten kernel instead of the kernel emitted from the tested TENSORGRAPH expression;
- omitting failed or outlier samples without a declared rule;
- presenting source-string substring checks as compiled execution;
- using machine-local `file://` links in committed reports;
- declaring an audit independent when it is generated solely by repository-owned test code without an identified external reviewer.

## Benchmark phases

Benchmarks must report these phases separately:

- frontend capture;
- IR construction and type admission;
- saturation;
- extraction;
- source generation;
- JIT or ahead-of-time compilation;
- first execution;
- steady-state execution.

A combined end-to-end number may be reported only in addition to the phase measurements.

## Numerical gates

The benchmark must state dtype-specific tolerances. It must fail closed when candidate execution did not occur. A source-generation success is not a numerical pass.

## Historical reports

Reports produced before this policy are classified as `quarantined` unless they already contain all required evidence. They may be retained for provenance, but they cannot support README, release, or production claims.

## Promotion

A capability may move from `experimental` to `bounded-experimental` or `supported-research` only when:

- the implementation scope is explicit;
- positive and adversarial tests are committed;
- required CI succeeds on the same revision;
- any hardware-dependent claim has an admitted evidence record;
- `STATUS.md` is updated in the promoting change.
