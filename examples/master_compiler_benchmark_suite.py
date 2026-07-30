"""Retired pre-recovery benchmark entry point.

The former script derived TENSORGRAPH latency and memory values from PyTorch
Inductor measurements and did not execute generated TENSORGRAPH code. Those
results are not evidence.

Use the fail-closed benchmark below on a clean CUDA/Triton checkout:

    python benchmarks/bench_verified_elementwise.py \
        --output artifacts/verified-elementwise.json
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "This benchmark was retired because it used derived candidate values rather "
        "than generated TENSORGRAPH execution. Run "
        "benchmarks/bench_verified_elementwise.py instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
