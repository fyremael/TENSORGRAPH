#!/usr/bin/env python3
"""Run the WP03 evidence matrix with deterministic Python hashing.

The matrix derives stable per-cell offsets from tuple hashes. Python selects its
hash secret before startup, so this entrypoint re-executes itself with
``PYTHONHASHSEED=0`` before loading the benchmark. Evidence runs must invoke this
script rather than calling the benchmark module directly.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_HASH_SEED = "0"


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != _HASH_SEED:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = _HASH_SEED
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )

    benchmark = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_native_cuda_inference.py"
    runpy.run_path(str(benchmark), run_name="__main__")


if __name__ == "__main__":
    main()
