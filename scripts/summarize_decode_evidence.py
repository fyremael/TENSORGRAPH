"""Validate retained WP03-B archives and derive per-cell latency summaries."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import zipfile
from pathlib import Path

BACKENDS = ("native", "direct_cuda", "eager", "inductor")
MODES = ("resident", "graph", "copy_graph")


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(path):
    with zipfile.ZipFile(path) as archive:
        process = json.loads(archive.read("process-status.json"))
        doc = json.loads(archive.read("results/evidence.json"))
        assert process["returncode"] == 0 and doc["passed"] is True
        assert not doc["dirty"] and len(doc["cells"]) == 27
        assert doc["torch"] == "2.11.0+cu130" and doc["cuda"] == "13.0"
        assert "V13.0.88" in doc["nvcc"] and doc["triton"] == "3.6.0"
        assert not doc["failures"] and process["commit"] == doc["commit"]
        keys = {(c["dtype"], c["batch"], c["hidden"]) for c in doc["cells"]}
        assert keys == set(
            itertools.product(("float32", "float16", "bfloat16"), (1, 4, 16), (1024, 4096, 8192))
        )
        result = {
            "gpu": doc["gpu"],
            "commit": doc["commit"],
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "evidence_sha256": hashlib.sha256(archive.read("results/evidence.json")).hexdigest(),
            "repetitions": doc["repetitions"],
            "rounds": doc["rounds"],
            "cells": [],
        }
        for cell in doc["cells"]:
            assert cell["completed"] and cell["nonfinite_and_alias_gates_passed"]
            assert cell["changed_input_replays_per_backend"] >= 100
            artifact = json.loads(
                archive.read(f"results/build-{cell['dtype']}-{cell['hidden']}/artifact.json")
            )
            assert (
                hashlib.sha256(artifact["source"].encode()).hexdigest()
                == artifact["source_sha256"]
                == cell["source_sha256"]
            )
            seen = {(r["backend"], r["mode"], r["round"]) for r in cell["records"]}
            assert seen == set(itertools.product(BACKENDS, MODES, range(doc["rounds"])))
            assert len(cell["records"]) == len(seen)
            item = {
                "dtype": cell["dtype"],
                "batch": cell["batch"],
                "hidden": cell["hidden"],
                "allocation": {},
                "timing": {},
            }
            for name in BACKENDS:
                allocation = cell["allocation"][name]
                trace = json.loads(archive.read("results/" + allocation["trace"]))
                events = trace["traceEvents"]
                cuda_allocs = [
                    e
                    for e in events
                    if e.get("name") == "[memory]"
                    and e.get("args", {}).get("Device Type") in (1, "CUDA")
                    and e.get("args", {}).get("Bytes", 0) > 0
                ]
                item["allocation"][name] = {
                    "peak_extra_bytes": allocation["peak_extra_bytes"],
                    "positive_cuda_allocation_events": len(cuda_allocs),
                    "positive_cuda_allocation_bytes": sum(e["args"]["Bytes"] for e in cuda_allocs),
                }
                item["timing"][name] = {}
                for mode in MODES:
                    records = [
                        r for r in cell["records"] if r["backend"] == name and r["mode"] == mode
                    ]
                    assert all(
                        len(r["device_us"]) == len(r["synchronized_host_ns"]) == doc["repetitions"]
                        for r in records
                    )
                    device = [v for r in records for v in r["device_us"]]
                    host = [v / 1000 for r in records for v in r["synchronized_host_ns"]]
                    assert all(math.isfinite(v) and v > 0 for v in device + host)
                    item["timing"][name][mode] = {
                        "device_median_us": statistics.median(device),
                        "device_p95_us": percentile(device, 0.95),
                        "device_p99_us": percentile(device, 0.99),
                        "host_median_us": statistics.median(host),
                        "samples": len(device),
                    }
            result["cells"].append(item)
        result["aggregate"] = {}
        for mode in MODES:
            result["aggregate"][mode] = {}
            for name in BACKENDS:
                result["aggregate"][mode][name] = {
                    "median_of_cell_medians_us": statistics.median(
                        c["timing"][name][mode]["device_median_us"] for c in result["cells"]
                    ),
                    "paired_native_speedup_median": statistics.median(
                        c["timing"][name][mode]["device_median_us"]
                        / c["timing"]["native"][mode]["device_median_us"]
                        for c in result["cells"]
                    ),
                    "host_median_of_cell_medians_us": statistics.median(
                        c["timing"][name][mode]["host_median_us"] for c in result["cells"]
                    ),
                }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps([summarize(p) for p in args.archives], indent=2))
