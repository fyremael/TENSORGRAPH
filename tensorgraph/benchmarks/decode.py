"""WP03-B persistent residual/RMSNorm experiment. No T4 allocations."""

# All loop-local callables are consumed before their owning iteration ends.
# ruff: noqa: B023
import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    import torch

    from tensorgraph.codegen.decode import DecodeGraph, compile_decode

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=["A100", "L4"], required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1000 or args.rounds < 5:
        parser.error("at least 1000 repetitions and five rounds are required")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    def shell(*cmd):
        return subprocess.check_output(cmd, text=True).strip()

    nvcc = shell("nvcc", "--version")
    device = torch.cuda.get_device_name()
    assert args.gpu in device, (args.gpu, device)
    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    assert torch.version.cuda == "13.0" and "V13.0.88" in nvcc, nvcc
    import triton

    assert triton.__version__ == "3.6.0", triton.__version__
    torch.manual_seed(731)
    random.seed(731)
    doc = {
        "schema": "TG-WP03-B/1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": shell("git", "rev-parse", "HEAD"),
        "dirty": bool(shell("git", "status", "--porcelain")),
        "command": sys.argv,
        "python": sys.version,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nvcc": nvcc,
        "ninja": shell("ninja", "--version"),
        "cxx": shell("c++", "--version"),
        "gpu": device,
        "driver": shell("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "seed": 731,
        "warmup": 20,
        "repetitions": args.repetitions,
        "rounds": args.rounds,
        "tolerance": {"float32": [2e-5, 2e-5], "float16": [0.002, 0.002], "bfloat16": [0.02, 0.02]},
        "phases": {
            "frontend_capture": "not applicable: directly constructed typed DAG",
            "saturation": "not implemented for this bounded IR",
            "extraction": "identity",
        },
        "allocation_contract": "resident immutable inputs; disjoint caller-owned persistent outputs; preallocated eager workspaces; Inductor internal temporaries measured separately",
        "limitations": [
            "no general IR saturation or full decoder claim",
            "Inductor may allocate internal temporaries; allocation traces govern eligibility for allocation-free comparison",
        ],
        "cells": [],
        "failures": [],
    }

    def save():
        (out / "evidence.json").write_text(json.dumps(doc, indent=2, allow_nan=False))

    def checked(fn, x, r, w, a, b, label):
        before = [t.clone() for t in (x, r, w)]
        fn()
        s = x.double() + r.double()
        oracle = (s, s * torch.rsqrt((s * s).mean(-1, keepdim=True) + 1e-5) * w.double())
        sf = x.float() + r.float()
        staged = (sf, sf * torch.rsqrt((sf * sf).mean(-1, keepdim=True) + 1e-5) * w.float())
        atol, rtol = doc["tolerance"][str(x.dtype).split(".")[-1]]
        metrics = []
        for got, ref, fp in zip((a, b), oracle, staged, strict=False):
            expected = ref.to(got.dtype)
            err = (got.double() - expected.double()).abs()
            metrics.append(
                {
                    "max_abs": float(err.max()),
                    "max_rel": float((err / expected.double().abs().clamp_min(atol)).max()),
                }
            )
            try:
                torch.testing.assert_close(got, expected, atol=atol, rtol=rtol, msg=label)
                torch.testing.assert_close(got, fp.to(got.dtype), atol=atol, rtol=rtol, msg=label)
            except AssertionError:
                torch.save(
                    {
                        "x": x.cpu(),
                        "residual": r.cpu(),
                        "weight": w.cpu(),
                        "residual_out": a.cpu(),
                        "normalized_out": b.cpu(),
                        "expected": expected.cpu(),
                        "staged": fp.cpu(),
                    },
                    out / "failed-values.pt",
                )
                doc["failures"].append({"label": label, "metrics": metrics})
                save()
                raise
        for t, old in zip((x, r, w), before, strict=False):
            assert torch.equal(t, old), "input modified"
        return metrics

    from torch.utils.cpp_extension import load_inline

    direct_source = Path(__file__).with_name("decode_direct.cu").read_text()
    direct_dir = out / "direct-build"
    direct_dir.mkdir()
    (direct_dir / "source.cu").write_text(direct_source)
    doc["direct_source_sha256"] = hashlib.sha256(direct_source.encode()).hexdigest()
    t = time.perf_counter_ns()
    direct = load_inline(
        name="tg_decode_direct_" + doc["direct_source_sha256"][:12],
        cpp_sources="void direct_out(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);",
        cuda_sources=direct_source,
        functions=["direct_out"],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        build_directory=str(direct_dir),
        verbose=True,
    )
    doc["direct_compile_ns"] = time.perf_counter_ns() - t
    for dtype_name in ("float32", "float16", "bfloat16"):
        dtype = getattr(torch, dtype_name)
        for h in (1024, 4096, 8192):
            t = time.perf_counter_ns()
            graph = DecodeGraph(dtype_name, h)
            artifact = graph.emit()
            generation = time.perf_counter_ns() - t
            build = out / f"build-{dtype_name}-{h}"
            build.mkdir()
            (build / "artifact.json").write_text(json.dumps(artifact, indent=2))
            t = time.perf_counter_ns()
            module = compile_decode(artifact, build)
            compile_ns = time.perf_counter_ns() - t
            for batch in (1, 4, 16):
                cell = {
                    "dtype": dtype_name,
                    "batch": batch,
                    "hidden": h,
                    "source_sha256": artifact["source_sha256"],
                    "graph_sha256": artifact["graph_sha256"],
                    "generation_ns": generation,
                    "compile_load_ns": compile_ns,
                    "records": [],
                    "numerical": [],
                    "allocation": {},
                    "captures": {},
                }
                doc["cells"].append(cell)
                x = torch.randn(batch, h, device="cuda", dtype=dtype)
                r = torch.randn_like(x)
                w = torch.randn(h, device="cuda", dtype=dtype)
                a = torch.empty_like(x)
                b = torch.empty_like(x)
                sx = torch.empty_like(x, dtype=torch.float32)
                sr = torch.empty_like(sx)
                ww = torch.empty_like(w, dtype=torch.float32)
                sq = torch.empty_like(sx)
                inv = torch.empty(batch, 1, device="cuda")
                tmp = torch.empty_like(sx)

                def eager():
                    sx.copy_(x)
                    sr.copy_(r)
                    ww.copy_(w)
                    torch.add(sx, sr, out=sx)
                    torch.mul(sx, sx, out=sq)
                    torch.mean(sq, dim=-1, keepdim=True, out=inv)
                    inv.add_(1e-5)
                    torch.rsqrt(inv, out=inv)
                    torch.mul(sx, inv, out=tmp)
                    torch.mul(tmp, ww, out=tmp)
                    a.copy_(sx)
                    b.copy_(tmp)

                def functional(x, r, w, a, b):
                    s = x.float() + r.float()
                    n = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + 1e-5) * w.float()
                    a.copy_(s)
                    b.copy_(n)

                torch._dynamo.reset()
                t = time.perf_counter_ns()
                compiled = torch.compile(functional, fullgraph=True)
                compiled(x, r, w, a, b)
                torch.cuda.synchronize()
                cell["inductor_compile_first_ns"] = time.perf_counter_ns() - t
                backends = {
                    "native": lambda: module.run_out(x, r, w, a, b),
                    "eager": eager,
                    "direct_cuda": lambda: direct.direct_out(x, r, w, a, b),
                    "inductor": lambda: compiled(x, r, w, a, b),
                }
                # Warm and preserve first-execution separately from steady state.
                captures = {}
                pointers = [t.data_ptr() for t in (x, r, w, a, b)]
                for name, fn in backends.items():
                    t = time.perf_counter_ns()
                    fn()
                    torch.cuda.synchronize()
                    cell.setdefault("first_execution_ns", {})[name] = time.perf_counter_ns() - t
                    for _ in range(20):
                        fn()
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        profile_memory=True,
                    ) as prof:
                        fn()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(
                        str(out / f"alloc-{dtype_name}-{h}-{batch}-{name}.json")
                    )
                    cell["allocation"][name] = {
                        "base_bytes": base,
                        "peak_extra_bytes": torch.cuda.max_memory_allocated() - base,
                        "trace": f"alloc-{dtype_name}-{h}-{batch}-{name}.json",
                    }
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    t = time.perf_counter_ns()
                    with torch.cuda.graph(g, stream=stream):
                        fn()
                    torch.cuda.synchronize()
                    cell["captures"][name] = {
                        "capture_ns": time.perf_counter_ns() - t,
                        "reserved_bytes": torch.cuda.memory_reserved(),
                    }
                    captures[name] = g
                    # Explicit special distributions, then 100 changing-input replay checks.
                    for case in ("normal", "zero", "nearzero", "cancellation", "large"):
                        x.normal_()
                        r.normal_()
                        w.normal_()
                        if case == "zero":
                            x.zero_()
                            r.zero_()
                        if case == "nearzero":
                            x.mul_(1e-7)
                            r.mul_(1e-7)
                        if case == "cancellation":
                            r.copy_(-x)
                        if case == "large":
                            x.fill_(min(1e10, torch.finfo(dtype).max / 8))
                            r.copy_(x)
                        metrics = checked(g.replay, x, r, w, a, b, f"{name}:{case}")
                        cell["numerical"].append(
                            {"backend": name, "case": case, "metrics": metrics}
                        )
                    for i in range(100):
                        x.normal_()
                        r.normal_()
                        w.normal_()
                        checked(g.replay, x, r, w, a, b, f"{name}:replay{i}")
                        assert pointers == [t.data_ptr() for t in (x, r, w, a, b)]
                    # Reject forbidden overlap in the public ABI, including partial views.
                    if name == "native":
                        for bad in ((x, r, w, x, b), (x, r, w, a, a), (x, x, w, a, b)):
                            try:
                                module.run_out(*bad)
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError("alias accepted")
                # Explicit nonfinite propagation and FP32 square-overflow policy.
                for special in (float("nan"), float("inf")):
                    x.fill_(special)
                    r.zero_()
                    w.fill_(1)
                    for _name, fn in backends.items():
                        fn()
                        sf = x.float() + r.float()
                        expected = (
                            sf * torch.rsqrt((sf * sf).mean(-1, keepdim=True) + 1e-5) * w.float()
                        )
                        torch.testing.assert_close(a, sf.to(dtype), equal_nan=True)
                        torch.testing.assert_close(b, expected.to(dtype), equal_nan=True)
                if dtype_name == "float32":
                    x.fill_(1e30)
                    r.zero_()
                    w.fill_(1)
                    for fn in backends.values():
                        fn()
                        assert torch.count_nonzero(b) == 0
                overlap = torch.empty(batch * h + 1, device="cuda", dtype=dtype)
                bad_x = overlap[:-1].view(batch, h)
                bad_r = overlap[1:].view(batch, h)
                try:
                    module.run_out(bad_x, bad_r, w, a, b)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("partial overlap accepted")
                x.normal_()
                r.normal_()
                w.normal_()
                incoming_x = torch.randn_like(x)
                incoming_r = torch.randn_like(r)
                for mode in ("resident", "graph", "copy_graph"):
                    for round_id in range(args.rounds):
                        order = list(backends)
                        random.shuffle(order)
                        for name in order:
                            fn = backends[name] if mode == "resident" else captures[name].replay

                            def launch():
                                if mode == "copy_graph":
                                    x.copy_(incoming_x)
                                    r.copy_(incoming_r)
                                fn()

                            starts = [
                                torch.cuda.Event(enable_timing=True)
                                for _ in range(args.repetitions)
                            ]
                            ends = [
                                torch.cuda.Event(enable_timing=True)
                                for _ in range(args.repetitions)
                            ]
                            host = []
                            for start, end in zip(starts, ends, strict=False):
                                t = time.perf_counter_ns()
                                start.record()
                                launch()
                                end.record()
                                end.synchronize()
                                host.append(time.perf_counter_ns() - t)
                            cell["records"].append(
                                {
                                    "backend": name,
                                    "mode": mode,
                                    "round": round_id,
                                    "order": order,
                                    "device_us": [
                                        s.elapsed_time(e) * 1000
                                        for s, e in zip(starts, ends, strict=False)
                                    ],
                                    "synchronized_host_ns": host,
                                }
                            )
                save()
                print("CELL PASSED", dtype_name, batch, h, flush=True)
                del captures, backends
                torch.cuda.empty_cache()
    doc["passed"] = len(doc["cells"]) == 27 and not doc["failures"]
    save()


if __name__ == "__main__":
    main()
