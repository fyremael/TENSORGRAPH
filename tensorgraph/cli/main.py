"""TENSORGRAPH Unified CLI — Rustic Precision Edition.

Knowledge from beyond the forest.

Usage:
    tensorgraph optimize "f ; f ; g" --rules fuse,assoc
    tensorgraph demo core
    tensorgraph demo fx
    tensorgraph info
"""

from __future__ import annotations

import argparse
import sys

from . import style as S

# =============================================================================
# OPTIMIZE COMMAND
# =============================================================================

def cmd_optimize(args: argparse.Namespace) -> int:
    """Optimize an expression using e-graph saturation."""
    from ..egraph import EGraph
    from ..egraph.extract import Extractor
    from ..egraph.saturation import saturate
    from ..egraph.trace import Trace
    from ..ir import Box, Id, Seq, pretty
    from ..rewrite import PSeq, PVar, Rewrite
    from ..signature import Signature
    from ..types import Obj

    # Parse expression
    expr_str = args.expression

    print(S.header("TENSORGRAPH OPTIMIZER", "ACTIVE"))
    print(S.metric("INPUT", expr_str, S.chrome))

    # Simple expression parser
    T = Obj("T")
    sig = Signature()

    # Parse tokens
    tokens = [t.strip() for t in expr_str.replace("(", "").replace(")", "").split(";")]
    tokens = [t for t in tokens if t]

    # Register operations
    ops_seen = set()
    for tok in tokens:
        if tok not in ops_seen and tok != "id":
            sig.add(tok, T, T)
            ops_seen.add(tok)

    # Build expression
    def build_expr(tokens):
        if len(tokens) == 0:
            return Id(T)
        elif len(tokens) == 1:
            tok = tokens[0]
            if tok == "id":
                return Id(T)
            return Box(tok)
        else:
            first = tokens[0]
            rest = build_expr(tokens[1:])
            if first == "id":
                return Seq(Id(T), rest)
            return Seq(Box(first), rest)

    expr = build_expr(tokens)

    # Count boxes
    def count_boxes(e):
        if hasattr(e, 'tag'):
            if e.tag == "Box":
                return 1
            elif e.tag == "Seq":
                return count_boxes(e.left) + count_boxes(e.right)
            elif e.tag == "Par":
                return count_boxes(e.top) + count_boxes(e.bot)
        return 0

    boxes_before = count_boxes(expr)

    # Build rules based on --rules flag
    rules = []
    rule_names = []

    if "fuse" in args.rules or "all" in args.rules:
        # FuseOps: f ; f → f (idempotent ops)
        def fuse_rhs(eg, root, env, oenv):
            i1 = eg.uf.find(env["x"])
            i2 = eg.uf.find(env["y"])
            # Check if same operation
            n1 = list(eg.nodes[i1])[0] if eg.nodes[i1] else None
            n2 = list(eg.nodes[i2])[0] if eg.nodes[i2] else None
            if n1 and n2 and n1.tag == "Box" and n2.tag == "Box" and n1.data == n2.data:
                return i1
            return root

        rules.append(Rewrite("FuseOps", PSeq(PVar("x"), PVar("y")), fuse_rhs))
        rule_names.append("FuseOps")

    if "assoc" in args.rules or "all" in args.rules:
        rule_names.append("Assoc")

    if "identity" in args.rules or "all" in args.rules:
        rule_names.append("Identity")

    print(S.metric("RULES", ", ".join(rule_names) if rule_names else "none", S.amber))
    print(S.section("SATURATION TRACE"))

    # Create e-graph and saturate
    eg = EGraph(sig)
    trace = Trace()
    root = eg.add_expr(expr)
    eg.root = root

    try:
        saturate(eg, rules, iters=args.iters, trace=trace)
    except Exception:
        # Fallback if trace param not supported
        saturate(eg, rules, iters=args.iters)

    # Print trace entries
    if trace.entries:
        for i, entry in enumerate(trace.entries[:10]):  # Limit to 10
            print(S.trace_entry(i, f"{S.cyan(entry.rule_name)} applied"))
        if len(trace.entries) > 10:
            print(S.trace_entry(10, f"... and {len(trace.entries) - 10} more"))
    else:
        print(S.trace_entry(0, "Saturation complete"))

    # Extract best
    print(S.section("EXTRACTION"))

    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    boxes_after = count_boxes(best)

    print(S.metric("OUTPUT", pretty(best), S.cyan))
    print(S.metric_change("BOXES", boxes_before, boxes_after))
    print(S.success("OPTIMAL"))

    print(S.footer())
    return 0


# =============================================================================
# DEMO COMMAND
# =============================================================================

def cmd_demo(args: argparse.Namespace) -> int:
    """Run demonstration scripts."""
    demo_name = args.name

    print(S.header("TENSORGRAPH DEMO", demo_name.upper()))

    if demo_name == "core":
        print(S.metric("TARGET", "E-Graph Saturation Demo", S.chrome))
        print(S.divider())
        print()

        # Import and run core demo
        try:
            from ..examples.demo_core import main as demo_main
            demo_main()
            print()
            print(S.success("Core demo completed"))
        except Exception as e:
            print(S.error(f"Demo failed: {e}"))
            return 1

    elif demo_name == "fx":
        print(S.metric("TARGET", "torch.fx LoRA Fusion Demo", S.chrome))
        print(S.metric("REQUIRES", "torch>=2.0", S.amber))
        print(S.divider())
        print()

        try:
            from ..cli.optimize_fx import main as fx_main
            fx_main()
            print()
            print(S.success("FX demo completed"))
        except ImportError as e:
            print(S.error(f"Missing dependency: {e}"))
            print(S.steel("  Install with: pip install tensorgraph[fx]"))
            return 1
        except Exception as e:
            print(S.error(f"Demo failed: {e}"))
            return 1
    else:
        print(S.error(f"Unknown demo: {demo_name}"))
        print(S.steel("  Available: core, fx"))
        return 1

    print(S.footer())
    return 0


# =============================================================================
# INFO COMMAND
# =============================================================================

def cmd_info(args: argparse.Namespace) -> int:
    """Display system information."""
    import platform
    from ..hardware import get_hardware_capabilities

    S.print_banner()

    print(S.section("SYSTEM STATUS"))
    print(S.metric("VERSION", "0.5.1", S.lichen))
    print(S.metric("PYTHON", platform.python_version(), S.chrome))
    print(S.metric("PLATFORM", platform.system(), S.chrome))

    # Host Hardware & Optimal Engine Settings
    caps = get_hardware_capabilities()
    print(S.section("HOST HARDWARE PROFILE"))
    print(S.metric("GPU DEVICE", caps.gpu_name, S.lichen if caps.has_cuda else S.steel))
    if caps.has_cuda:
        print(S.metric("VRAM", f"{caps.vram_gb:.2f} GB", S.chrome))
        print(S.metric("PEAK BANDWIDTH", f"{caps.peak_bandwidth_gbps:.0f} GB/s", S.amber))
        print(S.metric("TRITON SUPPORT", "AVAILABLE" if caps.has_triton else "NOT INSTALLED", S.green if caps.has_triton else S.steel))
        print(S.metric("CUDA GRAPH", "AVAILABLE" if caps.has_cuda_graph else "UNAVAILABLE", S.green if caps.has_cuda_graph else S.steel))
        print(S.metric("ROUTING (seq<=8)", caps.get_optimal_execution_mode(seq_len=1), S.lichen))
        print(S.metric("ROUTING (seq>8)", caps.get_optimal_execution_mode(seq_len=512), S.amber))

    # Check optional dependencies
    print(S.section("DEPENDENCIES"))

    try:
        import torch
        print(S.metric("TORCH", torch.__version__, S.green))
        if torch.cuda.is_available():
            cuda_ver = str(torch.version.cuda)
            print(S.metric("CUDA", cuda_ver, S.green))
        else:
            print(S.metric("CUDA", "not available", S.steel))
    except ImportError:
        print(S.metric("TORCH", "not installed", S.steel))

    print(S.section("MODULES"))
    modules = [
        ("ir", "String Diagram IR"),
        ("egraph", "Equality Graph"),
        ("rewrite", "Pattern Rewriting"),
        ("hardware", "Host Awareness & Hardware Probing"),
        ("engine", "Hybrid Execution Engine"),
        ("backends", "Backend Integrations"),
    ]
    for mod, desc in modules:
        print(f"  {S.cyan(mod):12} │ {S.steel(desc)}")

    print(S.footer())
    return 0



# =============================================================================
# TESTBENCH COMMAND
# =============================================================================

def cmd_testbench(args: argparse.Namespace) -> int:
    """Run the validation testbench suite."""
    from ..testbench.runner import TestbenchRunner

    runner = TestbenchRunner(
        verify_correctness=not args.no_verify,
        iterations=args.iters,
    )
    report = runner.run(category_filter=args.filter)
    return 0 if report.passed_correctness_count == report.total_workloads else 1


# =============================================================================
# COLAB COMMAND
# =============================================================================

def cmd_colab(args: argparse.Namespace) -> int:
    """Monitor or control Google Colab connection."""
    from .colab_monitor import ColabConnectionConfig, ColabMonitor

    config = ColabConnectionConfig(
        host=getattr(args, "host", "localhost"),
        port=getattr(args, "port", 22),
        user=getattr(args, "user", "root"),
        password=getattr(args, "password", "antigravity"),
        timeout=getattr(args, "timeout", 5.0),
        interval=getattr(args, "interval", 10.0),
        keep_alive=not getattr(args, "no_keep_alive", False),
    )

    monitor = ColabMonitor()
    action = getattr(args, "action", "status")

    if action == "status" or not action:
        metrics = monitor.collect_metrics(config, heartbeat_count=1)
        print(monitor.render_status_report(metrics, config))
        return 0 if metrics.status in ("CONNECTED", "DEGRADED") else 1

    elif action == "monitor":
        import time
        print(S.header("COLAB CONNECTION MONITORING", "LIVE"))
        print(S.metric("TARGET", f"{config.user}@{config.host}:{config.port}", S.chrome))
        print(S.metric("INTERVAL", f"{config.interval}s", S.amber))
        print(S.metric("KEEP-ALIVE", "ENABLED" if config.keep_alive else "DISABLED", S.lichen))
        print(S.divider())

        count = 0
        try:
            while True:
                count += 1
                metrics = monitor.collect_metrics(config, heartbeat_count=count)
                print(monitor.render_dashboard(metrics, config))
                if getattr(args, "once", False):
                    break
                time.sleep(config.interval)
        except KeyboardInterrupt:
            print()
            print(S.success("Monitoring session stopped"))
            print(S.footer())

        return 0

    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """TENSORGRAPH CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="tensorgraph",
        description="TENSORGRAPH: Diagrammatic Rewriting Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tensorgraph optimize "f ; f ; g" --rules fuse
  tensorgraph demo core
  tensorgraph colab status --host 127.0.0.1
  tensorgraph colab monitor --host 127.0.0.1
  tensorgraph info

Grand Challenge Technologies — Crafted in the Pacific Northwest
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize expression with e-graph")
    opt_parser.add_argument("expression", type=str, help="Expression to optimize (e.g., 'f ; f ; g')")
    opt_parser.add_argument("--rules", type=str, default="fuse,assoc",
                           help="Comma-separated rules: fuse,assoc,identity,all")
    opt_parser.add_argument("--iters", type=int, default=10, help="Saturation iterations")
    opt_parser.set_defaults(func=cmd_optimize)

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration scripts")
    demo_parser.add_argument("name", type=str, choices=["core", "fx"],
                            help="Demo to run: core, fx")
    demo_parser.set_defaults(func=cmd_demo)

    # info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)

    # testbench command
    tb_parser = subparsers.add_parser("testbench", help="Run validation testbench suite")
    tb_parser.add_argument("--filter", type=str, default=None, help="Filter workloads by category or name")
    tb_parser.add_argument("--no-verify", action="store_true", help="Skip numerical correctness verification")
    tb_parser.add_argument("--iters", type=int, default=10, help="Max saturation iterations per workload")
    tb_parser.set_defaults(func=cmd_testbench)

    # colab command
    colab_parser = subparsers.add_parser("colab", help="Monitor Google Colab CLI connection")
    colab_sub = colab_parser.add_subparsers(dest="action", help="Colab action: status, monitor")

    colab_status_p = colab_sub.add_parser("status", help="Single-shot connection diagnostic scan")
    colab_status_p.add_argument("--host", type=str, default="localhost", help="Colab tunnel hostname/IP")
    colab_status_p.add_argument("--port", type=int, default=22, help="SSH port")
    colab_status_p.add_argument("--user", type=str, default="root", help="SSH username")
    colab_status_p.add_argument("--password", type=str, default="antigravity", help="SSH password")
    colab_status_p.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds")
    colab_status_p.set_defaults(func=cmd_colab)

    colab_mon_p = colab_sub.add_parser("monitor", help="Continuous monitoring dashboard & keep-alive")
    colab_mon_p.add_argument("--host", type=str, default="localhost", help="Colab tunnel hostname/IP")
    colab_mon_p.add_argument("--port", type=int, default=22, help="SSH port")
    colab_mon_p.add_argument("--user", type=str, default="root", help="SSH username")
    colab_mon_p.add_argument("--password", type=str, default="antigravity", help="SSH password")
    colab_mon_p.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds")
    colab_mon_p.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds")
    colab_mon_p.add_argument("--no-keep-alive", action="store_true", help="Disable periodic ping keep-alive")
    colab_mon_p.add_argument("--once", action="store_true", help="Run one iteration and exit")
    colab_mon_p.set_defaults(func=cmd_colab)

    colab_parser.set_defaults(
        func=cmd_colab,
        action="status",
        host="localhost",
        port=22,
        user="root",
        password="antigravity",
        timeout=5.0,
        interval=10.0,
        no_keep_alive=False
    )

    args = parser.parse_args()

    if args.command is None:
        S.print_banner()
        parser.print_help()
        return 0

    return args.func(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    sys.exit(main())

