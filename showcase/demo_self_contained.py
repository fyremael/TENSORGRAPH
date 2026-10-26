"""TENSORGRAPH E-Graph Explorer - Self-Contained Demo.

This script runs BOTH the WebSocket server AND the saturation demo
in a single process, making it easy to demonstrate the visualization.

Usage:
    python showcase/demo_self_contained.py
    
Then open showcase/egraph_explorer.html in a browser.
The e-graph will be visualized with time-travel debugging.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from tensorgraph.egraph import EGraph
from tensorgraph.ir import Box, Seq, Par
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.rewrite import Rewrite
from tensorgraph.rewrite.pattern import PBox, PSeq
from tensorgraph.viz.server import VizServer, ObservableSaturation

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


async def run_demo():
    """Run visualization server and saturation demo."""
    print()
    print("=" * 60)
    print("   TENSORGRAPH v0.2.0 E-Graph Explorer Demo")
    print("=" * 60)
    print()
    
    # Start the visualization server
    server = VizServer(host="localhost", port=8765)
    await server.start()
    
    print()
    print(">>> Open this file in your browser:")
    print("    file:///E:/_antigravity/TENSORGRAPH/showcase/egraph_explorer.html")
    print()
    print(">>> Waiting 10 seconds for browser connection...")
    print("    (Open the explorer in your browser NOW!)")
    await asyncio.sleep(10)
    
    # Setup signature with neural network operations
    sig = Signature()
    T = Obj("Tensor")
    sig.add("conv", T, T)
    sig.add("bn", T, T)
    sig.add("relu", T, T)
    sig.add("fused_conv_bn", T, T)
    sig.add("fused_bn_relu", T, T)
    
    # Create E-Graph with a typical pattern
    eg = EGraph(sig)
    expr = Seq(Seq(Box("conv"), Box("bn")), Box("relu"))
    eg.add_expr(expr)
    
    print(f"[Demo] Created E-Graph with {len(eg.nodes)} e-classes")
    
    # Define fusion rewrites
    rewrites = [
        Rewrite("conv_bn_fusion", PSeq(PBox("conv"), PBox("bn")), PBox("fused_conv_bn")),
        Rewrite("bn_relu_fusion", PSeq(PBox("bn"), PBox("relu")), PBox("fused_bn_relu")),
    ]
    
    print(f"[Demo] Loaded {len(rewrites)} rewrite rules")
    print()
    print("-" * 40)
    print("Running saturation with visualization...")
    print("-" * 40)
    
    # Create observable saturation that broadcasts to server
    obs = ObservableSaturation(on_event=server.broadcast)
    
    # Run saturation (this will emit events to connected clients)
    trace = obs.saturate_with_events(eg, rewrites, iters=5)
    
    print()
    print(f"[Demo] Saturation complete!")
    print(f"  - Applied {len(trace.entries)} rewrites")
    print(f"  - Final E-Classes: {len(eg.nodes)}")
    print(f"  - Total E-Nodes: {sum(len(v) for v in eg.nodes.values())}")
    print()
    
    # Keep server running so user can interact
    print(">>> Check the browser - use the timeline slider to scrub through iterations!")
    print(">>> Press Ctrl+C to stop the server.")
    print()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n[Demo] Shutting down...")
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        pass
