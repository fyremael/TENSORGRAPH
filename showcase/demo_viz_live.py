"""TENSORGRAPH E-Graph Explorer Demo.

This script demonstrates the live visualization capabilities by:
1. Connecting to the running viz server
2. Running saturation with observable events
3. Streaming updates to the browser client

Usage:
    # Terminal 1: Start the viz server
    python -m tensorgraph.viz.server
    
    # Terminal 2: Run this demo
    python showcase/demo_viz_live.py
    
    # Browser: Open showcase/egraph_explorer.html
"""

import asyncio
import sys
sys.path.insert(0, '.')

from tensorgraph.egraph import EGraph
from tensorgraph.ir import Box, Seq, Par
from tensorgraph.signature import Signature
from tensorgraph.types import Obj
from tensorgraph.rewrite import Rewrite
from tensorgraph.rewrite.pattern import PBox, PSeq, PPar
from tensorgraph.viz.server import ObservableSaturation, SaturationEvent

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


async def run_demo():
    """Run the visualization demo."""
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║         TENSORGRAPH v0.2.0 E-Graph Explorer Demo                 ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    # Setup signature with neural network operations
    sig = Signature()
    T = Obj("Tensor")
    sig.add("conv", T, T)
    sig.add("bn", T, T)
    sig.add("relu", T, T)
    sig.add("add", T, T)
    sig.add("fused_conv_bn", T, T)
    sig.add("fused_bn_relu", T, T)
    sig.add("fused_conv_bn_relu", T, T)
    
    # Create E-Graph with a ResNet-like pattern
    # conv -> bn -> relu pattern (common in neural nets)
    eg = EGraph(sig)
    expr = Seq(Seq(Box("conv"), Box("bn")), Box("relu"))
    eg.add_expr(expr)
    
    # Also add parallel pattern
    expr2 = Par(Box("conv"), Box("add"))
    eg.add_expr(expr2)
    
    print(f"[Demo] Created E-Graph with {len(eg.nodes)} e-classes")
    
    # Define fusion rewrites (typical NN optimizations)
    rewrites = [
        Rewrite("conv_bn_fusion", PSeq(PBox("conv"), PBox("bn")), PBox("fused_conv_bn")),
        Rewrite("bn_relu_fusion", PSeq(PBox("bn"), PBox("relu")), PBox("fused_bn_relu")),
        Rewrite("conv_bn_relu_fusion", PSeq(PBox("fused_conv_bn"), PBox("relu")), PBox("fused_conv_bn_relu")),
    ]
    
    print(f"[Demo] Loaded {len(rewrites)} rewrite rules")
    print("[Demo] Connecting to viz server at ws://localhost:8765...")
    
    # Connect to WebSocket server
    try:
        async with websockets.connect("ws://localhost:8765") as ws:
            print("[Demo] Connected to viz server!")
            
            # Create observable saturation with WebSocket event handler
            async def send_event(event: SaturationEvent):
                import json
                msg = json.dumps({
                    "type": event.event_type,
                    "payload": event.payload,
                })
                await ws.send(msg)
                # Small delay for visualization
                await asyncio.sleep(0.3)
            
            # Wrap sync callback for async sending
            events_to_send = []
            def collect_event(event: SaturationEvent):
                events_to_send.append(event)
            
            obs = ObservableSaturation(on_event=collect_event)
            
            print("\n[Demo] Starting saturation...")
            print("─" * 50)
            
            trace = obs.saturate_with_events(eg, rewrites, iters=5)
            
            # Send collected events
            for i, event in enumerate(events_to_send):
                await send_event(event)
                rule_info = event.payload.get('ruleName', 'iter ' + str(event.payload.get('iteration', '')))
                print(f"  -> Sent {event.event_type}: {rule_info}")
            
            print("─" * 50)
            print(f"\n[Demo] Saturation complete!")
            print(f"  • Applied {len(trace.entries)} rewrites")
            print(f"  • Final E-Classes: {len(eg.nodes)}")
            print(f"  • Total E-Nodes: {sum(len(v) for v in eg.nodes.values())}")
            
            print("\n[Demo] Check the browser - you should see the e-graph visualization!")
            print("[Demo] Use the timeline slider to scrub through iterations.")
            
            # Keep connection open briefly
            await asyncio.sleep(2)
            
    except ConnectionRefusedError:
        print("\n[ERROR] Could not connect to viz server!")
        print("        Make sure to start it first:")
        print("        python -m tensorgraph.viz.server")
        return


if __name__ == "__main__":
    asyncio.run(run_demo())
