"""
TENSORGRAPH v0.2.0: Interactive E-Graph Visualization Server

This module provides a WebSocket-based server for real-time
visualization of E-Graph saturation. It broadcasts:
- E-Graph state updates (nodes, classes, merges)
- Rule application events
- Saturation progress

The client (React/D3) connects to ws://localhost:8765 to receive
live updates during optimization.

**Innovation: "Temporal Observability"**
Unlike standard egraph visualizers that show only final state,
this server streams EVERY intermediate state, enabling full
"time-travel" debugging through the saturation process.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..egraph import EGraph
from ..egraph.trace import Trace, TraceEntry


# -----------------------------------------------------------------------------
# Data Structures for Visualization
# -----------------------------------------------------------------------------


@dataclass
class EClassViz:
    """Visualization data for an E-Class."""
    id: int
    nodes: list[str]
    sort: tuple[str, str]


@dataclass
class EGraphSnapshot:
    """Complete snapshot of E-Graph state."""
    iteration: int
    eclasses: list[EClassViz]
    total_nodes: int
    total_classes: int


@dataclass
class RuleEvent:
    """Event representing a rule application."""
    iteration: int
    rule_name: str
    matched_class: int
    result_class: int


@dataclass
class SaturationEvent:
    """Union of events that can be broadcasted."""
    event_type: str  # 'snapshot', 'rule', 'complete'
    payload: dict[str, Any]


# -----------------------------------------------------------------------------
# Snapshot Builder
# -----------------------------------------------------------------------------


def build_snapshot(eg: EGraph, iteration: int) -> EGraphSnapshot:
    """Build a visualization snapshot from an EGraph."""
    eclasses = []
    
    for rep, nodes in eg.nodes.items():
        rep = eg.uf.find(rep)
        node_strs = [f"{n.tag}({n.data})" for n in nodes]
        sort = eg.sort.get(rep, ("?", "?"))
        sort_strs = (str(sort[0]), str(sort[1]))
        eclasses.append(EClassViz(id=rep, nodes=node_strs, sort=sort_strs))
    
    return EGraphSnapshot(
        iteration=iteration,
        eclasses=eclasses,
        total_nodes=sum(len(v) for v in eg.nodes.values()),
        total_classes=len(eg.nodes),
    )


# -----------------------------------------------------------------------------
# Observable Saturation Wrapper
# -----------------------------------------------------------------------------


@dataclass
class ObservableSaturation:
    """Saturation engine with event emission for visualization.
    
    This wraps the standard saturation loop and emits events
    that can be consumed by the WebSocket server.
    """
    
    on_event: Callable[[SaturationEvent], None] = field(default=lambda e: None)
    
    def saturate_with_events(
        self,
        eg: EGraph,
        rewrites,
        iters: int = 8,
        max_applications: int = 10_000,
    ) -> Trace:
        """Saturation with event emission.
        
        Emits:
        - 'snapshot' at start and after each iteration
        - 'rule' for each rule application
        - 'complete' when saturation finishes
        """
        from ..rewrite.pattern import ematch
        from ..rewrite.rule import Rewrite, instantiate_pattern
        
        trace = Trace()
        
        # Emit initial snapshot
        self._emit_snapshot(eg, 0)
        
        for i in range(iters):
            applied = 0
            
            for rw in rewrites:
                matches = ematch(eg, rw.lhs)
                
                for root, env, oenv in matches:
                    if applied >= max_applications:
                        break
                    
                    # Apply rewrite
                    if isinstance(rw.rhs, type(rw.lhs).__class__.__bases__[0]):
                        rhs_id = instantiate_pattern(eg, rw.rhs, env, oenv)
                    else:
                        rhs_id = rw.rhs(eg, root, env, oenv)
                    
                    merged_rep = eg.merge(root, rhs_id, reason=rw.name)
                    
                    # Emit rule event
                    self._emit_rule(i + 1, rw.name, root, merged_rep)
                    
                    # Record trace
                    trace.record(
                        rule_name=rw.name,
                        root_eclass=root,
                        rhs_eclass=rhs_id,
                        merged_from=root,
                        merged_to=merged_rep,
                        expr_env=env,
                        obj_env=oenv,
                    )
                    
                    applied += 1
                
                if applied >= max_applications:
                    break
            
            eg.rebuild()
            
            # Emit snapshot after iteration
            self._emit_snapshot(eg, i + 1)
            
            if applied == 0:
                break
        
        # Emit completion
        self._emit_complete(eg)
        
        return trace
    
    def _emit_snapshot(self, eg: EGraph, iteration: int) -> None:
        snapshot = build_snapshot(eg, iteration)
        event = SaturationEvent(
            event_type="snapshot",
            payload={
                "iteration": snapshot.iteration,
                "eclasses": [asdict(ec) for ec in snapshot.eclasses],
                "totalNodes": snapshot.total_nodes,
                "totalClasses": snapshot.total_classes,
            },
        )
        self.on_event(event)
    
    def _emit_rule(self, iteration: int, rule_name: str, matched: int, result: int) -> None:
        event = SaturationEvent(
            event_type="rule",
            payload={
                "iteration": iteration,
                "ruleName": rule_name,
                "matchedClass": matched,
                "resultClass": result,
            },
        )
        self.on_event(event)
    
    def _emit_complete(self, eg: EGraph) -> None:
        event = SaturationEvent(
            event_type="complete",
            payload={
                "totalNodes": sum(len(v) for v in eg.nodes.values()),
                "totalClasses": len(eg.nodes),
            },
        )
        self.on_event(event)


# -----------------------------------------------------------------------------
# WebSocket Server
# -----------------------------------------------------------------------------


class VizServer:
    """WebSocket server for E-Graph visualization.
    
    Usage:
        server = VizServer()
        await server.start()
        # ... run saturation with server.broadcast ...
        await server.stop()
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.server = None
        self.event_history: list[SaturationEvent] = []
    
    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for visualization. "
                "Install with: pip install websockets"
            )
        
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        print(f"[TENSORGRAPH Viz] Server started at ws://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("[TENSORGRAPH Viz] Server stopped")
    
    async def _handle_client(self, websocket):
        """Handle new client connection."""
        self.clients.add(websocket)
        print(f"[TENSORGRAPH Viz] Client connected ({len(self.clients)} total)")
        
        # Send event history to new client
        for event in self.event_history:
            await websocket.send(self._serialize(event))
        
        try:
            async for message in websocket:
                # Handle client commands (e.g., "replay", "seek")
                await self._handle_command(websocket, message)
        finally:
            self.clients.discard(websocket)
            print(f"[TENSORGRAPH Viz] Client disconnected ({len(self.clients)} total)")
    
    async def _handle_command(self, websocket, message: str):
        """Handle commands from client."""
        try:
            cmd = json.loads(message)
            if cmd.get("type") == "replay":
                for event in self.event_history:
                    await websocket.send(self._serialize(event))
                    await asyncio.sleep(0.05)  # Throttle for visualization
        except json.JSONDecodeError:
            pass
    
    def broadcast(self, event: SaturationEvent):
        """Broadcast event to all connected clients (sync wrapper)."""
        self.event_history.append(event)
        
        # Create async task for broadcast
        asyncio.create_task(self._async_broadcast(event))
    
    async def _async_broadcast(self, event: SaturationEvent):
        """Async broadcast to all clients."""
        if self.clients:
            message = self._serialize(event)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True,
            )
    
    def _serialize(self, event: SaturationEvent) -> str:
        """Serialize event to JSON string."""
        return json.dumps({
            "type": event.event_type,
            "payload": event.payload,
        })
