"""
TENSORGRAPH Web Console — The Minderling's API
============================================
Rustic Precision | FastAPI Backend

Serves the documentation console and exposes optimization endpoints.

Run: python -m tensorgraph.cli.web_console
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OptimizeRequest(BaseModel):
    """Request to optimize an expression."""
    expression: str
    rules: list[str] = ["fuse", "assoc"]
    max_iters: int = 10

class OptimizeResponse(BaseModel):
    """Response from optimization."""
    input_expr: str
    output_expr: str
    boxes_before: int
    boxes_after: int
    reduction_pct: float
    iterations: int
    trace: list[str]

class CompareResponse(BaseModel):
    """Response comparing TENSORGRAPH vs Greedy optimization."""
    input_expr: str
    boxes_before: int
    # TENSORGRAPH results
    aether_output: str
    aether_boxes: int
    aether_reduction: float
    aether_iterations: int
    # Greedy results
    greedy_output: str
    greedy_boxes: int
    greedy_reduction: float
    greedy_passes: int
    # Comparison
    aether_wins: bool
    improvement_over_greedy: float

class SystemInfo(BaseModel):
    """System information response."""
    version: str
    python_version: str
    platform: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================

def optimize_expression(req: OptimizeRequest) -> OptimizeResponse:
    """Run e-graph optimization on the given expression."""
    from ..egraph import EGraph
    from ..egraph.extract import Extractor
    from ..egraph.saturation import saturate
    from ..egraph.trace import Trace
    from ..ir import Box, Id, Seq, pretty
    from ..rewrite import PSeq, PVar, Rewrite
    from ..signature import Signature
    from ..types import Obj

    # Parse expression
    expr_str = req.expression
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
    def build_expr(tokens: list[str]) -> Any:
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

    # Count boxes using isinstance with IR classes
    def count_boxes(e: Any) -> int:
        if isinstance(e, Box):
            return 1
        elif isinstance(e, Seq):
            return count_boxes(e.first) + count_boxes(e.second)
        elif hasattr(e, 'top') and hasattr(e, 'bot'):  # Par
            return count_boxes(e.top) + count_boxes(e.bot)
        return 0

    boxes_before = count_boxes(expr)


    # Build rules
    rules = []
    trace_entries: list[str] = []

    # Associativity: (a ; b) ; c  ↔  a ; (b ; c)
    # We need BOTH directions to fully explore tree restructuring
    def assoc_right(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
        """Rewrite (a ; b) ; c  to  a ; (b ; c)."""
        from ..egraph.egraph import ENode
        a = eg.uf.find(env["a"])
        b = eg.uf.find(env["b"])
        c = eg.uf.find(env["c"])
        dom_a, cod_a = eg.sort[a]
        dom_b, cod_b = eg.sort[b]
        dom_c, cod_c = eg.sort[c]
        bc_enode = ENode("Seq", (), (b, c))
        bc_id = eg.add_enode(bc_enode, (dom_b, cod_c))
        abc_enode = ENode("Seq", (), (a, bc_id))
        abc_id = eg.add_enode(abc_enode, (dom_a, cod_c))
        return abc_id
    
    def assoc_left(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
        """Rewrite a ; (b ; c)  to  (a ; b) ; c."""
        from ..egraph.egraph import ENode
        a = eg.uf.find(env["a"])
        b = eg.uf.find(env["b"])
        c = eg.uf.find(env["c"])
        dom_a, cod_a = eg.sort[a]
        dom_b, cod_b = eg.sort[b]
        dom_c, cod_c = eg.sort[c]
        ab_enode = ENode("Seq", (), (a, b))
        ab_id = eg.add_enode(ab_enode, (dom_a, cod_b))
        abc_enode = ENode("Seq", (), (ab_id, c))
        abc_id = eg.add_enode(abc_enode, (dom_a, cod_c))
        return abc_id
    
    if "assoc" in req.rules:
        rules.append(Rewrite(
            "AssocRight", 
            PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
            assoc_right
        ))
        rules.append(Rewrite(
            "AssocLeft", 
            PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
            assoc_left
        ))

    if "fuse" in req.rules:
        def fuse_rhs(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
            """Fuse adjacent identical boxes: Box(op) ; Box(op) → Box(op)"""
            i1 = eg.uf.find(env["x"])
            i2 = eg.uf.find(env["y"])
            
            # Find Box nodes in each e-class
            def find_box(eclass_id):
                for node in eg.nodes.get(eclass_id, []):
                    if node.tag == "Box":
                        return node
                return None
            
            box1 = find_box(i1)
            box2 = find_box(i2)
            
            # If both are boxes with same op, fuse them
            if box1 and box2 and box1.data == box2.data:
                return i1  # Return the first box's e-class
            return root  # No fusion possible

        rules.append(Rewrite("FuseOps", PSeq(PVar("x"), PVar("y")), fuse_rhs))



    # Create e-graph and saturate
    eg = EGraph(sig)
    trace = Trace()
    root = eg.add_expr(expr)
    eg.root = root

    try:
        saturate(eg, rules, iters=req.max_iters, trace=trace)
    except Exception:
        saturate(eg, rules, iters=req.max_iters)

    # Collect trace
    for entry in trace.entries[:20]:
        trace_entries.append(f"{entry.rule_name} applied")

    # Extract best
    ex = Extractor(eg)
    ex.solve(eg.root)
    best = ex.extract(eg.root)

    boxes_after = count_boxes(best)
    reduction = ((boxes_before - boxes_after) / boxes_before * 100) if boxes_before > 0 else 0

    return OptimizeResponse(
        input_expr=expr_str,
        output_expr=pretty(best),
        boxes_before=boxes_before,
        boxes_after=boxes_after,
        reduction_pct=reduction,
        iterations=len(trace.entries),
        trace=trace_entries
    )

def get_system_info() -> SystemInfo:
    """Get system information."""
    import platform
    
    torch_available = False
    torch_version = None
    cuda_available = False
    
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    return SystemInfo(
        version="0.5.0",
        python_version=platform.python_version(),
        platform=platform.system(),
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available
    )

def compare_expression(req: OptimizeRequest) -> CompareResponse:
    """Compare TENSORGRAPH e-graph optimization vs greedy optimization."""
    from ..egraph import EGraph
    from ..egraph.extract import Extractor
    from ..egraph.saturation import saturate
    from ..egraph.trace import Trace
    from ..ir import Box, Id, Seq, pretty
    from ..rewrite import PSeq, PVar, Rewrite
    from ..signature import Signature
    from ..types import Obj

    # Parse expression
    expr_str = req.expression
    T = Obj("T")
    sig = Signature()

    tokens = [t.strip() for t in expr_str.replace("(", "").replace(")", "").split(";")]
    tokens = [t for t in tokens if t]

    ops_seen = set()
    for tok in tokens:
        if tok not in ops_seen and tok != "id":
            sig.add(tok, T, T)
            ops_seen.add(tok)

    def build_expr(tokens: list[str]) -> Any:
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

    def count_boxes(e: Any) -> int:
        if isinstance(e, Box):
            return 1
        elif isinstance(e, Seq):
            return count_boxes(e.first) + count_boxes(e.second)
        elif hasattr(e, 'top') and hasattr(e, 'bot'):
            return count_boxes(e.top) + count_boxes(e.bot)
        return 0

    boxes_before = count_boxes(expr)

    # === TENSORGRAPH E-GRAPH OPTIMIZATION ===
    rules = []
    
    # Associativity: (a ; b) ; c  ↔  a ; (b ; c)
    # We need BOTH directions to fully explore tree restructuring
    def assoc_right(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
        """Rewrite (a ; b) ; c  to  a ; (b ; c)."""
        from ..egraph.egraph import ENode
        a = eg.uf.find(env["a"])
        b = eg.uf.find(env["b"])
        c = eg.uf.find(env["c"])
        dom_a, cod_a = eg.sort[a]
        dom_b, cod_b = eg.sort[b]
        dom_c, cod_c = eg.sort[c]
        bc_enode = ENode("Seq", (), (b, c))
        bc_id = eg.add_enode(bc_enode, (dom_b, cod_c))
        abc_enode = ENode("Seq", (), (a, bc_id))
        abc_id = eg.add_enode(abc_enode, (dom_a, cod_c))
        return abc_id
    
    def assoc_left(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
        """Rewrite a ; (b ; c)  to  (a ; b) ; c."""
        from ..egraph.egraph import ENode
        a = eg.uf.find(env["a"])
        b = eg.uf.find(env["b"])
        c = eg.uf.find(env["c"])
        dom_a, cod_a = eg.sort[a]
        dom_b, cod_b = eg.sort[b]
        dom_c, cod_c = eg.sort[c]
        ab_enode = ENode("Seq", (), (a, b))
        ab_id = eg.add_enode(ab_enode, (dom_a, cod_b))
        abc_enode = ENode("Seq", (), (ab_id, c))
        abc_id = eg.add_enode(abc_enode, (dom_a, cod_c))
        return abc_id
    
    # Add both associativity directions
    rules.append(Rewrite(
        "AssocRight", 
        PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
        assoc_right
    ))
    rules.append(Rewrite(
        "AssocLeft", 
        PSeq(PVar("a"), PSeq(PVar("b"), PVar("c"))),
        assoc_left
    ))
    
    if "fuse" in req.rules:
        def fuse_rhs(eg: Any, root: int, env: dict, oenv: dict, denv: dict) -> int:
            """Fuse adjacent identical boxes: Box(op) ; Box(op) → Box(op)"""
            i1 = eg.uf.find(env["x"])
            i2 = eg.uf.find(env["y"])
            
            # Find Box nodes in each e-class
            def find_box(eclass_id):
                for node in eg.nodes.get(eclass_id, []):
                    if node.tag == "Box":
                        return node
                return None
            
            box1 = find_box(i1)
            box2 = find_box(i2)
            
            # If both are boxes with same op, fuse them
            if box1 and box2 and box1.data == box2.data:
                return i1  # Return the first box's e-class
            return root  # No fusion possible
        rules.append(Rewrite("FuseOps", PSeq(PVar("x"), PVar("y")), fuse_rhs))


    eg = EGraph(sig)
    trace = Trace()
    root = eg.add_expr(expr)
    eg.root = root

    try:
        saturate(eg, rules, iters=req.max_iters, trace=trace)
    except Exception:
        saturate(eg, rules, iters=req.max_iters)

    ex = Extractor(eg)
    ex.solve(eg.root)
    aether_best = ex.extract(eg.root)
    aether_boxes = count_boxes(aether_best)
    aether_reduction = ((boxes_before - aether_boxes) / boxes_before * 100) if boxes_before > 0 else 0

    # === GREEDY OPTIMIZER (Tree-Based) ===
    # Realistic greedy: works on tree structure, only fuses immediate Seq children
    # This exposes phase ordering problems that e-graph saturation avoids
    
    def greedy_fuse_pass(e: Any) -> tuple[Any, bool]:
        """Single pass of greedy fusion on expression tree.
        
        Only fuses Seq(Box(a), Box(a)) -> Box(a) at leaves.
        Returns (new_expr, changed).
        """
        if isinstance(e, Box):
            return e, False
        elif isinstance(e, Id):
            return e, False
        elif isinstance(e, Seq):
            # Recursively optimize children first (bottom-up)
            new_first, c1 = greedy_fuse_pass(e.first)
            new_second, c2 = greedy_fuse_pass(e.second)
            
            # Try to fuse if both children are identical boxes
            if (isinstance(new_first, Box) and isinstance(new_second, Box) 
                and new_first.op == new_second.op and "fuse" in req.rules):
                return new_first, True
            
            # Return potentially modified Seq
            if c1 or c2:
                return Seq(new_first, new_second), True
            return e, False
        else:
            return e, False
    
    greedy_expr = expr
    greedy_passes = 0
    max_passes = 10
    
    while greedy_passes < max_passes:
        greedy_expr, changed = greedy_fuse_pass(greedy_expr)
        greedy_passes += 1
        if not changed:
            break
    
    greedy_boxes = count_boxes(greedy_expr)
    greedy_reduction = ((boxes_before - greedy_boxes) / boxes_before * 100) if boxes_before > 0 else 0

    # Comparison
    aether_wins = aether_boxes < greedy_boxes
    improvement = greedy_boxes - aether_boxes if aether_boxes < greedy_boxes else 0

    return CompareResponse(
        input_expr=expr_str,
        boxes_before=boxes_before,
        aether_output=pretty(aether_best),
        aether_boxes=aether_boxes,
        aether_reduction=aether_reduction,
        aether_iterations=len(trace.entries),
        greedy_output=pretty(greedy_expr),
        greedy_boxes=greedy_boxes,
        greedy_reduction=greedy_reduction,
        greedy_passes=greedy_passes,
        aether_wins=aether_wins,
        improvement_over_greedy=improvement
    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="TENSORGRAPH Web Console",
    description="The Minderling's API — Knowledge from beyond the forest",
    version="0.5.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/api/info")
async def api_info() -> SystemInfo:
    """Get system information."""
    return get_system_info()

@app.post("/api/optimize")
async def api_optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Optimize an expression using e-graph saturation."""
    try:
        return optimize_expression(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "operational", "message": "The Minderling is awake"}

@app.post("/api/compare")
async def api_compare(req: OptimizeRequest) -> CompareResponse:
    """Compare TENSORGRAPH e-graph vs greedy optimization."""
    try:
        return compare_expression(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# STATIC FILE SERVING
# =============================================================================

# Determine console directory
CONSOLE_DIR = Path(__file__).parent.parent.parent / "docs" / "console"

@app.get("/")
async def serve_index():
    """Serve the main console page."""
    index_path = CONSOLE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Console not found")

@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve static files from the console directory."""
    file_path = CONSOLE_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail=f"File not found: {path}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the web console server."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    from . import style as S
    
    print(S.header("TENSORGRAPH WEB CONSOLE", "STARTING"))
    print(S.metric("VERSION", "0.5.0", S.lichen))
    print(S.metric("CONSOLE", str(CONSOLE_DIR), S.chrome))
    print(S.divider())
    print()
    print(S.lichen("  The Minderling awakens at http://localhost:8080"))
    print()
    print(S.section("API ENDPOINTS"))
    print(S.metric("GET", "/api/info", S.chrome))
    print(S.metric("POST", "/api/optimize", S.chrome))
    print(S.metric("GET", "/api/health", S.chrome))
    print(S.footer())
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")

if __name__ == "__main__":
    main()
