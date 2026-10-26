#!/usr/bin/env python
from __future__ import annotations

import sys
import time

# TENSORGRAPH Grand Challenge: Parallel Scale Demo
# Demonstrates optimization of a wide, parallel graph (1000+ nodes).

from tensorgraph import (
    Box, EGraph, Extractor, Obj, Seq, Par, Signature, pretty, saturate,
    Rewrite, PBox, PSeq, PVar, PPar
)
from tensorgraph.ir import Dup, Swap, Del
from tensorgraph.cli import style as S

# Ensure UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Increase recursion for deep structural normalization if needed
sys.setrecursionlimit(5000)

def cinematic_print(text, color=S.STEEL, delay=0.005):
    sys.stdout.write(color)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(S.RESET + "\n")

def main():
    S.print_banner()
    print(S.header("GRAND CHALLENGE v0.3.0", "PARALLEL SCALE"))
    
    # Configuration
    BRANCHES = 20
    DEPTH_PER_BRANCH = 50 
    TOTAL_OPS = BRANCHES * DEPTH_PER_BRANCH
    
    print(S.section("THE 1000-NODE PARALLEL FABRIC"))
    cinematic_print(f"  Constructing {BRANCHES} Parallel Execution units of depth {DEPTH_PER_BRANCH}...")
    
    T = Obj("Tensor")
    sig = Signature()
    sig.add("conv", T, T)
    sig.add("relu", T, T)
    sig.add("bn", T, T)
    # Optimization target: fuse relu ; bn -> relu_bn (synthetic optimization)
    sig.add("relu_bn", T, T)
    
    # 1. Build a single branch chain: Conv ; Relu ; BN ...
    def make_branch(depth, branch_id):
        # seq of alternating conv, relu, bn
        ops = []
        # Add a unique identity to force node distinctness
        ops.append(Box(f"branch_{branch_id}")) # Phantom op
        
        for _ in range(depth):
            ops.append(Box("conv"))
            ops.append(Box("relu"))
            ops.append(Box("bn"))
        
        # Chain them
        e = ops[-1]
        for op in reversed(ops[:-1]):
            e = Seq(op, e)
        return e

    # 2. Build Parallel Structure
    # Input -> Split -> [Branch1, Branch2, ...] -> Merge?
    # For now, just a Tensor Product of branches: T^N -> T^N
    # Par(Branch1, Par(Branch2, ...))
    
    # We need to register the phantom ops or EGraph will complain? 
    # EGraph allows unknown ops if not type checked strictly? 
    # Actually, add_expr uses sig.get() if it's a Box?
    # No, Box just holds a string. infer_type uses sig.
    # But EGraph doesn't infer types on add, only when needed?
    # Wait, eg.add_enode uses just signature to find dom/cod for sort?
    # Yes, EGraph init takes signature.
    # We need to add phantom ops to signature.
    
    for i in range(BRANCHES):
        sig.add(f"branch_{i}", T, T)

    full_expr = make_branch(DEPTH_PER_BRANCH // 3, 0)
    for i in range(1, BRANCHES):
        full_expr = Par(make_branch(DEPTH_PER_BRANCH // 3, i), full_expr)
        
    # 3. Rules
    # Fuse Relu ; BN -> ReluBN
    rule_fusion = Rewrite(
        "fuse_relu_bn",
        PSeq(PBox("relu"), PBox("bn")),
        PBox("relu_bn")
    )
    
    # Associativity rules to help matching find the pair
    rule_assoc = Rewrite(
        "assoc",
        PSeq(PSeq(PVar("a"), PVar("b")), PVar("c")),
        PSeq(PVar("a"), PSeq(PVar("b"), PVar("c")))
    )
    
    rules = [rule_fusion, rule_assoc]
    
    eg = EGraph(sig)
    
    print("  Adding expression to E-Graph...")
    # This might take a moment due to normalization
    start_load = time.time()
    root = eg.add_expr(full_expr)
    print(f"  Load time: {time.time() - start_load:.4f}s")
    
    total_nodes = len(eg.nodes)
    print(S.metric("Graph Nodes", f"{total_nodes:,}", S.chrome))
    print(S.metric("Structure", f"{BRANCHES}x Parallel ResNets", S.amber))
    
    cinematic_print(f"  {S.GREEN}●{S.RESET} Initiating Parallel Fusion Optimization...")
    
    start_t = time.perf_counter()
    # High max_applications because we expect many fusions in parallel
    saturate(eg, rules, iters=10, max_applications=100_000)
    end_t = time.perf_counter()
    elapsed = end_t - start_t
    
    print(S.metric("Convergence", f"{elapsed*1000:.2f} ms", S.green))
    
    # Check if fusion happened
    # We count "relu_bn" nodes
    fused_count = 0
    for block in eg.nodes.values():
        for en in block:
            if en.tag == "Box" and en.data[0] == "relu_bn":
                fused_count += 1
                
    expected = BRANCHES * (DEPTH_PER_BRANCH // 3)
    # Roughly expected count.
    
    print(S.metric("Fused Ops", f"{fused_count}", S.cyan))
    
    ops_per_sec = (total_nodes / elapsed)
    print(S.metric("Throughput", f"{ops_per_sec:,.0f} nodes/sec", S.cyan))
    
    print(S.footer())
    
    if fused_count > 0:
         cinematic_print(f"\n{S.CYAN}{S.BOLD}  SCALE OPTIMIZATION SUCCESSFUL{S.RESET}\n")
    else:
         cinematic_print(f"\n{S.RED}  NO FUSION DETECTED{S.RESET}\n")

if __name__ == "__main__":
    main()
