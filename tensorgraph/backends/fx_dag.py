"""
TENSORGRAPH v0.3.0: Robust DAG-aware FX Importer

This module extends the MVP FX backend to support:
- `call_function` nodes (F.relu, torch.add, etc.)
- `call_method` nodes (x.view, x.reshape, etc.)
- DAG structures (via explicit diagrammatic routing)
- Airity-aware signature inference (v0.3.0)

The key innovation is the `FrontierLifter` class which:
1. Topologically processes the FX graph.
2. Manages a "frontier" of available wires.
3. Inserts explicit `Swap` (routing), `Dup` (copying), and `Del` (cleanup) operations.
4. Produces a linear diagrammatic term avoiding exponential tree expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Box, Dup, Del, Expr, Id, Par, Seq, Swap, normalize
from ..signature import Signature
from ..types import Obj


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.fx as fx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the FX backend."
        ) from e


@dataclass
class NodeInfo:
    """Metadata for an FX node during lifting."""
    node: Any
    op_type: str           # 'placeholder', 'call_module', 'call_function', 'call_method', 'output'
    op_name: str           # Canonical operation name
    attrs: dict[str, Any]  # Operation attributes
    ref_count: int = 0     # Number of remaining consumers


@dataclass
class FrontierLifter:
    """Lifts FX GraphModule to TENSORGRAPH IR using a frontier-based approach.
    
    This generates a linear diagrammatic term (Seq chain) with explicit
    routing (Swap) and duplication (Dup) of wires.
    """
    
    sig: Signature
    tensor_obj: Obj
    node_map: dict[str, NodeInfo] = field(default_factory=dict)
    
    def lift(self, gm) -> Expr:
        _require_torch()
        
        # Phase 1: Analyze types and attributes
        self._analyze_graph(gm)
        
        # Phase 2: Compute ref counts
        self._compute_ref_counts(gm)
        
        # Phase 3: Build linear term via frontier tracking
        return self._build_frontier_term(gm)

    def _analyze_graph(self, gm) -> None:
        """Populate node_map with canonical info for each FX node."""
        import torch
        modules = dict(gm.named_modules())
        
        for node in gm.graph.nodes:
            info = NodeInfo(
                node=node,
                op_type=node.op,
                op_name=self._infer_op_name(node, modules),
                attrs=self._extract_attrs(node, modules),
            )
            self.node_map[node.name] = info

    def _infer_op_name(self, node, modules: dict) -> str:
        if node.op == "placeholder": return "input"
        if node.op == "output": return "output"
        if node.op == "call_module":
            mod = modules.get(str(node.target))
            return type(mod).__name__ if mod else str(node.target)
        if node.op == "call_function":
            func = node.target
            if hasattr(func, "__name__"): return func.__name__
            return str(func).split(".")[-1].rstrip("'>")
        if node.op == "call_method": return str(node.target)
        if node.op == "get_attr": return f"get_attr:{node.target}"
        return f"unknown:{node.op}"

    def _extract_attrs(self, node, modules: dict) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if node.op == "call_module":
            mod = modules.get(str(node.target))
            if mod:
                for k in ["in_features", "out_features", "kernel_size", "stride", "padding", "deltas"]:
                    if hasattr(mod, k): attrs[k] = getattr(mod, k)
        elif node.op in ("call_function", "call_method") and node.kwargs:
            attrs.update(node.kwargs)
        return attrs

    def _compute_ref_counts(self, gm) -> None:
        for node in gm.graph.nodes:
            # We only count explicit input edges
            for input_node in node.all_input_nodes:
                if input_node.name in self.node_map:
                    self.node_map[input_node.name].ref_count += 1
            
            # Note: 'output' node counts as a consumer, which is correct.

    def _build_frontier_term(self, gm) -> Expr:
        """Construct the diagram by processing nodes topologically."""
        
        # The frontier tracks currently available wires: [(node_name, Obj)]
        frontier: list[tuple[str, Obj]] = []
        
        # The sequence of expressions to compose
        term_seq: list[Expr] = []
        
        # Helper to append to sequence
        def emit(e: Expr):
            term_seq.append(e)
            
        # 1. Initialize frontier with placeholders (inputs)
        inputs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for inp in inputs:
            frontier.append((inp.name, self.tensor_obj))
            
        # 2. Process nodes
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue # Already in frontier via init
            
            if node.op == "output":
                # Handle output: route needed lines to end, Del rest.
                args = node.args[0] if node.args else ()
                if not isinstance(args, (list, tuple)):
                    args = (args,)
                
                needed_names = [arg.name for arg in args if hasattr(arg, 'name')]
                
                self._route_to_end(frontier, needed_names, term_seq)
                
                # Now frontier is [..., unneeded..., needed...]
                num_needed = len(needed_names)
                num_unneeded = len(frontier) - num_needed
                
                if num_unneeded > 0:
                    # Emit Del layer
                    dels = [Del(obj) for _, obj in frontier[:num_unneeded]]
                    del_expr = dels[0]
                    for d in dels[1:]: del_expr = Par(del_expr, d)
                    
                    # Pad with Id for needed
                    needed_ids = [Id(obj) for _, obj in frontier[num_unneeded:]]
                    full_expr = del_expr
                    for i in needed_ids: full_expr = Par(full_expr, i)
                    
                    emit(full_expr)
                    
                return self._finalize_seq(term_seq)

            # Standard Node
            info = self.node_map[node.name]
            
            # 2.1 Identify inputs
            needed_inputs = [arg.name for arg in node.args if hasattr(arg, 'name') and arg.name in self.node_map]
            
            # 2.2 Route inputs to the rightmost positions of the frontier
            self._route_to_end(frontier, needed_inputs, term_seq)
            
            # 2.3 Handle Duplication
            # Iterate inputs at end of frontier (active zone)
            num_inputs = len(needed_inputs)
            
            inputs_on_wire = frontier[-num_inputs:] if num_inputs > 0 else []
            frontier = frontier[:len(frontier)-num_inputs] if num_inputs > 0 else frontier
            
            dups_layer = []
            next_frontier_segment = []
            box_inputs = []
            
            for name, obj in inputs_on_wire:
                remaining = self.node_map[name].ref_count
                
                if remaining > 1:
                    # Dup: (Keep, Use)
                    dups_layer.append(Dup(obj))
                    self.node_map[name].ref_count -= 1
                    next_frontier_segment.append((name, obj))
                    box_inputs.append((name, obj))
                else:
                    # Id: (Use)
                    dups_layer.append(Id(obj))
                    self.node_map[name].ref_count -= 1
                    box_inputs.append((name, obj))
            
            # Emit Dup layer
            if dups_layer:
                # We need to prepend Id for existing frontier
                dup_core = dups_layer[0]
                for d in dups_layer[1:]: dup_core = Par(dup_core, d)
                
                if frontier:
                    # Add Ids for rest of frontier
                    frontier_ids = [Id(o) for _, o in frontier]
                    left_core = frontier_ids[0]
                    for f in frontier_ids[1:]: left_core = Par(left_core, f)
                    emit(Par(left_core, dup_core))
                else:
                    emit(dup_core)
                    
            # 2.4 Re-route post-Dup wires
            # Stack top currently: [Dup1_out..., Dup2_out...]
            # If Dup: Keep, Use. If Id: Use.
            # We want: [Keep..., Use...]
            
            current_stack = []
            target_order = next_frontier_segment + box_inputs
            
            for (name, obj), op in zip(inputs_on_wire, dups_layer):
                if isinstance(op, Dup):
                    current_stack.extend([(name, obj), (name, obj)])
                else:
                    current_stack.append((name, obj))
            
            # Permute active stack
            # We emit swaps relative to the *Local* stack context
            # So pass 'offset=len(frontier)' to swap emitter?
            if len(current_stack) > 1:
                self._emit_local_permutation(current_stack, target_order, term_seq, offset=len(frontier))
            
            # Update frontier
            frontier.extend(next_frontier_segment)
            
            # 2.5 Emit Box
            arity = len(box_inputs)
            op_name = f"{info.op_name}_{arity}" if arity > 1 else info.op_name
            
            dom = self.tensor_obj
            for _ in range(arity - 1): dom = dom @ self.tensor_obj
            if arity == 0: dom = Obj("I")
            
            if op_name not in self.sig and op_name not in ("input", "output"):
                self.sig.add(op_name, dom, self.tensor_obj)
            
            box = Box.with_attrs(op_name, **info.attrs) if info.attrs else Box(op_name)
            
            # Box applies to 'box_inputs' which are at TOP of stack
            # We need to wrap Box in Par(Id(frontier), Box)
            if frontier:
                frontier_ids = [Id(o) for _, o in frontier]
                left_core = frontier_ids[0]
                for f in frontier_ids[1:]: left_core = Par(left_core, f)
                emit(Par(left_core, box))
            else:
                emit(box)
                
            # Add output to frontier
            frontier.append((node.name, self.tensor_obj))
            
            # Clean up dead wires immediately?
            # Standard optimization: if a wire in frontier has 0 ref count?
            # Wait, ref count > 0 check handles "Need to use in Future".
            # If ref count == 0 and it was just produced?
            # E.g. Tuple return where 2nd elem unused.
            # We should check frontier for dead wires and Del them?
            # Leave for optimization pass.
            
        return self._finalize_seq(term_seq)

    def _route_to_end(self, frontier, needed_names, term_seq):
        """Emits swaps to move needed_names to the end of frontier in order."""
        if not needed_names: return
        
        # Bubble sort logic
        # We process needed_names in REVERSE to stack them at end
        # Target positions: -1, -2, ...
        
        unique_needed = []
        seen = set()
        for n in needed_names:
            if n not in seen:
                unique_needed.append(n)
                seen.add(n)
        
        for i, name in enumerate(reversed(unique_needed)):
            try:
                # Find current index
                curr_idx = next(idx for idx, (n, _) in enumerate(frontier) if n == name)
            except StopIteration:
                 # Logic error or duplicate usage before Dup?
                 # Assuming topological order and robust ref counting.
                 raise RuntimeError(f"Wire {name} missing from frontier")

            target_idx = len(frontier) - 1 - i
            
            # Bubble up
            while curr_idx < target_idx:
                self._emit_swap(curr_idx, frontier, term_seq)
                curr_idx += 1

    def _emit_local_permutation(self, current, target, term_seq, offset):
        """Permute just the top of stack (frontier is preserved below)."""
        sim_stack = list(current)
        
        for i, (t_name, _) in enumerate(target):
            # Find t_name in sim_stack[i:]
            try:
                curr_local_idx = next(idx for idx, (n, _) in enumerate(sim_stack) if idx >= i and n == t_name)
            except StopIteration:
                 continue
            
            while curr_local_idx > i:
                # Swap local indices (curr-1, curr)
                # Global index = offset + (curr-1)
                self._emit_swap_at(offset + curr_local_idx - 1, term_seq, global_frontier_len=offset+len(current), stack_objs=[o for _, o in sim_stack])
                
                # Update sim stack
                sim_stack[curr_local_idx-1], sim_stack[curr_local_idx] = sim_stack[curr_local_idx], sim_stack[curr_local_idx-1]
                curr_local_idx -= 1

    def _emit_swap(self, idx, frontier, term_seq):
        """Swap at absolute index idx of global frontier."""
        a_name, a_obj = frontier[idx]
        b_name, b_obj = frontier[idx+1]
        frontier[idx], frontier[idx+1] = frontier[idx+1], frontier[idx]
        
        self._emit_swap_at(idx, term_seq, global_frontier_len=len(frontier), stack_objs=[o for _, o in frontier])

    def _emit_swap_at(self, idx, term_seq, global_frontier_len, stack_objs):
        """Low level emission of Par(Id... Swap... Id...)."""
        # stack_objs serves as the type source for Ids
        # Swap happens at idx, idx+1
        
        # Left Ids
        left_ids = [Id(obj) for obj in stack_objs[:idx]]
        # Swap
        swap = Swap(stack_objs[idx], stack_objs[idx+1])
        # Right Ids
        right_ids = [Id(obj) for obj in stack_objs[idx+2:]]
        
        # Build layer
        # If no left ids, start with swap
        layer = swap
        if left_ids:
            layer = Par(left_ids[-1], layer)
            for l in reversed(left_ids[:-1]):
                layer = Par(l, layer)
        
        for r in right_ids:
            layer = Par(layer, r)
            
        term_seq.append(layer)

    def _finalize_seq(self, term_seq) -> Expr:
        if not term_seq: return Id(self.tensor_obj)
        res = term_seq[0]
        # Optimization: Don't normalize incrementally to avoid O(N^2) behavior.
        # The EGraph will normalize when added.
        for t in term_seq[1:]: res = Seq(res, t)
        return res


def lift_fx_graph(gm, sig: Signature, tensor_obj: Obj) -> Expr:
    """Convenience function to lift an FX GraphModule to TENSORGRAPH IR."""
    lifter = FrontierLifter(sig=sig, tensor_obj=tensor_obj)
    return lifter.lift(gm)


# Backwards compatibility alias
DAGLifter = FrontierLifter
