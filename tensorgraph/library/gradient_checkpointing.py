"""
TENSORGRAPH Gradient Checkpointing Adjunction Library.

This module defines the adjunctions required to synthesize backward passes
for checkpointed operations.

We model Checkpointing as a pair of adjoint operators:
- `Save`: Stores activations (or seeds) to memory.
- `Load`: Retrieves/Recomputes activations from memory.
- We assert `Save ⊣ Load`.

We also define a standard operator adjunction:
- `Linear ⊣ LinearT` (classic transpose relationship).

The goal is to show that `Adjoint(Linear ; Save)` synthesizes to `Load ; LinearT`.
"""

from ..ir import Box
from ..adjunction import Adjunction
from ..rewrite.rule import Rewrite
from ..rewrite.pattern import PBox, PSeq, PVar

def get_gradient_adjunctions() -> list[Adjunction]:
    # 1. Linear ⊣ LinearT
    # Forward: x -> Wx
    # Backward: dy -> W.T dy
    adj_linear = Adjunction(
        f_lower=Box("linear"),
        g_lift=Box("linear_t")
    )
    
    # 2. Save ⊣ Load (Checkpointing)
    # Forward: x -> save(x)
    # Backward: dy -> load(dy) (conceptually recomputing or loading)
    adj_ckpt = Adjunction(
        f_lower=Box("save"),
        g_lift=Box("load")
    )
    
    return [adj_linear, adj_ckpt]

def get_commuting_squares() -> list[Rewrite]:
    # We need to define the 'commuting squares' that drive the synthesis rule transport.
    # But wait, Adjunction.transport_rule *takes* a square and *produces* a mate.
    # In this case, we rely on the compositionality of adjunctions.
    
    # If we have F ⊣ G and H ⊣ K.
    # Then F ; H ⊣ K ; G.
    
    # TENSORGRAPH v0.3.0 doesn't seemingly have a global "Adjoint" operator that recurses.
    # It has `transport_rule`.
    
    # To demonstrate `Adjoint(Linear ; Save) -> Load ; LinearT`, we need to 
    # trick the system or use a rewrite that represents the "Adjoint Definition".
    
    # Let's say we have an identity:
    # Op ; Adjoint(Op) -> I (unit) ?
    # Or simply we want to define the rules that *witness* these adjunctions.
    
    # In `grand_challenge_v030.py`, they define a square: `Pool ; Warp ≡ Rotate ; Pool`.
    # And synthesize `Warp ≡ Unpool ; Rotate ; Pool`.
    # This assumes `Pool ⊣ Unpool` and synthesizes the mate of `Warp` (via Rotate/Pool).
    
    # This assumes we already KNOW the mate of `Pool`.
    
    # For Checkpointing:
    # We want to synthesize the Backward Pass of `Linear; Save`.
    # Let `F = Linear; Save`.
    # We want to find `G` such that `F ⊣ G`.
    
    # If we treat `Adjoint` as a rewrite strategy that reverses standard operators?
    # This might be outside the scope of `adjunction.py`'s current implementation (simple transport).
    
    # Alternative Plan:
    # We define `Linear ; Save` as a single composite Block `CheckpointedLinear`.
    # We define `Load ; LinearT` as `CheckpointedLinearT`.
    # We show that we can synthesize `CheckpointedLinearT` from components?
    
    return [] 
