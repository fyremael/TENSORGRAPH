"""
TENSORGRAPH Rewrite Rules for Transformer Attention Optimization.
Fuses `Softmax(Scale(Q @ K.T)) @ V` into `Attention(Q, K, V)`.
"""
from __future__ import annotations

from ..rewrite.pattern import PBox, PSeq, PVar, PPar
from ..rewrite.rule import Rewrite
from ..types import ObjVar

def get_attention_rules() -> list[Rewrite]:
    # Pattern variables
    Q = PVar("Q")
    K = PVar("K")
    V = PVar("V")
    
    # We assume inputs (Q, K, V) are on the stack via `Par(Par(Q, K), V)`
    # Or generically, we match the OPERATIONS applied.
    
    # Structure of Self-Attention:
    # 1. Transpose K: K -> K.T
    # 2. MatMul Q, K.T: (Q, K.T) -> Score
    # 3. Scale: Score -> ScaledScore
    # 4. Softmax: ScaledScore -> Prob
    # 5. MatMul Prob, V: (Prob, V) -> Output
    
    # We want to match the diagram trace.
    # Input: (Q, K, V)
    
    # Right-associative pattern construction helper
    def make_seq(*args):
        if not args: return PId(PVar("_")) # Should not happen
        if len(args) == 1: return args[0]
        return PSeq(args[0], make_seq(*args[1:]))

    # 1. KT = Seq(K, Transpose)
    KT = make_seq(K, PBox("transpose"))

    # 2. Match Prob chain
    # The normalized term for Prob is:
    # Seq(Par(Q, KT), Seq(bmm, Seq(div, softmax)))
    
    Prob_Ops = make_seq(PBox("bmm"), PBox("div_scalar"), PBox("softmax"))
    Prob = make_seq(PPar(Q, KT), Prob_Ops)

    # 3. Match Output
    # Expr = Seq(Par(Prob, V), Bmm)
    # Since Par doesn't distribute, Prob is a sub-term.
    # Output = Seq(Par(Prob, V), Bmm)
    
    Output = make_seq(PPar(Prob, V), PBox("bmm"))
    
    # RHS: Attention(Q, K, V)
    # We construct a term that takes (Q,K,V) inputs.
    # But effectively we want to replace the computation.
    # The merged class will be `Box("attention")`.
    # Wait, `Box("attention")` takes (Q, K, V).
    # Its input must be `Par(Par(Q, K), V)` (left assoc Par) or `Par(Q, Par(K, V))`?
    # TENSORGRAPH Par is just Par.
    # Usually inputs are left-associated pairs? ((Q, K), V).
    
    # We'll use the structure corresponding to the inputs of the matched term.
    # `Par(Prob, V)` -> `Par(Seq(Par(Q, Seq(K, T)), Ops), V)`
    # The inputs are Q, K, V.
    
    # Let's bind Q, K, V variables.
    RHS = make_seq(PPar(PPar(Q, K), V), PBox("attention"))
    
    return [
        Rewrite(
            name="fuse_attention",
            lhs=Output,
            rhs=RHS,
            origin="transformer_optimization"
        )
    ]
