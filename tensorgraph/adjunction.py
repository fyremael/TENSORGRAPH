from __future__ import annotations

from dataclasses import dataclass

from .ir import Box
from .rewrite import Pattern, PBox, PSeq, Rewrite


@dataclass(frozen=True)
class Adjunction:
    """Operational adjunction f ⊣ g.

    This is a pragmatic mechanism to synthesize "mate" rewrites.

    Convention:
      - `Seq(f, g)` means "do f, then do g".

    If we have a commuting square witnessed by a rewrite:

        (f ; u)  ≡  (v ; f)

    then a mate rule transports `u` across the interface boundary:

        u  ≡  (f ; v ; g)

    Types:
      f : A -> B
      g : B -> A
      u : A -> A
      v : B -> B
    """

    f_lower: Box
    g_lift: Box

    def transport_rule(self, alpha: Rewrite) -> Rewrite:
        """Automatically synthesize a categorical mate for the given rule.
        
        This detects the 'commuting square' pattern and produces the mate.
        Pattern 1: f ; u ≡ v ; f  =>  u ≡ f ; v ; g
        Pattern 2: u ; g ≡ g ; v  =>  v ≡ f ; u ; g
        """
        
        # Pattern 1: f ; u ≡ v ; f
        if isinstance(alpha.lhs, PSeq) and isinstance(alpha.rhs, PSeq):
            # Check LHS: f ; u
            if isinstance(alpha.lhs.a, PBox) and alpha.lhs.a.op == self.f_lower.op:
                u = alpha.lhs.b
                # Check RHS: v ; f
                if isinstance(alpha.rhs.b, PBox) and alpha.rhs.b.op == self.f_lower.op:
                    v = alpha.rhs.a
                    # Mate: u ≡ f ; v ; g
                    mate_rhs = PSeq(PBox(self.f_lower.op), PSeq(v, PBox(self.g_lift.op)))
                    return Rewrite(name=f"mate_ltr({alpha.name})", lhs=u, rhs=mate_rhs, origin=alpha.name)

            # Pattern 2: u ; g ≡ g ; v
            if isinstance(alpha.lhs.b, PBox) and alpha.lhs.b.op == self.g_lift.op:
                u = alpha.lhs.a
                # Check RHS: g ; v
                if isinstance(alpha.rhs.a, PBox) and alpha.rhs.a.op == self.g_lift.op:
                    v = alpha.rhs.a
                    # Wait, v = alpha.rhs.b based on my previous code
                    v = alpha.rhs.b
                    # Mate: v ≡ f ; u ; g
                    mate_rhs = PSeq(PBox(self.f_lower.op), PSeq(u, PBox(self.g_lift.op)))
                    return Rewrite(name=f"mate_rtl({alpha.name})", lhs=v, rhs=mate_rhs, origin=alpha.name)

        raise ValueError(f"Rule '{alpha.name}' does not match a recognizable commuting square for this adjunction.")

    def mate_left_to_right(self, alpha: Rewrite) -> Rewrite:
        """Deprecated: Use transport_rule instead."""
        return self.transport_rule(alpha)
