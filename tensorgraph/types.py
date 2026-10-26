from __future__ import annotations

from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Objects (types)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Obj:
    """A type object.

    Objects represent system interfaces.

    Tensor product is represented structurally as a binary tree:
        Obj.tensor(A, B)  corresponds to  A ⊗ B
    """

    name: str
    left: Obj | None = None
    right: Obj | None = None

    @staticmethod
    def tensor(a: Obj, b: Obj) -> Obj:
        return Obj(name="⊗", left=a, right=b)

    @staticmethod
    def sum_type(a: Obj, b: Obj) -> Obj:
        return Obj(name="+", left=a, right=b)

    def __matmul__(self, other: Obj) -> Obj:
        return Obj.tensor(self, other)

    def __add__(self, other: Obj) -> Obj:
        return Obj.sum_type(self, other)

    def is_tensor(self) -> bool:
        return self.name == "⊗" and self.left is not None and self.right is not None

    def is_sum(self) -> bool:
        return self.name == "+" and self.left is not None and self.right is not None

    def __str__(self) -> str:
        if self.is_tensor():
            return f"({self.left} ⊗ {self.right})"
        if self.is_sum():
            return f"({self.left} + {self.right})"
        return self.name


@dataclass(frozen=True)
class ObjVar:
    """Pattern variable that can match an object."""

    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


ObjLike = Obj | ObjVar
Sort = tuple[Obj, Obj]  # (dom, cod)
