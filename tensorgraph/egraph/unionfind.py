from __future__ import annotations


class UnionFind:
    """Disjoint-set union with path compression."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def make(self) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.rank.append(0)
        return i

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra
