"""
Empirical error model for checking moves at a frontier node (Russell & Wefald idea).

Russell & Wefald (1989) value a search step by the expected improvement in the
decision it can cause, weighed against what it costs. For that they need, per
kind of leaf, the distribution of the "error": backed-up value after the step
minus the leaf's value before it. They learned normal curves from ~35,000
sampled leaves, bucketed by board features.

We copy that recipe for ONE computation step: "search this quiet checking move
at a frontier node" (captures-only quiescence cannot see those). R&W also pick
individual leaves to expand, not whole nodes, so the unit here is one check:

    e = value(frontier node after searching this check) - value(plain quiescence)

(clipped at 0: searching one more move for the side to move can only raise its
value.) The feature that buckets a check is how FORCING it is: the number of
legal replies the opponent has (evasions). 0 replies is checkmate; 1-2 replies
is a forcing line quiescence would never reach; many replies is a loose check.
On the training sample this separated useful checks from useless ones clearly
better than the plain count of checks at the node (AUC 0.76 vs 0.67).

e is one-sided with a spike at 0, so we do not force a normal curve on it; we
keep K empirical quantiles per bucket, plus the mean nodes the step cost.

`expected_gain` answers R&W's question for the check at hand: by how much do we
expect searching it to move the value INSIDE the alpha-beta window? Movement
outside the window cannot change the search result: the alpha-beta version of
R&W's "relevant node" test.

Built offline by fit_error_model.py on the TRAIN split only.
"""

import json
import os
from typing import List, Sequence

DEFAULT_PATH = os.path.join("data", "error_model.json")
DEFAULT_EDGES = (0, 1, 2, 3, 5, 9)   # bucket i: edges[i] <= evasions < edges[i+1]


class ErrorModel:
    def __init__(self, edges: Sequence[int], quantiles: List[List[float]],
                 cost: List[float], n: List[int]):
        if not (len(edges) == len(quantiles) == len(cost) == len(n)) or not edges:
            raise ValueError("edges, quantiles, cost, n must be non-empty and equal length")
        self.edges, self.quantiles, self.cost_nodes, self.n = \
            list(edges), [list(q) for q in quantiles], list(cost), list(n)

    # ---- lookup -----------------------------------------------------------
    def bucket(self, evasions: int) -> int:
        i = 0
        for k, e in enumerate(self.edges):
            if evasions >= e:
                i = k
        return i

    def cost(self, evasions: int) -> float:
        """Mean nodes one check of this kind cost to search (measured, not guessed)."""
        return self.cost_nodes[self.bucket(evasions)]

    def expected_gain(self, evasions: int, base: int, alpha: int, beta: int) -> float:
        """
        E[ change of the window-clipped value ] if this check is searched, for a
        node whose plain-quiescence value is `base` (side-to-move relative):

            mean over quantiles e_i of   max(0, min(base + e_i, beta) - max(base, alpha))

        0 when base >= beta (nothing left to gain) or when even a big error
        cannot reach the window from below.
        """
        qs = self.quantiles[self.bucket(evasions)]
        lo = max(base, alpha)
        total = 0.0
        for e in qs:
            g = min(base + e, beta) - lo
            if g > 0:
                total += g
        return total / len(qs)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str = DEFAULT_PATH, meta: dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = {"edges": self.edges, "quantiles": self.quantiles,
             "cost_nodes": self.cost_nodes, "n": self.n, "meta": meta or {}}
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "ErrorModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error model not found: {path}. Build it first:\n"
                                    "  python fit_error_model.py")
        with open(path) as f:
            d = json.load(f)
        return cls(d["edges"], d["quantiles"], d["cost_nodes"], d["n"])


def quantile_grid(values: Sequence[float], k: int = 20) -> List[float]:
    """k evenly spaced quantiles (midpoints of k equal-probability slices)."""
    v = sorted(values)
    if not v:
        return [0.0] * k
    return [v[min(len(v) - 1, int((i + 0.5) * len(v) / k))] for i in range(k)]


def fit(rows: Sequence[dict], edges: Sequence[int] = DEFAULT_EDGES, k: int = 20) -> ErrorModel:
    """rows: dicts with evasions, e (cp, >= 0) and cost (nodes searching that check took)."""
    edges = list(edges)
    groups = [[] for _ in edges]
    for r in rows:
        i = 0
        for j, e in enumerate(edges):
            if r["evasions"] >= e:
                i = j
        groups[i].append(r)
    quantiles, cost, n = [], [], []
    for g in groups:
        quantiles.append(quantile_grid([r["e"] for r in g], k))
        cost.append(sum(r["cost"] for r in g) / len(g) if g else 1.0)
        n.append(len(g))
    return ErrorModel(edges, quantiles, cost, n)
