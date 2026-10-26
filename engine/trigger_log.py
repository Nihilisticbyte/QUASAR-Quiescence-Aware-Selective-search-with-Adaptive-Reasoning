"""
Phase 3 data collection: which frontier positions are worth extending?

Frontier node = a position where a depth-limited search stops and hands over
to quiescence (depth == 0). For realistic data we do not invent such
positions: we run the real search from a root position and record the
frontier nodes it actually reaches (via SearchStats.frontier_hook).

Ground truth for one frontier position P ("did deeper search change the value
at that node?", metric 3 in project-plan.md):

    v0 = value the search would use now        = quiescence(P)
    v1 = value after ONE more full ply of search = alphabeta(P, depth 1) + quiescence
    changed = |v1 - v0| >= tau        (tau in centipawns, default 100)

Both values are side-to-move relative and use the full window, so they do not
depend on how P was reached. The features logged with each label come from
engine/features.py, so a later analysis can ask which feature predicts
"changed" and at what threshold (analyze_instability.py).

This is node-level ground truth. A node whose value changes does not always
change the ROOT decision, so Phase 5 also measures the end-to-end effect
(decision error at equal node budgets). Both views are reported.
"""

import random
from typing import List

import chess

from .features import position_features, FEATURE_NAMES
from .search import SearchStats, alphabeta, quiescence, search_root, INF
from .tt import TranspositionTable

DEFAULT_TAU = 100   # one pawn. One extra ply moves the value of an ordinary quiet position by
                    # up to ~50cp just because the side to move gets to improve a piece
                    # square (a tempo effect), so a smaller tau would label positional
                    # drift as "tactically changed".
VALUE_CLIP = 1000    # clip values for the label so a mate does not count as "10000cp"


def sample_frontier_fens(board: chess.Board, depth: int, k: int,
                         rng: random.Random) -> List[str]:
    """Run a plain depth-`depth` search from `board`; return up to `k`
    frontier positions (as FENs), chosen uniformly at random (reservoir
    sampling) from all frontier nodes visited."""
    reservoir: List[str] = []
    seen = [0]

    def hook(b: chess.Board, ply: int):
        seen[0] += 1
        if len(reservoir) < k:
            reservoir.append(b.fen())
        else:
            j = rng.randrange(seen[0])
            if j < k:
                reservoir[j] = b.fen()

    stats = SearchStats(frontier_hook=hook)
    search_root(board.copy(), depth, TranspositionTable(), stats)
    return reservoir


def _clip(v: int) -> int:
    return max(-VALUE_CLIP, min(VALUE_CLIP, v))


def label_position(fen: str, tau: int = DEFAULT_TAU) -> dict:
    """Features + ground-truth label for one frontier position."""
    board = chess.Board(fen)
    feats = position_features(board).as_dict()
    v0 = quiescence(board, -INF, INF, SearchStats())
    v1 = alphabeta(board, 1, -INF, INF, SearchStats(), TranspositionTable())
    delta = _clip(v1) - _clip(v0)
    row = {"fen": fen, **feats, "v0": v0, "v1": v1, "delta": delta,
           "changed": int(abs(delta) >= tau)}
    return row


LOG_FIELDS = ["root_id", "source", "fen"] + FEATURE_NAMES + ["v0", "v1", "delta", "changed"]
