"""
Stability diagnostic, in the spirit of Clune (2007), "Heuristic Evaluation
Functions for General Game Playing".

Clune measures how smoothly a feature changes between adjacent game states:

    total variance     = variance of the feature over all sampled states
    adjacent variance  = mean of (feature(s') - feature(s))^2 over pairs where
                         s' is the immediate successor of s on some game path
    stability quotient = total variance / adjacent variance

A quotient near 1 means the feature jumps around from state to state
(unstable); a large quotient means it changes only incrementally (stable).

IMPORTANT (see project-plan.md, loophole 2): in Clune this is an OFFLINE
feature-selection tool run over sampled states. We use it the same way, as a
report over positions from real games -- NOT as a per-node runtime metric.
The per-node trigger used by our controllers (engine/extensions.py) is our
own design, inspired by this idea, not adapted from it.

How to read the result for QUASAR: material and static evaluation should be
highly stable (large quotient: good evaluation features), while tactical
features such as max_gain / n_good_captures / in_check should be much less
stable. That gap is the quantitative reason a fixed-depth cut-off misjudges
tactical positions, and why we look for a trigger on exactly those features.
"""

import math
import statistics
from typing import Dict, List, Sequence

import chess
import chess.pgn

from .eval import evaluate
from .features import position_features


def stability_quotient(sequences: Sequence[Sequence[float]]):
    """
    sequences: one list of feature values per game path, in move order.
    Returns (total_variance, adjacent_variance, quotient).
    quotient is math.inf when the feature never changes between neighbours
    but varies overall, and nan when there is no data / no variation at all.
    """
    values = [v for seq in sequences for v in seq]
    diffs = [(b - a) ** 2 for seq in sequences for a, b in zip(seq, seq[1:])]
    if len(values) < 2 or not diffs:
        return (math.nan, math.nan, math.nan)
    total = statistics.pvariance(values)
    adjacent = sum(diffs) / len(diffs)
    if adjacent == 0:
        return (total, 0.0, math.inf if total > 0 else math.nan)
    return (total, adjacent, total / adjacent)


def feature_paths_from_pgn(pgn_path: str, max_games: int = 40,
                           min_ply: int = 8, max_ply: int = 80,
                           stride: int = 1) -> Dict[str, List[List[float]]]:
    """
    Walk the main line of up to `max_games` games and record, for every
    `stride`-th position between `min_ply` and `max_ply`, the value of each
    feature. White-relative static evaluation ('static_white') and material
    are added, because side-to-move relative values flip sign every ply and
    would look artificially unstable.
    Adjacent samples are positions `stride` plies apart on the same game.
    """
    paths: Dict[str, List[List[float]]] = {}
    games = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while games < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games += 1
            board = game.board()
            per_game: Dict[str, List[float]] = {}
            for move in game.mainline_moves():
                board.push(move)
                ply = len(board.move_stack)
                if ply < min_ply or ply > max_ply or (ply - min_ply) % stride:
                    continue
                if board.is_game_over():
                    break
                feats = position_features(board).as_dict()
                feats.pop("static_stm")          # replaced by the white-relative value
                feats["static_white"] = evaluate(board, terminal_checks=False)
                for name, v in feats.items():
                    per_game.setdefault(name, []).append(float(v))
            for name, seq in per_game.items():
                if len(seq) >= 2:
                    paths.setdefault(name, []).append(seq)
    return paths


def stability_report(paths: Dict[str, List[List[float]]]) -> List[dict]:
    rows = []
    for name, seqs in sorted(paths.items()):
        total, adj, q = stability_quotient(seqs)
        rows.append({"feature": name, "samples": sum(len(s) for s in seqs),
                     "total_var": total, "adjacent_var": adj, "stability": q})
    return rows


def format_stability(rows: List[dict]) -> str:
    head = f"{'feature':<18}{'samples':>8}{'total var':>14}{'adjacent var':>15}{'stability':>11}"
    out = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (math.isnan(r["stability"]), -r["stability"]
                                         if not math.isnan(r["stability"]) else 0)):
        out.append(f"{r['feature']:<18}{r['samples']:>8}{r['total_var']:>14.1f}"
                   f"{r['adjacent_var']:>15.1f}{r['stability']:>11.2f}")
    return "\n".join(out)
