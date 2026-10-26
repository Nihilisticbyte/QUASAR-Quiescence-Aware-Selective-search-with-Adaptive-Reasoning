"""
Engine-vs-engine matches (Phase 5, metric 4: playing strength).

This is the lowest-rigour metric in the plan -- a headline number for the
presentation, not the primary evidence (that is decision error against
Stockfish, see harness.py). Both engines get the SAME node budget per move.

Fairness:
  - every opening is played twice with colours swapped, so neither engine
    gets the first-move advantage more often;
  - openings are the first few moves of real games from the PGN set, so the
    games start from varied, sensible positions;
  - a game still undecided after `max_plies` is scored as a draw (documented
    bias towards 0.5 -- it can only shrink the measured difference).
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import chess
import chess.pgn


def openings_from_pgn(pgn_path: str, n: int, plies: int = 8, seed: int = 0) -> List[str]:
    """FENs of positions after `plies` half-moves of randomly chosen games
    (no duplicates, game still running)."""
    fens, seen = [], set()
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            moves = list(g.mainline_moves())
            if len(moves) <= plies + 2:
                continue
            for m in moves[:plies]:
                b.push(m)
            if not b.is_game_over() and b.epd() not in seen:
                seen.add(b.epd())
                fens.append(b.fen())
    random.Random(seed).shuffle(fens)
    return fens[:n]


@dataclass
class GameResult:
    opening: str
    a_is_white: bool
    score_a: float          # 1 = engine A won, 0.5 draw, 0 = A lost
    plies: int
    reason: str


def play_game(engine_w: Callable, engine_b: Callable, fen: str, budget: int,
              max_plies: int = 160) -> Tuple[float, int, str]:
    """Returns (white's score, plies played, reason)."""
    board = chess.Board(fen)
    start = len(board.move_stack)
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - start >= max_plies:
            return 0.5, max_plies, "max plies"
        engine = engine_w if board.turn == chess.WHITE else engine_b
        move, _stats = engine(board.copy(), budget)
        if move is None or move not in board.legal_moves:
            raise RuntimeError(f"engine returned an illegal move {move} in {board.fen()}")
        board.push(move)
    res = board.result(claim_draw=True)
    plies = len(board.move_stack) - start
    if res == "1-0":
        return 1.0, plies, "white wins"
    if res == "0-1":
        return 0.0, plies, "black wins"
    return 0.5, plies, "draw"


def run_match(engine_a: Callable, engine_b: Callable, openings: List[str],
              budget: int, max_plies: int = 160, progress=None) -> List[GameResult]:
    out = []
    for i, fen in enumerate(openings, 1):
        for a_white in (True, False):
            w, bl = (engine_a, engine_b) if a_white else (engine_b, engine_a)
            white_score, plies, reason = play_game(w, bl, fen, budget, max_plies)
            out.append(GameResult(fen, a_white, white_score if a_white else 1 - white_score,
                                  plies, reason))
        if progress:
            progress(i, len(openings), out)
    return out


def elo_from_score(score: float) -> float:
    """Elo difference implied by an expected score (0 < score < 1)."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def match_summary(games: List[GameResult]) -> dict:
    n = len(games)
    scores = [g.score_a for g in games]
    mean = sum(scores) / n
    wins = sum(s == 1 for s in scores)
    draws = sum(s == 0.5 for s in scores)
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"games": n, "wins": wins, "draws": draws, "losses": n - wins - draws,
            "score": mean, "score_lo": mean - half, "score_hi": mean + half,
            "elo": elo_from_score(mean),
            "elo_lo": elo_from_score(mean - half) if half == half else float("nan"),
            "elo_hi": elo_from_score(mean + half) if half == half else float("nan")}
