"""
Stockfish oracle (Phase 2): ground truth for decision-error metrics.

Wraps Stockfish through python-chess's UCI interface so the evaluation
harness can ask two things about any position:
  1. What does Stockfish think the best move / score is?   -> analyse()
  2. How much worse (in centipawns) was OUR engine's move
     than Stockfish's best move?                           -> move_loss()

Design choices (explain these in the report):
  - Fixed DEPTH limit (default 12), not time: results are repeatable and
    don't depend on how busy the machine is. Threads=1 for the same reason.
  - All scores are from the SIDE TO MOVE's point of view (negamax style,
    same convention as search.py), never White-relative.
  - Mate scores are converted to +/-MATE_CP so they fit on the cp scale,
    then move_loss() clips to +/-cp_cap so one missed mate doesn't swamp
    an average over many positions.
  - Results are cached per (FEN, depth), so comparing several engine
    versions on the same positions only pays for Stockfish once.

Finding the binary (first match wins):
  1. the `path` argument, 2. the STOCKFISH_PATH environment variable,
  3. `stockfish` on PATH.
On Windows, e.g. (PowerShell):  $env:STOCKFISH_PATH = "C:\\tools\\stockfish\\stockfish.exe"
"""

import os
import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

MATE_CP = 10_000  # stand-in centipawn value for a forced mate


@dataclass
class OracleResult:
    best_move: Optional[chess.Move]
    score_cp: int            # side-to-move relative, mate mapped to +/-MATE_CP
    mate_in: Optional[int]   # +n = side to move mates in n, -n = gets mated, None = no forced mate


def find_stockfish(path: str = None) -> str:
    """Return a usable Stockfish executable path or raise FileNotFoundError."""
    candidates = [path, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Stockfish not found. Download it from https://stockfishchess.org/download/ "
        "then either pass path=... or set the STOCKFISH_PATH environment variable "
        "to the full path of the stockfish executable.")


class StockfishOracle:
    def __init__(self, path: str = None, depth: int = 12, hash_mb: int = 64):
        self.depth = depth
        self.calls = 0   # real Stockfish searches (cache hits don't count)
        self._cache: dict = {}
        self._engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(path))
        self._engine.configure({"Threads": 1, "Hash": hash_mb})
        self.version = self._engine.id.get("name", "unknown")   # e.g. "Stockfish 16"

    # --- lifecycle -------------------------------------------------------
    def close(self):
        self._engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- queries -----------------------------------------------------------
    def analyse(self, board: chess.Board) -> OracleResult:
        """Stockfish's best move and score for `board` (side-to-move relative)."""
        key = (board.fen(), self.depth)
        if key in self._cache:
            return self._cache[key]

        if board.is_checkmate():
            result = OracleResult(None, -MATE_CP, 0)
        elif board.is_game_over(claim_draw=False):
            result = OracleResult(None, 0, None)
        else:
            info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
            self.calls += 1
            pov = info["score"].pov(board.turn)
            mate = pov.mate()
            result = OracleResult(
                best_move=info["pv"][0] if info.get("pv") else None,
                score_cp=pov.score(mate_score=MATE_CP),
                mate_in=mate)
        self._cache[key] = result
        return result

    def move_loss(self, board: chess.Board, move: chess.Move, cp_cap: int = 1000) -> int:
        """
        Centipawn loss of playing `move` instead of Stockfish's best move,
        always >= 0. Both scores are clipped to +/-cp_cap first.

        Computed as: best score for the mover, minus the mover's score after
        `move` (Stockfish's evaluation of the resulting position, negated
        because it is then the opponent's turn).
        """
        best = self.analyse(board)
        if move == best.best_move:
            return 0  # identical move: no loss, and avoids depth-noise artefacts

        after_board = board.copy()
        after_board.push(move)
        after = -self.analyse(after_board).score_cp

        clip = lambda v: max(-cp_cap, min(cp_cap, v))
        return max(0, clip(best.score_cp) - clip(after))
