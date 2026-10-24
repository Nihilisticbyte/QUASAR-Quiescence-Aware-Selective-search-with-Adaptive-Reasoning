"""
Test-position loading for the Phase 2 evaluation harness.

Two sources, both returned as a list of `Position`:

  1. load_epd(path)            -- tactical suites such as data/wac.epd
                                  (Win At Chess, 300 positions). These carry
                                  the book's best move(s) in `best_moves`.
  2. sample_pgn_positions(...)  -- positions sampled from real games in a PGN
                                  file. No book answer exists, so
                                  `best_moves` is empty; Stockfish (see
                                  engine/oracle.py) is the ground truth.

Why both: Plaat et al. warn that standard test suites alone are poor
predictors of real playing behaviour, so we evaluate on tactical positions
AND on positions from ordinary games.
"""

import random
from dataclasses import dataclass, field
from typing import List

import chess
import chess.pgn


@dataclass
class Position:
    fen: str
    id: str
    source: str                       # "wac", "real", ...
    best_moves: List[chess.Move] = field(default_factory=list)  # empty if unknown

    def board(self) -> chess.Board:
        return chess.Board(self.fen)


def load_epd(path: str, source: str = None, limit: int = None) -> List[Position]:
    """Load an EPD file (one position per line, with `bm` and `id` opcodes).
    Blank lines are skipped. Raises ValueError naming the line if a position
    or its best move is illegal, so a corrupt suite fails loudly."""
    source = source or path.replace("\\", "/").rsplit("/", 1)[-1].split(".")[0]
    positions = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            board = chess.Board()
            try:
                ops = board.set_epd(line)
            except ValueError as e:
                raise ValueError(f"{path} line {n}: {e}") from e
            positions.append(Position(
                fen=board.fen(),
                id=str(ops.get("id", f"{source}.{n}")),
                source=source,
                best_moves=list(ops.get("bm", []))))
            if limit and len(positions) >= limit:
                break
    return positions


def sample_pgn_positions(pgn_path: str, n: int, min_ply: int = 12,
                         max_ply: int = 80, per_game: int = 3,
                         seed: int = 0) -> List[Position]:
    """
    Sample up to `n` positions from the games in a PGN file.

    - Only positions between `min_ply` and `max_ply` (half-moves played) are
      eligible: skips memorised openings and long, drawn-out endgames.
    - Positions with fewer than 2 legal moves or already game-over are
      skipped (nothing to decide).
    - At most `per_game` positions come from any one game, so a few long
      games can't dominate the set.
    - Deterministic for a given `seed`; duplicates (same FEN) are dropped.
    Returns fewer than `n` if the file doesn't contain enough games.
    """
    rng = random.Random(seed)
    positions, seen = [], set()
    game_index = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while len(positions) < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_index += 1
            board = game.board()
            candidates = []
            for move in game.mainline_moves():
                board.push(move)
                ply = len(board.move_stack)
                if min_ply <= ply <= max_ply and not board.is_game_over() \
                        and board.legal_moves.count() >= 2:
                    candidates.append((ply, board.fen()))
            rng.shuffle(candidates)
            taken = 0
            for ply, fen in candidates:
                if taken >= per_game or len(positions) >= n:
                    break
                if fen in seen:
                    continue
                seen.add(fen)
                positions.append(Position(fen=fen, id=f"RG.{game_index}.{ply}",
                                          source="real"))
                taken += 1
    return positions
