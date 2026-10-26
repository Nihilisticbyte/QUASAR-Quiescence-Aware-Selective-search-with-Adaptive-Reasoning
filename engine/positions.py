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
                         seed: int = 0, game_filter=None) -> List[Position]:
    """
    Sample up to `n` positions from the games in a PGN file.

    - Only positions between `min_ply` and `max_ply` (half-moves played) are
      eligible: skips memorised openings and long, drawn-out endgames.
    - Positions with fewer than 2 legal moves or already game-over are
      skipped (nothing to decide).
    - At most `per_game` positions come from any one game, so a few long
      games can't dominate the set.
    - The WHOLE file is read first and the game order is shuffled, so the
      order of games in the file (e.g. several PGNs pasted together) cannot
      bias which games get sampled.
    - `game_filter`, if given, is called with the 1-based game number and must
      return True for games that may be used. This is how run_baseline.py makes
      disjoint train/test splits: whole GAMES go to one side or the other, so
      no game contributes positions to both.
    - Deterministic for a given `seed`; duplicates (same FEN) are dropped.
    Returns fewer than `n` if the file doesn't contain enough games.
    (Holds every eligible position of the file in memory: fine for files of
    a few hundred games, not for multi-million-game databases.)
    """
    rng = random.Random(seed)
    games = []                       # (game_index, [(ply, fen), ...])
    game_index = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_index += 1
            if game_filter is not None and not game_filter(game_index):
                continue
            board = game.board()
            candidates = []
            for move in game.mainline_moves():
                board.push(move)
                ply = len(board.move_stack)
                if min_ply <= ply <= max_ply and not board.is_game_over() \
                        and board.legal_moves.count() >= 2:
                    candidates.append((ply, board.fen()))
            if candidates:
                games.append((game_index, candidates))

    rng.shuffle(games)
    positions, seen = [], set()
    for game_index, candidates in games:
        if len(positions) >= n:
            break
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


def select_positions(wac_path: str, pgn_path: str, wac_n: int, real_n: int,
                     split: str = "test", seed: int = 0) -> List[Position]:
    """
    The one place that decides which positions an experiment uses.

    split = "test"  : the held-out half. Report final results on this.
            "train" : the other half. Tune thresholds / lambda on THIS, never
                      on test (otherwise results are optimistic).
            "all"   : no split (what the first baseline run used).
    WAC is shuffled with `seed` and cut in half; real games are split by GAME
    number (odd = test, even = train), so no game feeds both sides.
    Returns up to wac_n WAC positions followed by up to real_n real-game ones.
    """
    if split not in ("test", "train", "all"):
        raise ValueError("split must be test, train or all")
    out: List[Position] = []
    if wac_n:
        wac = load_epd(wac_path)
        random.Random(seed).shuffle(wac)
        half = len(wac) // 2
        part = {"test": wac[:half], "train": wac[half:], "all": wac}[split]
        out += part[:wac_n]
    if real_n:
        flt = {"test": lambda g: g % 2 == 1, "train": lambda g: g % 2 == 0,
               "all": None}[split]
        out += sample_pgn_positions(pgn_path, real_n, seed=seed, game_filter=flt)
    return out

