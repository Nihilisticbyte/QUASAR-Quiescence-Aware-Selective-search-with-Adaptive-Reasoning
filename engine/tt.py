"""
Transposition table (TT) for Phase 1.

Keyed by python-chess's own position key (board._transposition_key()): a
tuple of piece bitboards, side to move, castling rights and the en-passant
square (only when a capture is legal). Compared with chess.polyglot.zobrist_hash
this is ~20x faster in Python (0.6us vs 13us per call, measured) and cannot
suffer 64-bit hash collisions, because it is the exact position, not a hash.
If a python-chess version ever lacks that method, we fall back to the
Zobrist hash so nothing breaks.

A TT entry stores what we learned about a position the last time we
searched it: to what depth, what score, and whether that score was exact
or just a bound (because alpha-beta cutoffs mean we don't always get the
true value -- see Flag below).
"""

import enum
from dataclasses import dataclass
from typing import Optional
import chess
import chess.polyglot


if hasattr(chess.Board, "_transposition_key"):
    def _position_key(board: chess.Board):
        return board._transposition_key()
else:                                                   # pragma: no cover
    def _position_key(board: chess.Board):
        return chess.polyglot.zobrist_hash(board)


class Flag(enum.Enum):
    EXACT = 0       # the true minimax value of this position at this depth
    LOWERBOUND = 1  # a beta cutoff happened; true value is >= this score
    UPPERBOUND = 2  # no move raised alpha; true value is <= this score


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: Flag
    best_move: Optional[chess.Move]


class TranspositionTable:
    """A plain dict-backed TT, keyed by Zobrist hash. No replacement
    scheme yet (Phase 1 keeps it simple: always overwrite) -- a
    depth-preferred or aging replacement policy is a reasonable later
    improvement if memory/collisions become a problem."""

    def __init__(self):
        self.table: dict[int, TTEntry] = {}
        self.hits = 0
        self.stores = 0

    def key(self, board: chess.Board):
        return _position_key(board)

    def probe(self, board: chess.Board) -> Optional[TTEntry]:
        return self.probe_key(self.key(board))

    def probe_key(self, key) -> Optional[TTEntry]:
        """Same as probe() but with a key the caller already computed (the
        search computes it once per node and reuses it for probe, store and
        repetition detection)."""
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(self, board: chess.Board, depth: int, score: int,
              flag: Flag, best_move: Optional[chess.Move]):
        self.store_key(self.key(board), depth, score, flag, best_move)

    def store_key(self, key, depth: int, score: int,
                  flag: Flag, best_move: Optional[chess.Move]):
        self.table[key] = TTEntry(depth, score, flag, best_move)
        self.stores += 1

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0

    def __len__(self):
        return len(self.table)
