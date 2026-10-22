"""
Transposition table (TT) for Phase 1.

Uses python-chess's built-in Zobrist hashing (chess.polyglot.zobrist_hash)
instead of hand-rolling one -- it's the same algorithm every serious engine
uses, and python-chess's implementation is already tested.

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

    def key(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def probe(self, board: chess.Board) -> Optional[TTEntry]:
        entry = self.table.get(self.key(board))
        if entry is not None:
            self.hits += 1
        return entry

    def store(self, board: chess.Board, depth: int, score: int,
              flag: Flag, best_move: Optional[chess.Move]):
        self.table[self.key(board)] = TTEntry(depth, score, flag, best_move)
        self.stores += 1

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0

    def __len__(self):
        return len(self.table)
