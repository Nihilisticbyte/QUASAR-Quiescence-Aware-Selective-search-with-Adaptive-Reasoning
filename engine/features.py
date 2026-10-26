"""
Instability features (Phase 3).

A feature here is a cheap number computed from a POSITION ONLY -- never from
how the search reached it (no parent value, no last move, no alpha/beta).
That is deliberate: the transposition table stores one result per position
and depth, so anything that made the same position look different on
different paths would let those paths disagree about its depth (see
project-plan.md, loophole 7). Window relevance (alpha/beta) is handled
separately in engine/extensions.py, as a gate on top of these features.

What they try to capture: "is this position tactically unsettled, i.e. would
searching one ply deeper probably change its value?"
  in_check          side to move is in check (forced reply, classic extension)
  n_captures        legal captures available
  n_good_captures   captures that win material on the first step of an
                    exchange: victim undefended, or victim worth more than
                    the capturing piece
  max_gain          best such immediate gain in centipawns (0 if none). This is
                    a one-step static-exchange estimate, not a full SEE.
  threat_gain       best capture the OPPONENT would have if it were their move
                    (a "null-move" threat); 0 when we are in check. Quiescence
                    cannot see this: it lets the side to move stand pat as if
                    nothing were attacked, so a pending threat makes the
                    static value too optimistic.
  n_checks          legal moves that give check (forcing moves)
  static_stm        static evaluation, side-to-move relative
  n_legal           number of legal moves (used as a cost proxy for the
                    cost-weighted controller)

Inspired by (not adapted from) Clune's idea that stable quantities make good
evaluation features; here we look for UNstable positions. See
engine/stability.py for the offline stability diagnostic.
"""

from dataclasses import dataclass, asdict

import chess

from .eval import evaluate, PIECE_VALUES


@dataclass(frozen=True)
class Features:
    in_check: bool
    n_captures: int
    n_good_captures: int
    max_gain: int
    threat_gain: int
    n_checks: int
    static_stm: int
    n_legal: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d["in_check"] = int(self.in_check)
        return d


FEATURE_NAMES = ["in_check", "n_captures", "n_good_captures", "max_gain",
                 "threat_gain", "n_checks", "static_stm", "n_legal"]


def capture_gain(board: chess.Board, move: chess.Move) -> int:
    """First-step material gain of capture `move` for the side to move:
    the victim's value if it is undefended, else victim minus attacker
    (negative = a losing trade on the first step)."""
    if board.is_en_passant(move):
        victim = PIECE_VALUES[chess.PAWN]
    else:
        victim = PIECE_VALUES[board.piece_type_at(move.to_square)]
    attacker = PIECE_VALUES[board.piece_type_at(move.from_square)]
    if board.is_attacked_by(not board.turn, move.to_square):
        return victim - attacker
    return victim


def max_capture_gain(board: chess.Board) -> int:
    """Cheapest instability signal: best immediate capture gain (>= 0).
    Used by the extension controllers at the search frontier."""
    best = 0
    for move in board.generate_legal_captures():
        g = capture_gain(board, move)
        if g > best:
            best = g
    return best


def threat_gain(board: chess.Board) -> int:
    """Best capture gain the opponent would have after a null move, i.e. what
    they threaten right now. 0 if the side to move is in check (a null move is
    illegal then, and the check itself is the dominant threat)."""
    if board.is_check():
        return 0
    board.push(chess.Move.null())
    try:
        return max_capture_gain(board)
    finally:
        board.pop()


def count_checks(board: chess.Board, limit: int = None) -> int:
    """Number of legal moves that give check. With `limit`, stops counting as
    soon as it is reached (the controllers only need "at least k checks"),
    which cuts the cost of the test roughly in half on average."""
    n = 0
    for move in board.legal_moves:
        if board.gives_check(move):
            n += 1
            if limit is not None and n >= limit:
                break
    return n


def position_features(board: chess.Board) -> Features:
    n_cap = n_good = n_checks = n_legal = 0
    max_gain = 0
    for move in board.legal_moves:
        n_legal += 1
        if board.gives_check(move):
            n_checks += 1
        if board.is_capture(move):
            n_cap += 1
            g = capture_gain(board, move)
            if g > 0:
                n_good += 1
                if g > max_gain:
                    max_gain = g
    stm = evaluate(board, terminal_checks=False)
    if board.turn == chess.BLACK:
        stm = -stm
    return Features(in_check=board.is_check(), n_captures=n_cap,
                    n_good_captures=n_good, max_gain=max_gain,
                    threat_gain=threat_gain(board), n_checks=n_checks, static_stm=stm, n_legal=n_legal)
