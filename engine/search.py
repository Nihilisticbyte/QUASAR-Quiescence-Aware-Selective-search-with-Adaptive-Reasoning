"""
Phase 1 search: alpha-beta + transposition table + captures-only
quiescence search + iterative deepening.

Changes from Phase 0 (for reference -- see project-plan.md for the full
diff description):
  - alphabeta() now probes/stores the TT, and orders the TT-recommended
    move first.
  - depth == 0 now calls quiescence() instead of evaluate() directly --
    this is the actual horizon-effect fix.
  - search_root() takes an optional `tt` so the table persists and is
    reused across iterative-deepening iterations (that reuse is most of
    where the speedup comes from: deeper iterations profit from shallower
    ones' TT entries for move ordering).
  - iterative_deepening() is new: calls search_root() for depth=1..max,
    reusing the same TT and stats across iterations.
"""

import chess
from dataclasses import dataclass, field
from .eval import evaluate
from .tt import TranspositionTable, Flag

MATE_SCORE = 100000
INF = 10 ** 9
MAX_QUIESCENCE_PLY = 16  # safety cap; captures-only is naturally bounded
                          # by remaining material, but this guards against
                          # any edge case (e.g. future check-extensions)


@dataclass
class SearchStats:
    nodes: int = 0
    leaf_nodes: int = 0
    qnodes: int = 0          # nodes spent inside quiescence search
    max_depth_reached: int = 0
    tt_cutoffs: int = 0      # times a TT entry let us skip a re-search


def order_moves(board: chess.Board, moves, tt_move: chess.Move = None):
    """TT-recommended move first, then captures (MVV-LVA-ish), then quiet."""
    def score(move):
        if tt_move is not None and move == tt_move:
            return 10_000  # always search the TT's suggestion first
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            if victim is None:
                victim = chess.PAWN  # en passant
            return 10 * (victim or 0) - (attacker or 0)
        return -1

    return sorted(moves, key=score, reverse=True)


def quiescence(board: chess.Board, alpha: int, beta: int,
                stats: SearchStats, ply: int = 0, q_ply: int = 0) -> int:
    """
    Captures-only quiescence search. Called at every depth==0 leaf instead
    of evaluate() directly -- this is what stops the engine from stopping
    mid-exchange and misjudging the position (the horizon effect).

    Two separate counters: `ply` is the TOTAL distance from the root (used
    for mate-distance scores, so they match the main search), `q_ply` is
    how deep we are inside quiescence only (used for the safety cap).

    Uses the standard "stand pat" idea: the side to move isn't FORCED to
    capture, so the static eval is itself a valid lower bound on the score
    (assuming there's always a quiet move available, which is true outside
    of check -- check-evasion in quiescence is a known Phase 1 limitation,
    see project-plan.md).
    """
    stats.nodes += 1
    stats.qnodes += 1

    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.can_claim_draw():
        return 0

    stand_pat = evaluate(board)
    stand_pat = stand_pat if board.turn == chess.WHITE else -stand_pat

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if q_ply >= MAX_QUIESCENCE_PLY:
        return alpha

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    captures = order_moves(board, captures)

    for move in captures:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, stats, ply + 1, q_ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int,
               stats: SearchStats, tt: TranspositionTable, ply: int = 0) -> int:
    stats.nodes += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)

    if board.is_checkmate():
        stats.leaf_nodes += 1
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.can_claim_draw():
        stats.leaf_nodes += 1
        return 0

    alpha_orig = alpha
    tt_move = None
    entry = tt.probe(board)
    if entry is not None:
        tt_move = entry.best_move
        if entry.depth >= depth:
            if entry.flag == Flag.EXACT:
                stats.tt_cutoffs += 1
                return entry.score
            elif entry.flag == Flag.LOWERBOUND:
                alpha = max(alpha, entry.score)
            elif entry.flag == Flag.UPPERBOUND:
                beta = min(beta, entry.score)
            if alpha >= beta:
                stats.tt_cutoffs += 1
                return entry.score

    if depth == 0:
        stats.leaf_nodes += 1
        return quiescence(board, alpha, beta, stats, ply=ply)

    best = -INF
    best_move = None
    moves = order_moves(board, list(board.legal_moves), tt_move)
    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, stats, tt, ply + 1)
        board.pop()

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # beta cutoff

    # Store what we learned into the TT.
    if best <= alpha_orig:
        flag = Flag.UPPERBOUND
    elif best >= beta:
        flag = Flag.LOWERBOUND
    else:
        flag = Flag.EXACT
    tt.store(board, depth, best, flag, best_move)

    return best


def search_root(board: chess.Board, depth: int, tt: TranspositionTable = None,
                 stats: SearchStats = None):
    """Returns (best_move, score, stats) for `board` at fixed `depth`.
    Pass a shared `tt`/`stats` across calls (e.g. from iterative_deepening)
    to reuse the table and accumulate node counts."""
    if tt is None:
        tt = TranspositionTable()
    if stats is None:
        stats = SearchStats()

    best_move = None
    best_score = -INF
    alpha, beta = -INF, INF

    entry = tt.probe(board)
    tt_move = entry.best_move if entry else None

    moves = order_moves(board, list(board.legal_moves), tt_move)
    if not moves:
        return None, evaluate(board), stats

    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, stats, tt, ply=1)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    tt.store(board, depth, best_score, Flag.EXACT, best_move)
    return best_move, best_score, stats


def iterative_deepening(board: chess.Board, max_depth: int,
                          tt: TranspositionTable = None):
    """
    Searches depth 1, 2, 3, ... max_depth in sequence, reusing the same TT
    across iterations. Returns the final (deepest) iteration's result, plus
    a list of (depth, move, score) for every iteration -- useful for
    showing the PV firming up as depth increases, and is also where a
    later time-budget cutoff would plug in (stop between iterations if
    out of time, always returning the last COMPLETE iteration's move).
    """
    if tt is None:
        tt = TranspositionTable()
    stats = SearchStats()

    history = []
    best_move, best_score = None, 0
    for depth in range(1, max_depth + 1):
        best_move, best_score, stats = search_root(board, depth, tt, stats)
        history.append((depth, best_move, best_score))

    return best_move, best_score, stats, history