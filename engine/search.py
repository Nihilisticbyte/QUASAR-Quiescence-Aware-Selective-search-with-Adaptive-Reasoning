"""
Phase 0 baseline: plain fixed-depth alpha-beta search (negamax form),
with a node counter and basic move ordering.

No transposition table, no iterative deepening, no quiescence search,
no extensions yet -- those come in later phases.
"""

import chess
from dataclasses import dataclass
from .eval import evaluate

MATE_SCORE = 100000
INF = 10 ** 9


@dataclass
class SearchStats:
    nodes: int = 0
    leaf_nodes: int = 0
    max_depth_reached: int = 0


def order_moves(board: chess.Board, moves):
    """Cheap move ordering: captures first (roughly MVV-LVA), quiet after."""
    def score(move):
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            if victim is None:
                victim = chess.PAWN  # en passant
            return 10 * (victim or 0) - (attacker or 0)
        return -1

    return sorted(moves, key=score, reverse=True)


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int,
               stats: SearchStats, ply: int = 0) -> int:
    stats.nodes += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)

    if board.is_checkmate():
        stats.leaf_nodes += 1
        return -MATE_SCORE + ply
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.can_claim_draw():
        stats.leaf_nodes += 1
        return 0

    if depth == 0:
        stats.leaf_nodes += 1
        val = evaluate(board)
        return val if board.turn == chess.WHITE else -val

    best = -INF
    moves = order_moves(board, list(board.legal_moves))
    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, stats, ply + 1)
        board.pop()

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    return best


def search_root(board: chess.Board, depth: int):
    """Returns (best_move, score, stats) for `board` at fixed `depth`."""
    stats = SearchStats()
    best_move = None
    best_score = -INF
    alpha, beta = -INF, INF

    moves = order_moves(board, list(board.legal_moves))
    if not moves:
        return None, evaluate(board), stats

    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, stats, ply=1)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return best_move, best_score, stats
