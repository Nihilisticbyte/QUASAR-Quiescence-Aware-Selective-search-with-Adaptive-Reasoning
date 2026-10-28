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
from dataclasses import dataclass
from .eval import evaluate
from .tt import TranspositionTable, Flag

SEARCH_VERSION = "4"   # bump whenever search behaviour changes: results files record it
                       # 1 = Phase 1 baseline; 2 = cheap draw/repetition, TT mate
                       # scores by ply, check evasions in quiescence, controllers;
                       # 3 = partial-iteration results, checks in quiescence (controller-gated);
                       # 4 = two-phase frontier: plain quiescence, then a controller-priced check pass

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
    node_limit: int = None   # hard cap on TOTAL nodes (alpha-beta + quiescence);
                             # None = unlimited (fixed-depth behaviour)
    completed_depth: int = 0 # deepest iterative-deepening iteration that finished
    partial_depth: int = 0   # depth of an iteration that was cut off by the budget but
                             # had finished at least one root move (its best move is used)
    extensions: int = 0      # times an extension controller added depth
    frontier_hook: object = None  # optional callable(board, ply), called at every
                                  # depth-0 node BEFORE quiescence (Phase 3 logging)


class SearchAborted(Exception):
    """Raised when the node budget runs out mid-search. Caught by
    iterative_deepening(). When raised out of search_root it also carries the
    best root move among those whose search had FINISHED (best_move is None if
    not even the first root move finished)."""
    best_move = None
    best_score = 0


MATE_BOUND = MATE_SCORE - 1000   # |score| beyond this means "a forced mate was found"


def to_tt(score: int, ply: int) -> int:
    """Mate scores are 'MATE_SCORE minus plies from the ROOT'. A TT entry may
    be reused at a different ply, so store them relative to the NODE
    (plies to mate from here) and convert back on read. Without this the
    engine can misjudge how fast a mate is when the entry is reused."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


def is_cheap_draw(board: chess.Board) -> bool:
    """Draws that need no move history: insufficient material, or the
    fifty-move rule (100 half-moves). Replaces board.can_claim_draw(), which
    replays moves to count repetitions and was measured at ~78% of all
    runtime. Repetitions are handled separately by RepetitionTracker."""
    return board.halfmove_clock >= 100 or board.is_insufficient_material()


class RepetitionTracker:
    """
    Counts how often each position occurred on the CURRENT path: the game
    history before the root plus the line being searched. A node whose
    position already occurred on that path is scored as a draw (two-fold
    repetition, the usual engine convention: if one side can repeat, it can
    force the third occurrence). Built once per root search from the board's
    move stack; alphabeta() adds/removes entries as it descends/returns.
    """

    def __init__(self, board: chess.Board, key_fn):
        self.counts = {}
        b = board.copy()
        keys = [key_fn(b)]
        while b.move_stack:
            b.pop()
            keys.append(key_fn(b))
        for k in keys:
            self.counts[k] = self.counts.get(k, 0) + 1


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
    capture, so the static eval is itself a valid lower bound on the score.
    That only holds OUTSIDE check. When the side to move is in check it may
    not stand pat (it must answer the check), so in check we search ALL legal
    moves (the evasions) and take no stand-pat bound. This fixes the Phase 1
    limitation where a position in check was scored as if it were quiet.
    """
    if stats.node_limit is not None and stats.nodes >= stats.node_limit:
        raise SearchAborted
    stats.nodes += 1
    stats.qnodes += 1

    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate() or is_cheap_draw(board):
        return 0

    stand_pat = evaluate(board, terminal_checks=False)
    stand_pat = stand_pat if board.turn == chess.WHITE else -stand_pat

    if board.is_check():
        if q_ply >= MAX_QUIESCENCE_PLY:
            return max(alpha, min(beta, stand_pat))   # safety cap: fall back to static
        moves = order_moves(board, list(board.legal_moves))   # evasions, no stand-pat
    else:
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if q_ply >= MAX_QUIESCENCE_PLY:
            return alpha
        moves = order_moves(board, list(board.generate_legal_captures()))

    for move in moves:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, stats, ply + 1, q_ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def check_pass(board: chess.Board, checks, base: int, alpha: int, beta: int,
               stats: SearchStats, ply: int) -> int:
    """
    Second phase of a frontier search: try the quiet CHECKING moves `checks`
    on top of a value `base` that plain quiescence already secured.

    Adding moves for the side to move can only RAISE its value (it takes the
    max over more options), so the result is max(base, best check) clipped to
    the window. That monotonicity is what lets a controller decide, from `base`
    and the window alone, whether this pass could possibly matter (see
    RWChecks in engine/extensions.py).
    """
    a = max(alpha, base)
    for move in checks:
        board.push(move)
        score = -quiescence(board, -beta, -a, stats, ply + 1, 1)
        board.pop()
        if score >= beta:
            return beta
        if score > a:
            a = score
    return a


def frontier_search(board: chess.Board, alpha: int, beta: int,
                    stats: SearchStats, ply: int, ctl=None) -> int:
    """
    Value of a depth-0 node. Plain quiescence first; then, only if the
    controller says the extra work is worth its price, the check pass.
    Nothing here is stored in the TT (depth-0 nodes are never stored), so a
    window-dependent decision cannot leak into the table.
    """
    q = quiescence(board, alpha, beta, stats, ply=ply)
    if ctl is None or q >= beta or board.is_check():
        # q >= beta: already a cutoff and extra moves can only raise it -> nothing
        # can change. In check: quiescence already searched every evasion.
        return q
    checks = ctl.frontier_checks(board, q, alpha, beta, stats)
    if not checks:
        return q
    stats.extensions += 1
    return check_pass(board, checks, q, alpha, beta, stats, ply)


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int,
               stats: SearchStats, tt: TranspositionTable, ply: int = 0,
               rep: RepetitionTracker = None, ctl=None) -> int:
    if stats.node_limit is not None and stats.nodes >= stats.node_limit:
        raise SearchAborted
    stats.nodes += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)

    if board.is_checkmate():
        stats.leaf_nodes += 1
        return -MATE_SCORE + ply
    if board.is_stalemate() or is_cheap_draw(board):
        stats.leaf_nodes += 1
        return 0

    key = tt.key(board)          # computed once, reused for probe/store/repetition
    if rep is not None and rep.counts.get(key, 0) >= 1:
        stats.leaf_nodes += 1
        return 0                 # position already occurred on this path: draw

    alpha_orig = alpha
    tt_move = None
    entry = tt.probe_key(key)
    if entry is not None:
        tt_move = entry.best_move
        if entry.depth >= depth:
            tt_score = from_tt(entry.score, ply)
            if entry.flag == Flag.EXACT:
                stats.tt_cutoffs += 1
                return tt_score
            elif entry.flag == Flag.LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif entry.flag == Flag.UPPERBOUND:
                beta = min(beta, tt_score)
            if alpha >= beta:
                stats.tt_cutoffs += 1
                return tt_score

    if depth == 0:
        stats.leaf_nodes += 1
        if stats.frontier_hook is not None:
            stats.frontier_hook(board, ply)
        return frontier_search(board, alpha, beta, stats, ply, ctl)

    best = -INF
    best_move = None
    moves = order_moves(board, list(board.legal_moves), tt_move)
    if rep is not None:
        rep.counts[key] = 1      # this node is now on the path
    for move in moves:
        board.push(move)
        child_depth = depth - 1
        if ctl is not None:
            ext = ctl.extension(board, move, child_depth, ply + 1, -beta, -alpha, stats)
            if ext:
                stats.extensions += 1
                child_depth += ext
        score = -alphabeta(board, child_depth, -beta, -alpha, stats, tt, ply + 1, rep, ctl)
        board.pop()

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # beta cutoff
    if rep is not None:
        del rep.counts[key]

    # Store what we learned into the TT.
    if best <= alpha_orig:
        flag = Flag.UPPERBOUND
    elif best >= beta:
        flag = Flag.LOWERBOUND
    else:
        flag = Flag.EXACT
    tt.store_key(key, depth, to_tt(best, ply), flag, best_move)

    return best


def search_root(board: chess.Board, depth: int, tt: TranspositionTable = None,
                 stats: SearchStats = None, ctl=None):
    """Returns (best_move, score, stats) for `board` at fixed `depth`.
    Pass a shared `tt`/`stats` across calls (e.g. from iterative_deepening)
    to reuse the table and accumulate node counts. `ctl` is an optional
    extension controller (engine/extensions.py); None = plain search.
    Scores are relative to the side to move."""
    if tt is None:
        tt = TranspositionTable()
    if stats is None:
        stats = SearchStats()

    best_move = None
    best_score = -INF
    alpha, beta = -INF, INF
    rep = RepetitionTracker(board, tt.key)

    entry = tt.probe(board)
    tt_move = entry.best_move if entry else None

    moves = order_moves(board, list(board.legal_moves), tt_move)
    if not moves:
        # No legal move: checkmate (we lost) or stalemate (draw). Side-to-move score.
        return None, (-MATE_SCORE if board.is_check() else 0), stats

    if ctl is not None:
        ctl.start_search(depth)
    for move in moves:
        board.push(move)
        child_depth = depth - 1
        if ctl is not None:
            ext = ctl.extension(board, move, child_depth, 1, -beta, -alpha, stats)
            if ext:
                stats.extensions += 1
                child_depth += ext
        try:
            score = -alphabeta(board, child_depth, -beta, -alpha, stats, tt, ply=1,
                               rep=rep, ctl=ctl)
        except SearchAborted as e:
            # Out of nodes. The board is mid-search and must not be reused (the
            # callers search a copy). Hand back the best FINISHED root move so
            # a cut-off iteration still counts: the first root move tried is the
            # previous iteration's best, and a later move only becomes "best" if
            # it beat it, so this result is never worse than the last iteration.
            e.best_move, e.best_score = best_move, best_score
            raise
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    tt.store(board, depth, best_score, Flag.EXACT, best_move)
    return best_move, best_score, stats


def iterative_deepening(board: chess.Board, max_depth: int,
                          tt: TranspositionTable = None,
                          node_budget: int = None, controller=None):
    """
    Searches depth 1, 2, 3, ... max_depth in sequence, reusing the same TT
    across iterations. Returns the final (deepest) iteration's result, plus
    a list of (depth, move, score) for every iteration -- useful for
    showing the PV firming up as depth increases, and is also where a
    later time-budget cutoff would plug in (stop between iterations if
    out of time, always returning the last COMPLETE iteration's move).

    node_budget: if given, search stops as soon as TOTAL nodes (alpha-beta +
    quiescence) would exceed it. If the cut-off iteration had already finished
    at least one root move, the best of those is returned (stats.partial_depth
    says which depth); otherwise the last COMPLETE iteration's move/score is
    returned. Set max_depth high
    (e.g. 64) to let the budget be the only limit. stats.completed_depth
    records how deep the search got. Each iteration runs on a copy of the
    board, so an abort can never leave the caller's board mid-search.

    controller: optional extension controller (see engine/extensions.py).
    None gives exactly the plain Phase 1 search.
    """
    if tt is None:
        tt = TranspositionTable()
    stats = SearchStats(node_limit=node_budget)

    history = []
    best_move, best_score = None, 0
    partial = None
    for depth in range(1, max_depth + 1):
        try:
            move, score, stats = search_root(board.copy(), depth, tt, stats,
                                             ctl=controller)
        except SearchAborted as e:
            if e.best_move is not None:
                partial = (e.best_move, e.best_score)
                stats.partial_depth = depth
            break
        best_move, best_score = move, score
        stats.completed_depth = depth
        history.append((depth, best_move, best_score))
    if partial is not None:
        best_move, best_score = partial

    if best_move is None:
        # Budget too small to even finish depth 1: still return a legal move
        # (first by move ordering) instead of None. Score is unknown -> 0.
        moves = order_moves(board, list(board.legal_moves))
        best_move = moves[0] if moves else None

    return best_move, best_score, stats, history