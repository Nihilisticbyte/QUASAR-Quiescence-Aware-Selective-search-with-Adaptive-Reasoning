"""
Phase 1 checks: TT correctness, node-count improvement vs Phase 0 baseline,
and the actual horizon-effect fix from quiescence search.

Run: python3 test_phase1.py
"""

import chess
from engine.search import search_root, iterative_deepening, quiescence, SearchStats
from engine.tt import TranspositionTable
from engine.eval import evaluate

# Phase 0 baseline node counts (plain alpha-beta, no TT, no quiescence),
# recorded in project-plan.md -- compare against these.
PHASE0_BASELINE = {1: 20, 2: 83, 3: 776, 4: 3108}


def check_still_legal():
    print("=" * 60)
    print("1. LEGALITY CHECK (search_root + iterative_deepening)")
    print("=" * 60)
    board = chess.Board()
    move, score, stats = search_root(board, 3)
    assert move in board.legal_moves
    print(f"  search_root depth=3: {board.san(move)} (legal, OK)")

    board = chess.Board()
    move, score, stats, history = iterative_deepening(board, 4)
    assert move in board.legal_moves
    print(f"  iterative_deepening to depth=4: {board.san(move)} (legal, OK)")
    for d, m, s in history:
        print(f"    depth {d}: {board.san(m):>6}  score={s:>5}")
    print()


def check_node_reduction():
    print("=" * 60)
    print("2. NODE COUNT vs PHASE 0 BASELINE")
    print("=" * 60)
    print("  Phase 0 never ran quiescence search, so comparing RAW totals")
    print("  isn't fair -- quiescence nodes are new work Phase 0 never did.")
    print("  Split: alphabeta-only nodes (fair vs phase0) vs total (the")
    print("  real cost of also fixing the horizon effect).\n")
    for depth, baseline_nodes in PHASE0_BASELINE.items():
        board = chess.Board()
        tt = TranspositionTable()
        move, score, stats = search_root(board, depth, tt)
        ab_nodes = stats.nodes - stats.qnodes
        direction = "fewer" if ab_nodes < baseline_nodes else "MORE"
        print(f"  depth {depth}: phase0_ab={baseline_nodes:>6}  "
              f"phase1_ab={ab_nodes:>6} ({direction})  "
              f"+qnodes={stats.qnodes:>6}  total={stats.nodes:>6}  "
              f"TT size={len(tt)}")
    print()


def check_quiescence_catches_recapture():
    print("=" * 60)
    print("3. QUIESCENCE CATCHES A RECAPTURE (the actual horizon-effect fix)")
    print("=" * 60)
    # White queen e2, White king e1, Black rook e7 (defends e5), Black
    # pawn e5, Black king e8. Qxe5 LOOKS like a free pawn if you stop
    # right after capturing -- but Rxe5 recaptures. Static evaluate()
    # right after Qxe5 can't see that; quiescence() must.
    board = chess.Board("4k3/4r3/8/4p3/8/8/4Q3/4K3 w - - 0 1")
    move = board.parse_san("Qxe5")
    board.push(move)  # board.turn is now BLACK

    # evaluate() is always White-relative, regardless of whose turn it is.
    naive_white_relative = evaluate(board)

    # quiescence() returns a side-to-move-relative score (negamax
    # convention), and side-to-move here is Black -- flip it to compare
    # on the same (White-relative) scale as naive_white_relative above.
    stats = SearchStats()
    black_relative = quiescence(board, -10**9, 10**9, stats)
    true_white_relative = -black_relative

    print("  position after Qxe5 (Black rook on e7 defends e5):")
    print(f"  naive snapshot (White's view, ignoring reply): "
          f"{naive_white_relative:+d}cp (looks good for White)")
    print(f"  quiescence's true value (White's view, after Rxe5): "
          f"{true_white_relative:+d}cp (Black just wins the queen)")
    print(f"  quiescence nodes spent resolving the exchange: {stats.qnodes}")

    assert true_white_relative < naive_white_relative, \
        "Quiescence should reveal this is worse for White than it first looks!"
    print("  -> quiescence correctly sees past the recapture. PASS\n")


def check_tt_scores_match():
    print("=" * 60)
    print("4. TT CONSISTENCY (searching twice should give the same score)")
    print("=" * 60)
    board = chess.Board()
    tt = TranspositionTable()
    move1, score1, _ = search_root(board, 3, tt)
    move2, score2, _ = search_root(board, 3, tt)
    print(f"  first search:  {board.san(move1)}  score={score1}")
    print(f"  second search: {board.san(move2)}  score={score2}  "
          f"(TT hits so far: {tt.hits})")
    assert score1 == score2, "Same position/depth should give the same score!"
    print("  -> consistent. PASS\n")


if __name__ == "__main__":
    check_still_legal()
    check_node_reduction()
    check_quiescence_catches_recapture()
    check_tt_scores_match()
    print("Phase 1 complete: TT + quiescence + iterative deepening verified.")
