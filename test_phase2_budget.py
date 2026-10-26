"""
Phase 2, step 1: node-budgeted search checks.

Run: python test_phase2_budget.py

1. The budget is a hard cap: total nodes never exceed it.
2. The returned move is legal and the caller's board is left untouched
   (an abort mid-search must not corrupt the board).
3. More budget -> same or deeper completed depth.
4. A tiny budget (can't finish depth 1) still returns a legal move.
5. With a budget large enough to finish, the result equals plain
   iterative deepening to the same depth (budget logic changes nothing
   when it never fires).
"""

import chess
from engine.search import iterative_deepening

POSITIONS = {
    "startpos": chess.STARTING_FEN,
    "middlegame": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "tactical": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
}
BUDGETS = [50, 500, 2000, 8000]


def check_cap_and_board_intact():
    print("=" * 60)
    print("1+2. BUDGET IS A HARD CAP, MOVE LEGAL, BOARD UNTOUCHED")
    print("=" * 60)
    for name, fen in POSITIONS.items():
        for budget in BUDGETS:
            board = chess.Board(fen)
            before = board.fen()
            move, score, stats, hist = iterative_deepening(
                board, 64, node_budget=budget)
            assert stats.nodes <= budget, \
                f"{name}: used {stats.nodes} nodes with budget {budget}"
            assert move in board.legal_moves, f"{name}: illegal move {move}"
            assert board.fen() == before, f"{name}: board was modified!"
            print(f"  {name:<10} budget={budget:>5}: nodes={stats.nodes:>5} "
                  f"depth={stats.completed_depth} move={board.san(move)}")
    print("  -> PASS\n")


def check_more_budget_not_shallower():
    print("=" * 60)
    print("3. MORE BUDGET -> NOT SHALLOWER")
    print("=" * 60)
    for name, fen in POSITIONS.items():
        depths = []
        for budget in BUDGETS:
            board = chess.Board(fen)
            _, _, stats, _ = iterative_deepening(board, 64, node_budget=budget)
            depths.append(stats.completed_depth)
        print(f"  {name:<10} depths at budgets {BUDGETS}: {depths}")
        assert depths == sorted(depths), f"{name}: depth went DOWN with more budget"
    print("  -> PASS\n")


def check_tiny_budget_fallback():
    print("=" * 60)
    print("4. TINY BUDGET STILL RETURNS A LEGAL MOVE")
    print("=" * 60)
    board = chess.Board()
    move, score, stats, hist = iterative_deepening(board, 64, node_budget=5)
    assert move in board.legal_moves
    assert stats.completed_depth == 0 and hist == []
    print(f"  budget=5: no iteration finished, fallback move "
          f"{board.san(move)} (legal).")

    # Depth 1 needs ~40 nodes from the start position. With 30 the iteration is
    # cut off, but some root moves had finished: the best of THOSE is returned
    # (partial result), not an arbitrary first move.
    board = chess.Board()
    fen = board.fen()
    move, score, stats, hist = iterative_deepening(board, 64, node_budget=30)
    assert stats.completed_depth == 0 and stats.partial_depth == 1, stats
    assert move in board.legal_moves and board.fen() == fen and stats.nodes <= 30
    print(f"  budget=30: depth 1 cut off after some root moves; partial result "
          f"{board.san(move)} used (legal, board untouched). PASS\n")


def check_big_budget_matches_plain():
    print("=" * 60)
    print("5. BUDGET THAT NEVER FIRES == PLAIN ITERATIVE DEEPENING")
    print("=" * 60)
    for name, fen in POSITIONS.items():
        plain_board = chess.Board(fen)
        pm, ps, pstats, _ = iterative_deepening(plain_board, 3)
        b_board = chess.Board(fen)
        bm, bs, bstats, _ = iterative_deepening(b_board, 3, node_budget=10**9)
        assert (pm, ps) == (bm, bs), f"{name}: results differ"
        assert pstats.nodes == bstats.nodes, f"{name}: node counts differ"
        print(f"  {name:<10} same move/score/nodes ({pstats.nodes}). OK")
    print("  -> PASS\n")


def check_partial_iteration():
    print("=" * 60)
    print("6. A CUT-OFF ITERATION STILL COUNTS (best finished root move)")
    print("=" * 60)
    seen_partial = 0
    for name, fen in POSITIONS.items():
        for budget in range(150, 4001, 250):
            board = chess.Board(fen)
            move, score, stats, hist = iterative_deepening(board, 64, node_budget=budget)
            assert stats.nodes <= budget and move in board.legal_moves
            if stats.partial_depth:
                seen_partial += 1
                # a partial iteration is always the one AFTER the last complete one
                assert stats.partial_depth == stats.completed_depth + 1, (name, budget)
        big = iterative_deepening(chess.Board(fen), 3, node_budget=10**9)[2]
        assert big.partial_depth == 0, "no iteration is cut off when the budget never fires"
    assert seen_partial > 0, "sweep never produced a partial iteration (test would be vacuous)"
    print(f"  {seen_partial} budgets ended inside an iteration; partial depth = completed + 1 each time. PASS\n")


if __name__ == "__main__":
    check_cap_and_board_intact()
    check_more_budget_not_shallower()
    check_tiny_budget_fallback()
    check_big_budget_matches_plain()
    check_partial_iteration()
    print("Node-budgeted search verified.")
