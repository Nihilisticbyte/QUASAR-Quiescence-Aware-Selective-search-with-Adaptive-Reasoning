"""
Phase 4 checks: extension controllers and how they plug into the search.

Run: python test_extensions.py        (no Stockfish needed)

1. A controller that never extends gives exactly the plain search.
2. Extension-depth bookkeeping (the per-line cap) is arithmetic-correct.
3. CheckExtension fires on checks only, and respects its cap.
4. InstabilityExtension: frontier-only, window gate, ablation switches.
5. TriggerModel buckets and the cost-weighted rule.
6. Every registered engine runs: stays inside its node budget, returns a legal
   move, leaves the caller's board untouched, and is deterministic.
"""

import os
import chess
from engine.extensions import (NoExtension, CheckExtension, InstabilityExtension,
                               CostWeightedExtension, QuiescenceChecks, TriggerModel,
                               controller_factories, Controller)
from engine.harness import ENGINES
from engine.positions import load_epd
from engine.search import iterative_deepening, SearchStats, quiescence, MATE_SCORE

TWO_CHECKS = "4k3/8/8/8/8/8/Q7/R3K3 w - - 0 1"       # Qe2+ / Qa4+ ... available, not in check
CHECKING_MOVE_FEN = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"   # Ra8+ gives check


def ext_call(ctl, board, depth_after, ply, alpha=-10**6, beta=10**6):
    return ctl.extension(board, chess.Move.null(), depth_after, ply, alpha, beta, SearchStats())


def check_noext_equals_plain():
    print("=" * 60)
    print("1. NoExtension == PLAIN SEARCH")
    print("=" * 60)
    for p in load_epd(os.path.join("data", "wac.epd"), limit=4):
        a = iterative_deepening(p.board(), 3)
        b = iterative_deepening(p.board(), 3, controller=NoExtension())
        assert (a[0], a[1], a[2].nodes) == (b[0], b[1], b[2].nodes), p.id
        assert b[2].extensions == 0
    print("  same move, score and node count on 4 WAC positions. PASS\n")


def check_bookkeeping():
    print("=" * 60)
    print("2. PER-LINE EXTENSION COUNT")
    print("=" * 60)
    c = Controller()
    c.start_search(3)
    assert c.path_extensions(depth_after=1, ply=2) == 0     # 2 + 1 = 3 = root depth
    assert c.path_extensions(depth_after=2, ply=2) == 1     # one ply already added
    assert c.path_extensions(depth_after=4, ply=3) == 4
    print("  ply + depth - root_depth counts plies already added. PASS\n")


def check_check_extension():
    print("=" * 60)
    print("3. CheckExtension")
    print("=" * 60)
    ctl = CheckExtension(max_ext=2)
    ctl.start_search(3)
    checked = chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1")    # black king in check from Ra8
    quiet = chess.Board()
    assert checked.is_check() and ext_call(ctl, checked, 1, 2) == 1
    assert ext_call(ctl, quiet, 1, 2) == 0
    # line already extended twice (ply 3 + depth 2 - root 3 = 2): cap reached
    assert ext_call(ctl, checked, 2, 3) == 0
    print("  extends on check, not on quiet moves, stops at the cap. PASS\n")


def check_instability():
    print("=" * 60)
    print("4. InstabilityExtension")
    print("=" * 60)
    two = chess.Board(TWO_CHECKS)
    ctl = InstabilityExtension(min_checks=2, margin=150)
    ctl.start_search(3)
    # static value of this position for White is large (~ +1400): window must contain it
    assert ext_call(ctl, two, 0, 3, alpha=1000, beta=2000) == 1, "frontier + unstable + in window"
    assert ext_call(ctl, two, 1, 2, alpha=1000, beta=2000) == 0, "not at the frontier -> no extension"
    assert ext_call(ctl, two, 0, 3, alpha=-100, beta=100) == 0, "value far above window -> irrelevant"
    off = InstabilityExtension(min_checks=2, use_window=False)
    off.start_search(3)
    assert ext_call(off, two, 0, 3, alpha=-100, beta=100) == 1, "window ablation extends anyway"
    assert ext_call(ctl, chess.Board(), 0, 3, alpha=-100, beta=100) == 0, "no checks -> stable"
    # in check: only the check rule may fire (the instability rule defers to quiescence evasions)
    in_check = chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert ext_call(ctl, in_check, 0, 3) == 1
    nocheck = InstabilityExtension(min_checks=2, use_check=False)
    nocheck.start_search(3)
    assert ext_call(nocheck, in_check, 0, 3) == 0, "check-rule ablation"
    # cap
    capped = InstabilityExtension(min_checks=2, max_ext=1)
    capped.start_search(3)
    assert ext_call(capped, two, 0, 3, alpha=1000, beta=2000) == 1, "no plies added yet -> allowed"
    assert ext_call(capped, two, 0, 4, alpha=1000, beta=2000) == 0, "ply 4 + depth 0 - root 3 = 1 added -> capped"
    try:
        InstabilityExtension(min_checks=0, min_gain=0)
        raise AssertionError("a trigger with no condition must be rejected")
    except ValueError:
        pass
    print("  frontier-only, window gate, both ablations, cap, bad config rejected. PASS\n")


def check_cost_weighted():
    print("=" * 60)
    print("5. TriggerModel AND CostWeightedExtension")
    print("=" * 60)
    m = TriggerModel("n_checks", [0, 1, 2], [0.0, 0.5, 0.9], [0.0, 100.0, 300.0])
    assert m.p_change(0) == 0.0 and m.p_change(1) == 0.5 and m.p_change(7) == 0.9
    assert m.expected_shift(2) == 270.0 and m.expected_shift(1) == 50.0
    try:
        TriggerModel("n_checks", [0, 1], [0.1], [1.0])
        raise AssertionError("length mismatch must be rejected")
    except ValueError:
        pass
    try:
        TriggerModel.load("/nonexistent/model.json")
        raise AssertionError("missing model must raise")
    except FileNotFoundError:
        pass
    two = chess.Board(TWO_CHECKS)
    cheap = CostWeightedExtension(m, lam=10.0)
    cheap.start_search(3)
    dear = CostWeightedExtension(m, lam=1000.0)
    dear.start_search(3)
    assert ext_call(cheap, two, 0, 3, alpha=1000, beta=2000) == 1, "270cp expected shift beats a small lambda"
    assert ext_call(dear, two, 0, 3, alpha=1000, beta=2000) == 0, "...but not a huge lambda"
    assert ext_call(cheap, chess.Board(), 0, 3, alpha=-100, beta=100) == 0, "no checks -> never"
    print("  bucket lookup, expected shift, bad models rejected, lambda trades benefit for cost. PASS\n")


def check_quiescence_checks():
    print("=" * 60)
    print("5b. CHECKS IN QUIESCENCE")
    print("=" * 60)
    # White to move, back-rank mate by Ra8#. Captures-only quiescence cannot find it.
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    plain = quiescence(mate, -10**9, 10**9, SearchStats())
    with_checks = quiescence(mate, -10**9, 10**9, SearchStats(), checks=True)
    assert plain < 1000 and with_checks > MATE_SCORE - 10, (plain, with_checks)
    print(f"  mate in 1 by a quiet check: plain quiescence {plain}, with checks {with_checks}")

    two = chess.Board(TWO_CHECKS)
    always = QuiescenceChecks(min_checks=0)
    gated = QuiescenceChecks(min_checks=2)
    assert always.checks_at_frontier(chess.Board(), -100, 100), "unconditional: always"
    assert not gated.checks_at_frontier(chess.Board(), -100, 100), "no checks available"
    assert gated.checks_at_frontier(two, 1000, 2000), "enough checks and value in window"
    assert not gated.checks_at_frontier(two, -100, 100), "value far outside the window"
    assert not gated.checks_at_frontier(chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1"), -10**6, 10**6), \
        "in check: quiescence already searches evasions"
    off = QuiescenceChecks(min_checks=2, use_window=False)
    assert off.checks_at_frontier(two, -100, 100), "window ablation"
    # the instability part must NOT add depth, only the check rule may
    gated.start_search(3)
    assert ext_call(gated, two, 0, 3, alpha=1000, beta=2000) == 0
    assert ext_call(gated, chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1"), 1, 2) == 1
    print("  trigger, window gate, in-check exemption, no extra depth. PASS\n")


def check_registry():
    print("=" * 60)
    print("6. EVERY REGISTERED ENGINE RUNS SAFELY")
    print("=" * 60)
    positions = load_epd(os.path.join("data", "wac.epd"), limit=5)
    names = sorted(ENGINES)
    total_ext = {}
    for name in names:
        for p in positions:
            board = p.board()
            fen = board.fen()
            move, stats = ENGINES[name](board, 1500)
            assert stats.nodes <= 1500, (name, p.id, stats.nodes)
            assert move in board.legal_moves and board.fen() == fen, (name, p.id)
            move2, stats2 = ENGINES[name](p.board(), 1500)
            assert (move, stats.nodes, stats.extensions) == (move2, stats2.nodes, stats2.extensions), \
                f"{name} is not deterministic"
            total_ext[name] = total_ext.get(name, 0) + stats.extensions
    assert total_ext["baseline"] == 0
    assert total_ext["check"] > 0, "check extension never fired on 5 tactical positions"
    print("  engines:", ", ".join(f"{n}({total_ext[n]})" for n in names))
    print("  (number = extensions over 5 positions) budgets respected, deterministic. PASS\n")


if __name__ == "__main__":
    assert set(controller_factories()) <= set(ENGINES)
    check_noext_equals_plain()
    check_bookkeeping()
    check_check_extension()
    check_instability()
    check_cost_weighted()
    check_quiescence_checks()
    check_registry()
    print("Extension controllers verified.")
