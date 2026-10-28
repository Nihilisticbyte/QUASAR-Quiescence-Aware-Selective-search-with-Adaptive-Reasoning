"""
Extension controllers and how they plug into the search.

Run: python test_extensions.py        (no Stockfish needed)

1. NoExtension gives exactly the plain search.
2. The per-line extension count is arithmetic-correct.
3. CheckExtension fires on checks only, and respects its cap.
4. The two-phase frontier search: a quiet-check mate is found by the check
   pass; a node that already reaches beta never reaches the controller.
5. ErrorModel: bucket lookup, expected gain inside the window, persistence.
6. RWChecks: gain vs price per check, relevance (window), forcing order,
   budget pressure, ablations.
7. Every registered engine runs: stays inside its node budget, returns a
   legal move, leaves the caller's board untouched, and is deterministic.
"""

import os
import tempfile
import chess
from engine.error_model import ErrorModel, fit, quantile_grid
from engine.extensions import (NoExtension, CheckExtension, BlindChecks, RWChecks,
                               Controller, controller_factories, MAX_PRESSURE)
from engine.harness import ENGINES
from engine.positions import load_epd
from engine.search import (iterative_deepening, SearchStats, quiescence,
                           frontier_search, MATE_SCORE)

TWO_CHECKS = "4k3/8/8/8/8/8/Q7/R3K3 w - - 0 1"       # several quiet checks available, not in check
BACK_RANK = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"      # Ra8# is mate in 1 by a quiet check


def ext_call(ctl, board, depth_after, ply, alpha=-10**6, beta=10**6):
    return ctl.extension(board, chess.Move.null(), depth_after, ply, alpha, beta, SearchStats())


def flat_model(e=300.0, cost=10.0):
    """One bucket: searching any check moves the value by exactly `e` and costs `cost` nodes."""
    return ErrorModel([0], [[e] * 20], [cost], [100])


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
    assert c.path_extensions(depth_after=1, ply=2) == 0
    assert c.path_extensions(depth_after=2, ply=2) == 1
    assert c.path_extensions(depth_after=4, ply=3) == 4
    print("  ply + depth - root_depth counts plies already added. PASS\n")


def check_check_extension():
    print("=" * 60)
    print("3. CheckExtension")
    print("=" * 60)
    ctl = CheckExtension(max_ext=2)
    ctl.start_search(3)
    checked = chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert checked.is_check() and ext_call(ctl, checked, 1, 2) == 1
    assert ext_call(ctl, chess.Board(), 1, 2) == 0
    assert ext_call(ctl, checked, 2, 3) == 0, "cap reached"
    print("  extends on check, not on quiet moves, stops at the cap. PASS\n")


def check_frontier_search():
    print("=" * 60)
    print("4. TWO-PHASE FRONTIER SEARCH")
    print("=" * 60)
    mate = chess.Board(BACK_RANK)
    plain = quiescence(mate, -10**9, 10**9, SearchStats())
    blind = frontier_search(mate, -10**9, 10**9, SearchStats(), 0, BlindChecks())
    assert plain < 1000 and blind > MATE_SCORE - 10, (plain, blind)
    print(f"  mate in 1 by a quiet check: plain quiescence {plain}, with check pass {blind}")

    class Spy(Controller):
        calls = 0

        def frontier_checks(self, *a):
            Spy.calls += 1
            return []

    # static value of this position is huge for White: with beta = 100 plain quiescence already cuts off
    frontier_search(chess.Board(TWO_CHECKS), -100, 100, SearchStats(), 0, Spy())
    assert Spy.calls == 0, "q >= beta: nothing can change, controller must not be asked"
    frontier_search(chess.Board(TWO_CHECKS), -10**6, 10**6, SearchStats(), 0, Spy())
    assert Spy.calls == 1
    frontier_search(chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1"), -10**6, 10**6, SearchStats(), 0, Spy())
    assert Spy.calls == 1, "in check: evasions already searched, controller not asked"
    s = SearchStats()
    assert frontier_search(chess.Board(BACK_RANK), -10**9, 10**9, s, 0, None) == plain and s.extensions == 0
    print("  quiet-check mate found; q>=beta and in-check skip the controller. PASS\n")


def check_error_model():
    print("=" * 60)
    print("5. ErrorModel")
    print("=" * 60)
    m = ErrorModel([1, 3], [[0.0, 100.0], [0.0, 400.0]], [10.0, 50.0], [5, 5])
    assert m.bucket(1) == 0 and m.bucket(2) == 0 and m.bucket(3) == 1 and m.bucket(9) == 1
    assert m.cost(2) == 10.0 and m.cost(4) == 50.0
    # bucket 0, base 0, window (-50, 50): e=0 gains 0; e=100 is clipped to the window top -> 50. mean 25
    assert m.expected_gain(1, 0, -50, 50) == 25.0
    # window entirely above what any error can reach -> nothing to gain
    assert m.expected_gain(1, 0, 200, 1000) == 0.0
    # base already at/over beta -> nothing to gain
    assert m.expected_gain(1, 60, -50, 50) == 0.0
    # base below alpha: only the part of the error that climbs INTO the window counts
    assert m.expected_gain(3, -100, 0, 1000) == 150.0   # (0 + (300-0)) / 2
    for bad in (lambda: ErrorModel([1, 2], [[0.0]], [1.0], [1]), lambda: ErrorModel([], [], [], [])):
        try:
            bad()
            raise AssertionError("bad model must be rejected")
        except ValueError:
            pass
    rows = [{"evasions": 1, "e": 0, "cost": 4}, {"evasions": 1, "e": 200, "cost": 6},
            {"evasions": 4, "e": 50, "cost": 30}]
    f = fit(rows, edges=[1, 3], k=2)
    assert f.n == [2, 1] and f.cost_nodes == [5.0, 30.0] and f.quantiles[0] == [0, 200]
    assert quantile_grid([], 3) == [0.0, 0.0, 0.0]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        m.save(path)
        m2 = ErrorModel.load(path)
        assert (m2.edges, m2.quantiles, m2.cost_nodes) == (m.edges, m.quantiles, m.cost_nodes)
        try:
            ErrorModel.load(os.path.join(d, "missing.json"))
            raise AssertionError("missing model must raise")
        except FileNotFoundError:
            pass
    print("  buckets, clipped gain, relevance, fit, save/load, bad input. PASS\n")


def check_rw_controller():
    print("=" * 60)
    print("6. RWChecks")
    print("=" * 60)
    two = chess.Board(TWO_CHECKS)
    fresh = SearchStats()
    # window (-50, 50), base 0, error 300: expected gain = 50. price = lam * 10 nodes.
    got = RWChecks(flat_model(), lam=1).frontier_checks(two, 0, -50, 50, fresh)
    assert got and all(two.gives_check(m) and not two.is_capture(m) for m in got), "gain 50 >= price 10"
    assert not RWChecks(flat_model(), lam=10).frontier_checks(two, 0, -50, 50, fresh), "gain 50 < price 100"
    assert not RWChecks(flat_model(), lam=1).frontier_checks(two, 0, 5000, 6000, fresh), \
        "window far above any reachable value: irrelevant"
    assert RWChecks(flat_model(), lam=10, use_window=False).frontier_checks(two, 0, 5000, 6000, fresh), \
        "window ablation counts the full 300cp"
    assert not RWChecks(flat_model(), lam=1).frontier_checks(chess.Board(), 0, -50, 50, fresh), "no quiet checks"

    assert RWChecks.pressure(SearchStats()) == 1.0, "no budget -> price never rises"
    assert RWChecks.pressure(SearchStats(nodes=0, node_limit=1000)) == 1.0
    assert abs(RWChecks.pressure(SearchStats(nodes=500, node_limit=1000)) - 2.0) < 1e-9
    assert RWChecks.pressure(SearchStats(nodes=999, node_limit=1000)) == MAX_PRESSURE
    late = SearchStats(nodes=900, node_limit=1000)        # pressure 10 -> price 100 > gain 50
    assert not RWChecks(flat_model(), lam=1).frontier_checks(two, 0, -50, 50, late)
    assert RWChecks(flat_model(), lam=1, budget_aware=False).frontier_checks(two, 0, -50, 50, late), \
        "budget ablation ignores the remaining budget"

    # forcing-ness: a mating check (0 replies) is worth its 1 node even at a high price,
    # a loose check (model says it gains nothing) is never searched
    forcing = ErrorModel([0, 1], [[1000.0] * 20, [0.0] * 20], [1.0, 10.0], [10, 10])
    mate = chess.Board(BACK_RANK)
    assert [m.uci() for m in RWChecks(forcing, lam=100).frontier_checks(mate, 145, -10**6, 10**6, fresh)] == ["a1a8"]
    assert not RWChecks(forcing, lam=1).frontier_checks(two, 0, -10**6, 10**6, fresh), \
        "checks with replies available gain nothing in this model"
    ordered = ErrorModel([0, 1, 2], [[0.0] * 20, [500.0] * 20, [500.0] * 20], [1.0, 1.0, 1.0], [1, 1, 1])
    mv = RWChecks(ordered, lam=1).frontier_checks(two, 0, -10**6, 10**6, fresh)
    counts = []
    for m in mv:
        two.push(m); counts.append(two.legal_moves.count()); two.pop()
    assert counts == sorted(counts), "most forcing check first"

    ctl = RWChecks(flat_model(), lam=1)
    ctl.start_search(3)
    in_check = chess.Board("R3k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert ext_call(ctl, in_check, 1, 2) == 1, "check rule is kept"
    assert ext_call(RWChecks(flat_model(), use_check=False), in_check, 1, 2) == 0, "check-rule ablation"
    print("  gain vs price, window relevance, mate-in-1 priced at 1 node, forcing order, budget pressure, ablations. PASS\n")


def check_registry():
    print("=" * 60)
    print("7. EVERY REGISTERED ENGINE RUNS SAFELY")
    print("=" * 60)
    positions = load_epd(os.path.join("data", "wac.epd"), limit=5)
    total_ext = {}
    for name in sorted(ENGINES):
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
    assert total_ext["check"] > 0 and total_ext["qchk_all"] > total_ext["check"]
    assert total_ext["rw_l0.1"] >= total_ext["rw_l2"], "a cheaper price extends at least as often"
    print("  engines:", ", ".join(f"{n}({total_ext[n]})" for n in sorted(ENGINES)))
    print("  (number = extensions over 5 positions) budgets respected, deterministic. PASS\n")


if __name__ == "__main__":
    assert set(controller_factories()) <= set(ENGINES)
    check_noext_equals_plain()
    check_bookkeeping()
    check_check_extension()
    check_frontier_search()
    check_error_model()
    check_rw_controller()
    check_registry()
    print("Extension controllers verified.")
