"""
Phase 2, step 4: comparison-harness checks.

Run: python test_harness.py
Checks 1-3 need no Stockfish. Check 4 needs it (set STOCKFISH_PATH) and is
skipped with a message otherwise.

1. Statistics helpers (Wilson interval, bootstrap CI) behave sensibly.
2. summarize() gives the hand-computed numbers on synthetic results.
3. CSV save/load round-trips exactly (including "no book answer" = None).
4. Live run: nodes never exceed the budget, one row per run, losses >= 0,
   no-book positions get book_hit=None, re-running with the same CSV
   resumes (adds nothing), and resuming under different settings (oracle
   depth) is refused.
"""

import math
import os
import tempfile

import chess
from engine.harness import (Result, wilson, bootstrap_mean_ci, summarize,
                            load_results, _to_row, run_experiment, ENGINES, _FIELDS)
from engine.oracle import StockfishOracle, find_stockfish
from engine.positions import load_epd, Position
import csv


def mk(pos_id, budget, loss, book=None, match=False, depth=3, source="wac"):
    return Result("baseline", pos_id, source, budget, "e2e4", budget // 2, depth,
                  0.1, "e2e4", match, book, loss)


def check_stats():
    print("=" * 60)
    print("1. STATISTICS HELPERS")
    print("=" * 60)
    lo, hi = wilson(50, 100)
    print(f"  wilson(50/100) = [{lo:.3f}, {hi:.3f}]")
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002
    lo0, hi0 = wilson(0, 10)
    assert lo0 == 0.0 and 0.25 < hi0 < 0.34
    assert all(math.isnan(x) for x in wilson(0, 0))
    vals = [0, 0, 10, 20, 400, 0, 5, 30, 0, 100]
    a = bootstrap_mean_ci(vals)
    assert a == bootstrap_mean_ci(vals), "bootstrap must be deterministic"
    m = sum(vals) / len(vals)
    print(f"  bootstrap mean CI for mean {m}: [{a[0]:.1f}, {a[1]:.1f}]")
    assert a[0] <= m <= a[1]
    print("  PASS\n")


def check_summarize():
    print("=" * 60)
    print("2. SUMMARIZE MATCHES HAND-COMPUTED NUMBERS")
    print("=" * 60)
    rs = [mk("A", 1000, 0, True, True), mk("B", 1000, 300, False),
          mk("C", 1000, 100, True), mk("D", 1000, 0, None, True, depth=0),
          mk("R1", 1000, 50, None, False, source="real")]
    rows = {(r["source"], r["budget"]): r for r in summarize(rs)}
    w = rows[("wac", 1000)]
    assert w["n"] == 4 and w["failed"] == 1
    assert w["book_n"] == 3 and abs(w["book_solve"] - 2 / 3) < 1e-9
    assert abs(w["oracle_match"] - 0.5) < 1e-9
    assert abs(w["mean_loss"] - 100) < 1e-9 and w["median_loss"] == 50
    assert abs(w["big_miss"] - 0.25) < 1e-9
    r = rows[("real", 1000)]
    assert r["book_n"] == 0 and math.isnan(r["book_solve"])
    print("  WAC: n=4 failed=1 book solve 2/3 loss mean 100 median 50 big-miss 25%")
    print("  real: no book answers -> book solve is NaN, not 0. PASS\n")


def check_csv_roundtrip():
    print("=" * 60)
    print("3. CSV ROUND-TRIP")
    print("=" * 60)
    rs = [mk("A", 500, 12, True, True), mk("B", 500, 0, None, False, source="real")]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            for r in rs:
                w.writerow(_to_row(r))
        back = load_results(p)
    assert back == rs, "round-trip changed the data"
    assert back[1].book_hit is None
    print("  identical after save+load, None preserved. PASS\n")


def check_live_run():
    print("=" * 60)
    print("4. LIVE RUN (needs Stockfish)")
    print("=" * 60)
    try:
        find_stockfish()
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}\n")
        return
    wac = load_epd(os.path.join("data", "wac.epd"), limit=3)
    nobook = Position(chess.STARTING_FEN, "START", "real")
    positions = wac + [nobook]
    budgets = [300, 1500]
    with tempfile.TemporaryDirectory() as d, StockfishOracle(depth=8) as sf:
        out = os.path.join(d, "live.csv")
        only_baseline = {"baseline": ENGINES["baseline"]}
        rs = run_experiment(only_baseline, positions, budgets, sf, out_csv=out,
                            progress=lambda *_: None)
        assert len(rs) == len(positions) * len(budgets) == 8
        for r in rs:
            assert r.nodes <= r.budget, f"{r.pos_id}: {r.nodes} > {r.budget}"
            assert r.cp_loss >= 0
            if r.pos_id == "START":
                assert r.book_hit is None
            else:
                assert r.book_hit is not None
        calls = sf.calls
        again = run_experiment(only_baseline, positions, budgets, sf, out_csv=out,
                               progress=lambda *_: None)
        assert len(again) == 8 and len(load_results(out)) == 8, "resume added rows"
        assert sf.calls == calls, "resume re-ran Stockfish"
        print("  8 runs, nodes within budget, losses >= 0; resume added nothing. PASS")

        # settings guard: a different oracle depth must NOT be appended to the same file
        with StockfishOracle(depth=9) as other:
            try:
                run_experiment(only_baseline, positions, budgets, other, out_csv=out,
                               progress=lambda *_: None)
                raise AssertionError("mixed oracle settings were accepted")
            except ValueError as e:
                assert "different settings" in str(e)
        assert len(load_results(out)) == 8
        print("  resuming under a different oracle depth is refused, file untouched. PASS\n")


if __name__ == "__main__":
    check_stats()
    check_summarize()
    check_csv_roundtrip()
    check_live_run()
    print("Comparison harness verified.")
