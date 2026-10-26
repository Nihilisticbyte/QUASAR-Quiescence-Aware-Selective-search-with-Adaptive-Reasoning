"""
Phase 3 checks: instability features, the stability diagnostic, ground-truth
labelling, and the analysis that builds the trigger model.

Run: python test_phase3.py        (no Stockfish needed)
"""

import json
import math
import os
import random
import tempfile

import chess
from engine.extensions import TriggerModel
from engine.features import position_features, count_checks, threat_gain
from engine.stability import stability_quotient
from engine.trigger_analysis import auc, bucket_model, threshold_rows
from engine.trigger_log import label_position, sample_frontier_fens


def check_features():
    print("=" * 60)
    print("1. FEATURES ON HAND-CHECKED POSITIONS")
    print("=" * 60)
    f = position_features(chess.Board())
    assert (f.n_legal, f.n_captures, f.max_gain, f.n_checks, f.in_check) == (20, 0, 0, 0, False)
    assert f.threat_gain == 0
    print("  start position: 20 moves, no captures, no checks, no threats")

    # White rook d1 attacks an undefended black queen on d5.
    f = position_features(chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"))
    assert f.max_gain == 900 and f.n_good_captures == 1, f
    print("  free queen: max_gain 900, 1 good capture")

    # Black queen d4 attacks the undefended white rook a1; Ra8+ is White's only check.
    b = chess.Board("4k3/8/8/8/3q4/8/8/R3K3 w - - 0 1")
    f = position_features(b)
    assert f.threat_gain == 500 and f.n_checks == 1 and not f.in_check, f
    assert threat_gain(b) == 500 and b.fen() == "4k3/8/8/8/3q4/8/8/R3K3 w - - 0 1", \
        "threat_gain must leave the board unchanged"
    print("  hanging rook: threat_gain 500, exactly 1 check (Ra8+), board unchanged")

    # In check: threat_gain is defined as 0 (a null move would be illegal).
    f = position_features(chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"))
    assert f.in_check and f.threat_gain == 0
    print("  in check: in_check True, threat_gain 0")

    two = chess.Board("4k3/8/8/8/8/8/Q7/R3K3 w - - 0 1")
    assert count_checks(two) >= 2 and count_checks(two, limit=1) == 1
    print(f"  count_checks: {count_checks(two)} checks; limit=1 stops at 1. PASS\n")


def check_stability():
    print("=" * 60)
    print("2. STABILITY QUOTIENT (total vs adjacent variance)")
    print("=" * 60)
    tv, av, q = stability_quotient([list(range(10))])         # smooth ramp
    assert abs(tv - 8.25) < 1e-9 and abs(av - 1.0) < 1e-9 and abs(q - 8.25) < 1e-9
    tv, av, q = stability_quotient([[0, 10] * 5])              # wildly oscillating
    assert abs(tv - 25) < 1e-9 and abs(av - 100) < 1e-9 and abs(q - 0.25) < 1e-9
    assert math.isnan(stability_quotient([[3, 3, 3, 3]])[2])   # no variation at all
    assert math.isnan(stability_quotient([[1]])[2])            # no adjacent pairs
    print("  ramp -> 8.25 (stable), oscillation -> 0.25 (unstable), constant -> nan. PASS\n")


def check_auc_and_rules():
    print("=" * 60)
    print("3. AUC AND TRIGGER-RULE ARITHMETIC")
    print("=" * 60)
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert auc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5
    assert math.isnan(auc([1, 2], [1, 1]))
    rows = [{"n_checks": c, "threat_gain": 0, "max_gain": 0, "in_check": 0,
             "changed": ch, "delta": 100.0 if ch else 0.0}
            for c, ch in [(0, 0), (0, 0), (0, 0), (0, 1), (2, 1), (2, 1), (2, 0), (3, 1)]]
    rule = {d["rule"]: d for d in threshold_rows(rows)}["n_checks >= 2"]
    # flagged: 4 of 8; of those 3 changed; 4 changed overall
    assert rule["flagged"] == 0.5 and rule["precision"] == 0.75 and rule["recall"] == 0.75
    print("  AUC 1 / 0 / 0.5 on known cases; n_checks>=2: flagged 50%, precision 75%, recall 75%. PASS\n")


def check_labels():
    print("=" * 60)
    print("4. GROUND-TRUTH LABELS")
    print("=" * 60)
    # Back-rank mate in one: quiescence cannot see it (not a capture), one more ply can.
    mate = label_position("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    assert mate["changed"] == 1 and mate["n_checks"] >= 1 and abs(mate["v1"]) > 90000, mate
    quiet = label_position(chess.STARTING_FEN)
    assert quiet["changed"] == 0, quiet
    print(f"  back-rank mate: v0={mate['v0']} -> v1={mate['v1']} (changed). "
          f"start position: unchanged. PASS\n")


def check_sampling():
    print("=" * 60)
    print("5. FRONTIER SAMPLING")
    print("=" * 60)
    b = chess.Board()
    a = sample_frontier_fens(b, 2, 10, random.Random(1))
    c = sample_frontier_fens(b, 2, 10, random.Random(1))
    d = sample_frontier_fens(b, 2, 10, random.Random(2))
    assert a == c and a != d and len(a) == 10
    for fen in a:
        pos = chess.Board(fen)
        assert len(pos.move_stack) == 0 and pos.fullmove_number == 2   # 2 plies from the start
    assert b.fen() == chess.STARTING_FEN, "root board must be untouched"
    print("  10 frontier positions, deterministic per seed, all 2 plies from the root. PASS\n")


def check_model_pipeline():
    print("=" * 60)
    print("6. LOG -> MODEL -> CONTROLLER INPUT")
    print("=" * 60)
    rng = random.Random(0)
    fens = []
    for fen in ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
                "4r1k1/5ppp/8/8/8/8/8/4R2K w - - 0 1",
                "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"]:
        fens += sample_frontier_fens(chess.Board(fen), 2, 20, rng)
    rows = [label_position(f) for f in fens]
    for r in rows:
        r["source"], r["root_id"] = "t", "t"
    model = bucket_model(rows)
    assert len(model["p"]) == len(model["edges"]) == len(model["shift"])
    assert all(0 <= p <= 1 for p in model["p"]) and sum(model["n"]) == len(rows)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        with open(path, "w") as f:
            json.dump({k: v for k, v in model.items() if k not in ("ci_lo", "ci_hi")}, f)
        tm = TriggerModel.load(path)
    assert tm.feature == "n_checks" and 0 <= tm.p_change(2) <= 1
    print(f"  {len(rows)} labelled nodes -> model with {len(model['edges'])} buckets, "
          f"saved and reloaded as a TriggerModel. PASS\n")


if __name__ == "__main__":
    check_features()
    check_stability()
    check_auc_and_rules()
    check_labels()
    check_sampling()
    check_model_pipeline()
    print("Phase 3 verified.")
