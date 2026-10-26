"""
Phase 5 checks: paired statistics and engine-vs-engine matches.

Run: python test_phase5.py        (no Stockfish needed)
"""

import os
import tempfile

import chess
from engine.compare import paired_comparison, sign_test_p, format_paired
from engine.harness import Result, ENGINES, summarize
from engine.report import markdown_summary, markdown_paired
from engine.match import (elo_from_score, match_summary, openings_from_pgn, play_game,
                          run_match, GameResult)
from test_positions import make_random_pgn


def mk(engine, pos, loss, book=None, source="wac", budget=1000, t=1.0, ext=0):
    return Result(engine, pos, source, budget, "e2e4", 500, 3, t, "e2e4", False, book, loss, ext)


def check_sign_test():
    print("=" * 60)
    print("1. EXACT SIGN TEST")
    print("=" * 60)
    assert abs(sign_test_p(9, 1) - 22 / 1024) < 1e-12        # 2 * (1 + 10) / 2^10
    assert abs(sign_test_p(10, 0) - 2 / 1024) < 1e-12
    assert sign_test_p(5, 5) == 1.0 and sign_test_p(0, 0) == 1.0
    print("  9-1 -> p = 0.0215, 10-0 -> p = 0.0020, 5-5 and 0-0 -> p = 1. PASS\n")


def check_paired():
    print("=" * 60)
    print("2. PAIRED COMPARISON ON HAND-BUILT RESULTS")
    print("=" * 60)
    # four WAC positions. diffs (engine - baseline): -100, -50, 0, +30  -> mean -30
    base = [mk("baseline", "A", 100, False), mk("baseline", "B", 50, False),
            mk("baseline", "C", 0, True), mk("baseline", "D", 0, True)]
    new = [mk("x", "A", 0, True, ext=4), mk("x", "B", 0, False, ext=2),
           mk("x", "C", 0, True), mk("x", "D", 30, False)]
    rows = paired_comparison(base + new)
    assert len(rows) == 1
    r = rows[0]
    assert r["engine"] == "x" and r["n"] == 4
    assert abs(r["mean_diff"] - (-30.0)) < 1e-9
    assert (r["better"], r["worse"]) == (2, 1)
    assert (r["solve_plus"], r["solve_minus"]) == (1, 1)      # A gained, D lost
    assert abs(r["mean_ext"] - 1.5) < 1e-9
    # a position missing for one engine must be dropped, not mismatched
    rows2 = paired_comparison(base + new[:3])
    assert rows2[0]["n"] == 3
    # a baseline-only file gives nothing to compare
    assert paired_comparison(base) == []
    assert "x" in format_paired(rows)
    print("  mean diff -30, 2 better / 1 worse, solve +1/-1, unmatched positions dropped. PASS\n")


def check_elo():
    print("=" * 60)
    print("3. ELO CONVERSION AND MATCH SUMMARY")
    print("=" * 60)
    assert abs(elo_from_score(0.5)) < 1e-9
    assert abs(elo_from_score(0.75) - 190.85) < 0.1
    assert abs(elo_from_score(0.25) + elo_from_score(0.75)) < 1e-9
    games = [GameResult("", True, s, 10, "") for s in (1, 1, 0.5, 0, 1, 0.5, 0, 1)]
    s = match_summary(games)
    assert (s["wins"], s["draws"], s["losses"]) == (4, 2, 2) and abs(s["score"] - 0.625) < 1e-9
    assert s["score_lo"] < s["score"] < s["score_hi"] and s["elo_lo"] < s["elo"] < s["elo_hi"]
    print("  0.75 -> +190.8 Elo, symmetric, summary W/D/L and interval ordered. PASS\n")


def check_games():
    print("=" * 60)
    print("4. GAMES: OPENINGS, LEGALITY, COLOUR SWAP")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.pgn")
        make_random_pgn(path, games=8, plies=60, seed=3)
        a = openings_from_pgn(path, 4, plies=8, seed=1)
        b = openings_from_pgn(path, 4, plies=8, seed=1)
        assert a == b and len(a) == 4 and len(set(a)) == 4
        for fen in a:
            assert chess.Board(fen).fullmove_number == 5        # 8 plies from the start
    eng = ENGINES["baseline"]
    games = run_match(eng, eng, a[:1], budget=300, max_plies=30)
    assert len(games) == 2 and {g.a_is_white for g in games} == {True, False}
    # identical deterministic engines: the swapped-colour game is the same game,
    # so the two scores for "A" must add up to exactly 1.
    assert abs(sum(g.score_a for g in games) - 1.0) < 1e-9
    assert all(g.plies <= 30 for g in games)
    # an engine returning an illegal move is caught, not played
    try:
        play_game(lambda b, n: (chess.Move.from_uci("a1a8"), None), eng, chess.STARTING_FEN, 100)
        raise AssertionError("illegal move was accepted")
    except RuntimeError:
        pass
    print("  4 distinct deterministic openings, colour-swapped pair adds to 1, illegal moves caught. PASS\n")


def check_report_tables():
    print("=" * 60)
    print("5. MARKDOWN TABLES CARRY THE SAME NUMBERS")
    print("=" * 60)
    base = [mk("baseline", "A", 100, False), mk("baseline", "B", 50, False)]
    new = [mk("x", "A", 0, True, ext=4), mk("x", "B", 20, False, ext=2)]
    summ = markdown_summary(summarize(base + new))
    pair = markdown_paired(paired_comparison(base + new))
    lines = summ.splitlines()
    assert len(lines) == 2 + 2                       # header, rule, one row per engine
    assert "| baseline | wac | 1000 | 2 |" in summ and "| x | wac | 1000 | 2 |" in summ
    assert "-65.0" in pair and "| 2 / 0 |" in pair   # mean diff (-100 + -30)/2 and better/worse
    print("  tables have one row per engine/group; mean difference -65.0 and 2/0 better/worse appear. PASS\n")


if __name__ == "__main__":
    check_sign_test()
    check_paired()
    check_elo()
    check_games()
    check_report_tables()
    print("Phase 5 tooling verified.")
