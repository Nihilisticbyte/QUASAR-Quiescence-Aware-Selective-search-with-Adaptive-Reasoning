"""
Phase 2, step 3: test-position loading checks.

Run: python test_positions.py

1. WAC loads: 300 positions, unique ids, every best move legal on its board.
2. A malformed EPD fails loudly (doesn't silently skip positions).
3. The PGN sampler respects ply range, per-game cap, and is deterministic.
4. Smoke run: our budgeted engine on the first WAC positions (informational).
"""

import os
import random
import tempfile
import chess
import chess.pgn
from engine.positions import load_epd, sample_pgn_positions
from engine.search import iterative_deepening

WAC = os.path.join("data", "wac.epd")


def make_random_pgn(path, games=6, plies=70, seed=1):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for g in range(games):
            board = chess.Board()
            game = chess.pgn.Game()
            node = game
            for _ in range(plies):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                node = node.add_variation(move)
            print(game, file=f, end="\n\n")


def check_wac():
    print("=" * 60)
    print("1. WAC LOADS CLEANLY")
    print("=" * 60)
    ps = load_epd(WAC)
    assert len(ps) == 300, len(ps)
    assert len({p.id for p in ps}) == 300, "duplicate ids"
    for p in ps:
        b = p.board()
        assert p.best_moves, f"{p.id}: no best move"
        assert all(m in b.legal_moves for m in p.best_moves), f"{p.id}: illegal bm"
    multi = sum(len(p.best_moves) > 1 for p in ps)
    print(f"  300 positions, ids unique, all best moves legal "
          f"({multi} have more than one accepted move). PASS\n")


def check_bad_epd():
    print("=" * 60)
    print("2. MALFORMED EPD FAILS LOUDLY")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.epd")
        with open(path, "w") as f:
            f.write('4k3/8/8/8/8/8/8/4K3 w - - bm Qh5; id "BAD.1";\n')  # no queen
        try:
            load_epd(path)
        except ValueError as e:
            print(f"  raised ValueError as expected: {e}")
        else:
            raise AssertionError("illegal best move was accepted")
    print("  PASS\n")


def check_pgn_sampler():
    print("=" * 60)
    print("3. PGN SAMPLER")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.pgn")
        make_random_pgn(path)
        a = sample_pgn_positions(path, 10, min_ply=12, max_ply=60, per_game=2, seed=7)
        b = sample_pgn_positions(path, 10, min_ply=12, max_ply=60, per_game=2, seed=7)
        c = sample_pgn_positions(path, 10, min_ply=12, max_ply=60, per_game=2, seed=8)
        assert [p.fen for p in a] == [p.fen for p in b], "not deterministic"
        assert [p.fen for p in a] != [p.fen for p in c], "seed has no effect"
        assert len(a) == 10 and len({p.fen for p in a}) == 10
        per_game = {}
        for p in a:
            _, g, ply = p.id.split(".")
            per_game[g] = per_game.get(g, 0) + 1
            assert 12 <= int(ply) <= 60, f"{p.id}: ply out of range"
            bd = p.board()
            assert not bd.is_game_over() and bd.legal_moves.count() >= 2
            assert p.best_moves == [] and p.source == "real"
        assert max(per_game.values()) <= 2, "per-game cap violated"
        print(f"  10 positions from {len(per_game)} games, ply in range, "
              f"max {max(per_game.values())}/game, deterministic. PASS")

        # Regression: file order must not bias the sample. 20 games in the
        # file, ask for 6 positions at 1/game: the old sampler always
        # returned games 1-6 (the start of the file).
        path2 = os.path.join(d, "many.pgn")
        make_random_pgn(path2, games=20, seed=5)
        picked = sample_pgn_positions(path2, 6, min_ply=12, max_ply=60,
                                      per_game=1, seed=3)
        games_used = sorted(int(p.id.split(".")[1]) for p in picked)
        print(f"  6 positions drawn from games {games_used} of 20")
        assert max(games_used) > 6, "sample is biased towards the start of the file"
        print("  no file-order bias. PASS\n")


def smoke_run(n=5, budget=3000):
    print("=" * 60)
    print(f"4. SMOKE RUN: first {n} WAC positions, {budget}-node budget")
    print("=" * 60)
    hits = 0
    for p in load_epd(WAC, limit=n):
        b = p.board()
        move, score, stats, _ = iterative_deepening(b, 64, node_budget=budget)
        ok = move in p.best_moves
        hits += ok
        print(f"  {p.id}: engine={b.san(move):<7} book={b.san(p.best_moves[0]):<7} "
              f"{'HIT' if ok else 'miss'}  (depth {stats.completed_depth})")
    print(f"  {hits}/{n} match the book move (informational only)\n")


if __name__ == "__main__":
    check_wac()
    check_bad_epd()
    check_pgn_sampler()
    smoke_run()
    print("Position loading verified.")
