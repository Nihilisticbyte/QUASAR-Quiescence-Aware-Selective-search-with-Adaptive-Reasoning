"""
Phase 2, step 2: Stockfish oracle checks.

Run: python test_oracle.py
Needs Stockfish: set STOCKFISH_PATH (see engine/oracle.py) or have it on PATH.
If Stockfish can't be found, the test says so and exits without failing.

1. Oracle finds a known mate-in-1.
2. The oracle's own best move has 0 centipawn loss.
3. A clear blunder has a large loss.
4. Repeat queries are served from the cache (no extra Stockfish search).
5. Our own engine (node-budgeted) is scored against the oracle end to end.
"""

import sys
import chess
from engine.oracle import StockfishOracle, find_stockfish
from engine.search import iterative_deepening

MATE_IN_1 = "4r1k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"          # Rxe8#
TACTICAL = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"  # Qxf7#


def main():
    try:
        find_stockfish()
    except FileNotFoundError as e:
        print(f"SKIPPED: {e}")
        return

    with StockfishOracle(depth=10) as oracle:
        print("=" * 60)
        print("1. ORACLE FINDS MATE IN 1")
        print("=" * 60)
        board = chess.Board(MATE_IN_1)
        res = oracle.analyse(board)
        print(f"  best={board.san(res.best_move)}  mate_in={res.mate_in}  cp={res.score_cp}")
        assert res.best_move == chess.Move.from_uci("e1e8")
        assert res.mate_in == 1
        print("  PASS\n")

        print("=" * 60)
        print("2. ORACLE'S OWN MOVE HAS ZERO LOSS")
        print("=" * 60)
        assert oracle.move_loss(board, res.best_move) == 0
        print("  PASS\n")

        print("=" * 60)
        print("3. A BLUNDER HAS A LARGE LOSS")
        print("=" * 60)
        tact = chess.Board(TACTICAL)
        blunder = tact.parse_san("Qh4")      # misses Qxf7#
        loss = oracle.move_loss(tact, blunder)
        print(f"  Qh4 instead of Qxf7#: loss = {loss}cp")
        assert loss > 300
        print("  PASS\n")

        print("=" * 60)
        print("4. CACHE: REPEAT QUERIES DON'T RE-SEARCH")
        print("=" * 60)
        before = oracle.calls
        oracle.analyse(board)
        oracle.analyse(tact)
        oracle.move_loss(tact, blunder)
        print(f"  Stockfish searches before={before}, after={oracle.calls}")
        assert oracle.calls == before
        print("  PASS\n")

        print("=" * 60)
        print("5. OUR ENGINE vs THE ORACLE (end to end)")
        print("=" * 60)
        for name, fen in [("mate-in-1", MATE_IN_1), ("tactical", TACTICAL)]:
            b = chess.Board(fen)
            move, score, stats, _ = iterative_deepening(b, 64, node_budget=8000)
            loss = oracle.move_loss(b, move)
            best = oracle.analyse(b).best_move
            print(f"  {name:<10} engine={b.san(move):<6} oracle={b.san(best):<6} "
                  f"loss={loss}cp  (depth {stats.completed_depth}, {stats.nodes} nodes)")
            assert loss >= 0
        print("  PASS\n")

    print("Stockfish oracle verified.")


if __name__ == "__main__":
    main()
