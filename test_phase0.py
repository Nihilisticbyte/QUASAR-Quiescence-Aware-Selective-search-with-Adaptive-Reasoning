"""
Phase 0 sanity checks.

Run: python3 test_phase0.py

1. perft() matches known-correct values -> move generation trustworthy.
2. The alpha-beta engine can self-play a full legal game without crashing
   or producing illegal moves.
3. Node counts / timing at a few depths -- the baseline reference point
   for later phases (TT, quiescence, extensions should all change these
   numbers, so you need today's numbers to know if they helped).
"""

import time
import chess
from engine.search import search_root

KNOWN_PERFT = {1: 20, 2: 400, 3: 8902, 4: 197281}


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    count = 0
    for move in board.legal_moves:
        board.push(move)
        count += perft(board, depth - 1)
        board.pop()
    return count


def run_perft_check():
    print("=" * 60)
    print("1. PERFT CHECK (move generation correctness)")
    print("=" * 60)
    board = chess.Board()
    for depth, expected in KNOWN_PERFT.items():
        start = time.time()
        actual = perft(board, depth)
        elapsed = time.time() - start
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  depth {depth}: expected={expected:>7} actual={actual:>7} "
              f"[{status}]  ({elapsed:.2f}s)")
        assert actual == expected, f"Perft mismatch at depth {depth}!"
    print("  -> move generation is correct.\n")


def run_self_play(max_moves: int = 40, depth: int = 3):
    print("=" * 60)
    print(f"2. SELF-PLAY CHECK (alpha-beta vs itself, depth={depth})")
    print("=" * 60)
    board = chess.Board()
    move_count = 0
    total_nodes = 0
    start = time.time()

    while not board.is_game_over() and move_count < max_moves:
        move, score, stats = search_root(board, depth)
        assert move in board.legal_moves, \
            f"ILLEGAL MOVE PRODUCED: {move} in position {board.fen()}"
        board.push(move)
        total_nodes += stats.nodes
        move_count += 1

    elapsed = time.time() - start
    print(f"  played {move_count} plies, no illegal moves")
    print(f"  game over: {board.is_game_over()}  "
          f"result: {board.result() if board.is_game_over() else 'in progress'}")
    print(f"  total nodes searched: {total_nodes:,}  "
          f"in {elapsed:.1f}s  ({total_nodes/max(elapsed,0.001):,.0f} nodes/sec)")
    print(f"  final position:\n{board}\n")


def run_node_count_baseline():
    print("=" * 60)
    print("3. NODE COUNT BASELINE (starting position, several depths)")
    print("=" * 60)
    print("  Record these numbers -- Phase 1+ (TT, quiescence, extensions)")
    print("  should be compared against this baseline.\n")
    board = chess.Board()
    for depth in [1, 2, 3, 4]:
        start = time.time()
        move, score, stats = search_root(board, depth)
        elapsed = time.time() - start
        print(f"  depth {depth}: best_move={board.san(move):>6}  "
              f"score={score:>6}  nodes={stats.nodes:>8,}  "
              f"leaf_nodes={stats.leaf_nodes:>8,}  time={elapsed:.2f}s")


if __name__ == "__main__":
    run_perft_check()
    run_self_play(max_moves=40, depth=3)
    run_node_count_baseline()
    print("\nPhase 0 complete: engine is legal and searching correctly.")
