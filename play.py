"""
Play a game against the engine from the terminal.

This is a utility for development/demo purposes, NOT part of the research
evaluation (that's test suites + the harness in later phases). It exists so
you can:
  1. Feel the horizon effect yourself (try --depth 2 and play tactically —
     watch it blunder right before a capture it can't see past).
  2. Have a live demo for your presentation.

Usage:
    python3 play.py                  # you play White, engine is depth 3
    python3 play.py --color black    # you play Black
    python3 play.py --depth 2        # shallower = dumber = easier to see
                                      # the horizon effect in action
    python3 play.py --depth 5        # slower, stronger

Moves are entered in SAN (Standard Algebraic Notation), e.g.:
    e4, Nf3, exd5, O-O, Qxh7#
UCI notation also works if you prefer it, e.g.: e2e4, g1f3
"""

import argparse
import time
import chess
from engine.search import search_root


def print_board(board: chess.Board):
    print()
    print(board)
    print()


def get_human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move: ").strip()
        if raw.lower() in ("quit", "exit", "resign"):
            raise SystemExit("Game aborted.")
        if raw.lower() == "moves":
            print(", ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue
        try:
            # Try SAN first ("Nf3"), fall back to UCI ("g1f3").
            try:
                move = board.parse_san(raw)
            except ValueError:
                move = chess.Move.from_uci(raw)
            if move in board.legal_moves:
                return move
            print("Illegal move. Type 'moves' to list legal moves.")
        except ValueError:
            print("Couldn't parse that. Use SAN (e.g. Nf3) or UCI (e.g. g1f3). "
                  "Type 'moves' to list legal moves.")


def get_engine_move(board: chess.Board, depth: int) -> chess.Move:
    start = time.time()
    move, score, stats = search_root(board, depth)
    elapsed = time.time() - start
    print(f"Engine plays {board.san(move)}  "
          f"(score={score:+d}cp, nodes={stats.nodes:,}, {elapsed:.2f}s)")
    return move


def main():
    parser = argparse.ArgumentParser(description="Play against the Phase 0 engine.")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                         help="Which side you play (default: white)")
    parser.add_argument("--depth", type=int, default=3,
                         help="Engine search depth in plies (default: 3). "
                              "Try 2 to easily see the horizon effect.")
    args = parser.parse_args()

    human_is_white = args.color == "white"
    board = chess.Board()

    print("=" * 60)
    print(f"You are playing {'White' if human_is_white else 'Black'}.")
    print(f"Engine depth: {args.depth} plies "
          f"(plain alpha-beta, no TT, no quiescence -- Phase 0 baseline)")
    print("Enter moves in SAN (e.g. Nf3) or UCI (e.g. g1f3). "
          "Type 'moves' to list legal moves, 'quit' to stop.")
    print("=" * 60)

    print_board(board)

    while not board.is_game_over():
        human_to_move = (board.turn == chess.WHITE) == human_is_white

        if human_to_move:
            move = get_human_move(board)
        else:
            move = get_engine_move(board, args.depth)

        board.push(move)
        print_board(board)

        if board.is_check() and not board.is_checkmate():
            print("Check!")

    print("=" * 60)
    print(f"Game over: {board.result()}")
    outcome = board.outcome()
    if outcome:
        print(f"Reason: {outcome.termination.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
