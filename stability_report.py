"""
Phase 3: Clune-style stability report (total variance vs adjacent variance).

Samples positions along the main lines of real games and reports, for each
feature, how smoothly it changes from one position to the next
(see engine/stability.py for what the numbers mean and how they may be used).

Usage:  python stability_report.py [--pgn data/real_games.pgn] [--games 60] [--stride 1]
"""

import argparse
import os

from engine.stability import feature_paths_from_pgn, stability_report, format_stability


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=os.path.join("data", "real_games.pgn"))
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1, help="plies between sampled positions")
    args = ap.parse_args()
    if not os.path.exists(args.pgn):
        raise SystemExit(f"PGN not found: {args.pgn}")
    paths = feature_paths_from_pgn(args.pgn, max_games=args.games, stride=args.stride)
    print(format_stability(stability_report(paths)))
    print("\nstability ~ 1: feature jumps around between neighbouring positions;"
          "\nlarge: it changes only gradually. (Offline diagnostic, see engine/stability.py.)")


if __name__ == "__main__":
    main()
