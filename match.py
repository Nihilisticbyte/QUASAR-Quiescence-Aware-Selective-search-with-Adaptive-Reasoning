"""
Phase 5 (lowest priority metric): engine A vs engine B at equal node budgets.

Usage (from the project folder):
    python match.py --a inst_c2 --b baseline --budget 2000 --openings 10
    python match.py --a check   --b baseline --budget 5000 --openings 20 --out results/match_check.csv

Each opening is played twice with colours swapped. Needs dozens of games
before the Elo figure means anything; read the confidence interval, and treat
this as presentation material, not the main evidence (see engine/match.py).
"""

import argparse
import csv
import os
import time

from engine.harness import ENGINES
from engine.match import openings_from_pgn, run_match, match_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, choices=list(ENGINES))
    ap.add_argument("--b", default="baseline", choices=list(ENGINES))
    ap.add_argument("--budget", type=int, default=2000, help="nodes per move, both engines")
    ap.add_argument("--openings", type=int, default=10, help="games = 2 x openings")
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--pgn", default=os.path.join("data", "real_games.pgn"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.pgn):
        raise SystemExit(f"PGN not found: {args.pgn}")
    ops = openings_from_pgn(args.pgn, args.openings, args.opening_plies, args.seed)
    print(f"{args.a} vs {args.b}: {len(ops)} openings x 2 colours, {args.budget} nodes/move")
    t0 = time.time()

    def progress(i, n, games):
        print(f"  opening {i}/{n} done ({len(games)} games, {(time.time()-t0)/60:.1f} min)")

    games = run_match(ENGINES[args.a], ENGINES[args.b], ops, args.budget, args.max_plies, progress)
    s = match_summary(games)
    print(f"\n{args.a}: +{s['wins']} ={s['draws']} -{s['losses']}  "
          f"score {100*s['score']:.1f}% [{100*s['score_lo']:.1f}, {100*s['score_hi']:.1f}]  "
          f"Elo {s['elo']:+.0f} [{s['elo_lo']:+.0f}, {s['elo_hi']:+.0f}]  over {s['games']} games")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["opening", "a_is_white", "score_a", "plies", "reason"])
            for g in games:
                w.writerow([g.opening, g.a_is_white, g.score_a, g.plies, g.reason])
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
