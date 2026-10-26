"""
Run the baseline experiment: engine version(s) x test positions x node budgets,
scored against Stockfish. Results are saved after every single run and the
script can be re-run safely: finished runs are skipped.

Usage (from the project folder):
    python run_baseline.py                          # 100 WAC + 100 real (held-out TEST split), budgets 1k/5k/20k
    python run_baseline.py --engines baseline check inst_c2   # compare several engine versions
    python run_baseline.py --split train --engines baseline costw_l20 costw_l40 costw_l80 --out results/train.csv
    python run_baseline.py --wac-n 30 --real-n 30 --budgets 1000 5000   # quick trial
    python run_baseline.py --summary-only           # just re-print the table from the CSV

Stockfish: pass --stockfish "C:\\path\\to\\stockfish.exe" or set STOCKFISH_PATH.
IMPORTANT: use the SAME Stockfish version and --oracle-depth for every run
you want to compare, and a fresh --out file if you change either one.
"""

import argparse
import os

from engine.harness import (ENGINES, run_experiment, load_results, summarize,
                            format_summary, write_summary_csv)
from engine.oracle import StockfishOracle
from engine.positions import select_positions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wac", default=os.path.join("data", "wac.epd"))
    ap.add_argument("--pgn", default=os.path.join("data", "real_games.pgn"))
    ap.add_argument("--wac-n", type=int, default=100, help="WAC positions to use (0 = none)")
    ap.add_argument("--real-n", type=int, default=100, help="real-game positions to use (0 = none)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000, 20000])
    ap.add_argument("--engines", nargs="+", default=["baseline"], choices=list(ENGINES))
    ap.add_argument("--oracle-depth", type=int, default=12)
    ap.add_argument("--stockfish", default=None)
    ap.add_argument("--split", choices=["test", "train", "all"], default="test",
                    help="test = held-out positions (report on these); train = tune on these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("results", "baseline.csv"))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    summary_path = args.out.replace(".csv", "_summary.csv")

    if not args.summary_only:
        if args.real_n and not os.path.exists(args.pgn):
            raise SystemExit(f"Real-game PGN not found: {args.pgn}\n"
                             "Build data/real_games.pgn first, or run with --real-n 0.")
        positions = select_positions(args.wac, args.pgn, args.wac_n, args.real_n,
                                     split=args.split, seed=args.seed)
        print(f"split={args.split}: {len(positions)} positions: "
              f"{sum(p.source != 'real' for p in positions)} book, "
              f"{sum(p.source == 'real' for p in positions)} real-game; "
              f"budgets {args.budgets}; seed {args.seed}")
        engines = {n: ENGINES[n] for n in args.engines}
        with StockfishOracle(args.stockfish, depth=args.oracle_depth) as oracle:
            run_experiment(engines, positions, args.budgets, oracle, out_csv=args.out)

    rows = summarize(load_results(args.out))
    write_summary_csv(rows, summary_path)
    print()
    print(format_summary(rows))
    print(f"\nraw results: {args.out}\nsummary:     {summary_path}")


if __name__ == "__main__":
    main()
