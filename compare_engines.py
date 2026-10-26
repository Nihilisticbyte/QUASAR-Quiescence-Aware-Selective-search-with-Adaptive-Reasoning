"""
Phase 5: paired comparison of engine versions from a results CSV.

Run run_baseline.py with several engines first, e.g.
    python run_baseline.py --engines baseline check inst_c2 --out results/exp1.csv
then
    python compare_engines.py --results results/exp1.csv

Every engine in the file is compared with --base (default: baseline) on the
positions they share, per source (wac / real) and node budget.
See engine/compare.py for what each column means.
"""

import argparse
import csv

from engine.compare import paired_comparison, format_paired
from engine.harness import load_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default="baseline")
    args = ap.parse_args()

    rows = paired_comparison(load_results(args.results), args.base)
    if not rows:
        raise SystemExit(f"No comparisons possible: need '{args.base}' plus at least one other "
                         "engine on the same positions in the file.")
    print(format_paired(rows, args.base))
    out = args.results.replace(".csv", "_paired.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
