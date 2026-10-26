"""
Phase 6 helper: turn a results CSV into Markdown tables for the report.

    python make_report_tables.py --results results/exp1.csv

Writes results/exp1_tables.md (summary table + paired comparison with the
baseline) and prints it. Paste the tables into REPORT.md. The numbers are the
same ones printed by run_baseline.py and compare_engines.py.
"""

import argparse

from engine.compare import paired_comparison
from engine.harness import load_results, summarize
from engine.report import markdown_summary, markdown_paired


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default="baseline")
    args = ap.parse_args()

    results = load_results(args.results)
    text = "## Per-engine results\n\n" + markdown_summary(summarize(results)) + "\n"
    paired = paired_comparison(results, args.base)
    if paired:
        text += "\n## Paired comparison with " + args.base + "\n\n" + markdown_paired(paired, args.base) + "\n"
    out = args.results.replace(".csv", "_tables.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
