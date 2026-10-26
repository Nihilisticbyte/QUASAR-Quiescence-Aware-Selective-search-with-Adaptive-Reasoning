"""
Markdown tables for the write-up (Phase 6), built from results CSVs.

Nothing here computes new statistics: it only formats the numbers from
harness.summarize() and compare.paired_comparison(), so the report's tables
are exactly the numbers the experiments produced.
"""

import math
from typing import List


def _pct(x: float) -> str:
    return "-" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def markdown_summary(rows: List[dict]) -> str:
    out = ["| engine | source | budget | n | failed | depth | ext/search | book solve [95% CI] "
           "| = Stockfish move | mean cp loss [95% CI] | median | big misses |",
           "|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|"]
    for r in rows:
        book = "-" if r["book_n"] == 0 else (
            f"{_pct(r['book_solve'])} [{100*r['book_solve_lo']:.0f}, {100*r['book_solve_hi']:.0f}]")
        out.append(f"| {r['engine']} | {r['source']} | {r['budget']} | {r['n']} | {r['failed']} "
                   f"| {r['mean_depth']:.1f} | {r['mean_ext']:.1f} | {book} | {_pct(r['oracle_match'])} "
                   f"| {r['mean_loss']:.1f} [{r['mean_loss_lo']:.1f}, {r['mean_loss_hi']:.1f}] "
                   f"| {r['median_loss']:.0f} | {_pct(r['big_miss'])} |")
    return "\n".join(out)


def markdown_paired(rows: List[dict], base: str = "baseline") -> str:
    out = [f"Difference = engine loss - {base} loss, in centipawns (negative = engine better).", "",
           "| source | budget | engine | n | mean diff [95% CI] | better / worse | sign p "
           "| solve +/- | McNemar p | time x | ext/search |",
           "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        solve = "-" if r["solve_plus"] is None else f"{r['solve_plus']}/{r['solve_minus']}"
        mcn = "-" if r["solve_plus"] is None else f"{r['mcnemar_p']:.3f}"
        out.append(f"| {r['source']} | {r['budget']} | {r['engine']} | {r['n']} "
                   f"| {r['mean_diff']:.1f} [{r['diff_lo']:.1f}, {r['diff_hi']:.1f}] "
                   f"| {r['better']} / {r['worse']} | {r['sign_p']:.3f} | {solve} | {mcn} "
                   f"| {r['time_ratio']:.2f} | {r['mean_ext']:.1f} |")
    return "\n".join(out)
