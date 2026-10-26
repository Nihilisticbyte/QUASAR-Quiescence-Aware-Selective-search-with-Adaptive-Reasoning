"""
Paired comparison of engine versions (Phase 5).

All versions are run on the SAME positions at the SAME node budgets, so the
right question is per-position: "on this position, did version B lose fewer
centipawns than the baseline?" Comparing two separate averages throws that
pairing away and needs far more positions to see the same effect.

For every (source, budget, engine) against the baseline we report
  mean_diff     mean of (engine loss - baseline loss) over positions; NEGATIVE
                means the engine is better. Bootstrap 95% CI.
  better/worse  number of positions where the engine lost fewer / more cp
  sign_p        exact two-sided sign test on better vs worse (ties ignored)
  solve +/-     WAC only: positions the engine solved that baseline missed (+)
                and the reverse (-), with an exact McNemar p-value
  nodes         mean nodes used. Budgets are equal by construction; this
                column shows whether the engine really spent them.
  time x        mean wall-clock time relative to baseline (overhead of the
                extension logic that node counts do not show)
  ext           mean extensions per search
"""

import math
import statistics
from typing import Dict, List

from .harness import Result, bootstrap_mean_ci


def sign_test_p(better: int, worse: int) -> float:
    """Exact two-sided sign test (binomial, p=0.5) on the non-tied pairs."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_comparison(results: List[Result], base: str = "baseline") -> List[dict]:
    by_key: Dict[tuple, Dict[str, Result]] = {}
    for r in results:
        by_key.setdefault((r.source, r.budget, r.pos_id), {})[r.engine] = r

    groups: Dict[tuple, List[tuple]] = {}
    for (source, budget, _pid), engs in by_key.items():
        if base not in engs:
            continue
        for name, r in engs.items():
            if name != base:
                groups.setdefault((source, budget, name), []).append((engs[base], r))

    rows = []
    for (source, budget, name), pairs in sorted(groups.items()):
        diffs = [r.cp_loss - b.cp_loss for b, r in pairs]
        better = sum(d < 0 for d in diffs)
        worse = sum(d > 0 for d in diffs)
        lo, hi = bootstrap_mean_ci(diffs)
        book = [(b, r) for b, r in pairs if b.book_hit is not None]
        plus = sum((not b.book_hit) and r.book_hit for b, r in book)
        minus = sum(b.book_hit and (not r.book_hit) for b, r in book)
        t_base = statistics.mean(b.time_s for b, _ in pairs)
        rows.append({
            "source": source, "budget": budget, "engine": name, "n": len(pairs),
            "mean_diff": statistics.mean(diffs), "diff_lo": lo, "diff_hi": hi,
            "better": better, "worse": worse, "sign_p": sign_test_p(better, worse),
            "solve_plus": plus if book else None, "solve_minus": minus if book else None,
            "mcnemar_p": sign_test_p(plus, minus) if book else math.nan,
            "mean_nodes": statistics.mean(r.nodes for _, r in pairs),
            "time_ratio": (statistics.mean(r.time_s for _, r in pairs) / t_base) if t_base else math.nan,
            "mean_ext": statistics.mean(r.extensions for _, r in pairs),
        })
    return rows


def format_paired(rows: List[dict], base: str = "baseline") -> str:
    head = (f"{'source':<7}{'budget':>7} {'engine':<16}{'n':>5}"
            f"{'mean diff cp [95% CI]':>29}{'better':>8}{'worse':>7}{'sign p':>8}"
            f"{'solve +/-':>11}{'McN p':>7}{'nodes':>8}{'time x':>8}{'ext':>7}")
    out = [f"difference = engine loss - {base} loss (negative = engine better)", head,
           "-" * len(head)]
    for r in rows:
        ci = f"{r['mean_diff']:7.1f} [{r['diff_lo']:6.1f},{r['diff_hi']:6.1f}]"
        solve = "     -" if r["solve_plus"] is None else f"{r['solve_plus']:>5}/{r['solve_minus']:<4}"
        mcn = "    -" if r["solve_plus"] is None else f"{r['mcnemar_p']:7.3f}"
        out.append(f"{r['source']:<7}{r['budget']:>7} {r['engine']:<16}{r['n']:>5}{ci:>29}"
                   f"{r['better']:>8}{r['worse']:>7}{r['sign_p']:>8.3f}{solve:>11}{mcn:>7}"
                   f"{r['mean_nodes']:>8.0f}{r['time_ratio']:>8.2f}{r['mean_ext']:>7.1f}")
    return "\n".join(out)
