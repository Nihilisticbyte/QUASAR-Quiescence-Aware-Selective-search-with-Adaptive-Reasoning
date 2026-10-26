"""
Phase 3 analysis: how well do the instability features predict "the value at
this frontier node changes with one more ply"? (metric 3, trigger accuracy)

Everything here works on the CSV written by log_instability.py.
  - auc():                 threshold-free ranking quality of one feature
  - threshold_rows():      precision / recall / share flagged for rules like
                           "max_gain >= 200" or "in_check"
  - bucket_model():        P(changed | n_checks bucket) and the typical size
                           of the change, the table the cost-weighted
                           controller uses (TriggerModel)

Reading precision/recall here: "positive" = the node's value really changed.
A flagged node that did not change is a WASTED extension (costs nodes, buys
nothing); an unflagged node that did change is a MISSED one. 'flagged' is the
share of ALL frontier nodes the rule would extend, a direct proxy for the
extra search it would trigger.
"""

import csv
import math
from typing import Dict, List

from .harness import wilson

FEATURES_NUMERIC = ["in_check", "n_captures", "n_good_captures", "max_gain",
                    "threat_gain", "n_checks", "static_stm", "n_legal"]
DEFAULT_EDGES = [0, 1, 2, 3, 5]      # buckets of n_checks


def load_log(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: float(r[k]) for k in FEATURES_NUMERIC + ["v0", "v1", "delta", "changed"]}
            row["source"] = r["source"]
            row["root_id"] = r["root_id"]
            rows.append(row)
    return rows


def auc(scores: List[float], labels: List[int]) -> float:
    """Probability that a random 'changed' node scores higher than a random
    'unchanged' one (0.5 = no information, 1.0 = perfect). Mann-Whitney U
    with average ranks for ties."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def feature_auc_rows(rows) -> List[dict]:
    labels = [int(r["changed"]) for r in rows]
    out = []
    for name in FEATURES_NUMERIC:
        out.append({"feature": name, "auc": auc([r[name] for r in rows], labels)})
    return sorted(out, key=lambda d: -(d["auc"] if not math.isnan(d["auc"]) else 0))


def rule_row(name: str, flagged: List[bool], labels: List[int]) -> dict:
    tp = sum(f and l for f, l in zip(flagged, labels))
    n_flag = sum(flagged)
    n_pos = sum(labels)
    return {"rule": name,
            "flagged": n_flag / len(labels) if labels else math.nan,
            "precision": tp / n_flag if n_flag else math.nan,
            "recall": tp / n_pos if n_pos else math.nan}


def threshold_rows(rows) -> List[dict]:
    labels = [int(r["changed"]) for r in rows]
    rules = [
        ("in_check", lambda r: r["in_check"] >= 1),
        ("n_checks >= 1", lambda r: r["n_checks"] >= 1),
        ("n_checks >= 2", lambda r: r["n_checks"] >= 2),
        ("n_checks >= 3", lambda r: r["n_checks"] >= 3),
        ("threat_gain >= 300", lambda r: r["threat_gain"] >= 300),
        ("n_checks >= 2 AND threat >= 300",
         lambda r: r["n_checks"] >= 2 and r["threat_gain"] >= 300),
        ("max_gain >= 200 (first hypothesis)", lambda r: r["max_gain"] >= 200),
    ]
    return [rule_row(name, [f(r) for r in rows], labels) for name, f in rules]


def bucket_model(rows, feature: str = "n_checks", edges=DEFAULT_EDGES) -> dict:
    """
    For each bucket of `feature`: P(changed) (Laplace-smoothed) and the mean
    size of the change when it happens ('shift', centipawns, |v1 - v0| of the
    changed nodes, already clipped). Empty buckets inherit the previous
    bucket's numbers, never an invented value. Wilson 95% intervals included
    so thin buckets are visibly uncertain.
    """
    m = len(edges)
    n_b, k_b, shift_sum = [0] * m, [0] * m, [0.0] * m
    for r in rows:
        i = 0
        for idx, e in enumerate(edges):
            if r[feature] >= e:
                i = idx
        n_b[i] += 1
        if r["changed"]:
            k_b[i] += 1
            shift_sum[i] += abs(r["delta"])
    changed = [r for r in rows if r["changed"]]
    base = len(changed) / len(rows)
    base_shift = (sum(abs(r["delta"]) for r in changed) / len(changed)) if changed else 0.0
    p, shift, lo, hi = [], [], [], []
    for i in range(m):
        if n_b[i] == 0:
            p.append(p[-1] if p else base)
            shift.append(shift[-1] if shift else base_shift)
            lo.append(math.nan)
            hi.append(math.nan)
        else:
            p.append((k_b[i] + 1) / (n_b[i] + 2))
            shift.append(shift_sum[i] / k_b[i] if k_b[i] else (shift[-1] if shift else base_shift))
            l, h = wilson(k_b[i], n_b[i])
            lo.append(l)
            hi.append(h)
    return {"feature": feature, "edges": list(edges), "p": p, "shift": shift,
            "n": n_b, "k": k_b, "ci_lo": lo, "ci_hi": hi,
            "base_rate": base, "base_shift": base_shift}


def format_report(rows, model: dict) -> str:
    L = []
    L.append(f"{len(rows)} frontier nodes; base rate of 'value changed' = "
             f"{100 * model['base_rate']:.1f}%\n")
    L.append("Feature ranking quality (AUC, 0.5 = useless, 1.0 = perfect):")
    for d in feature_auc_rows(rows):
        L.append(f"  {d['feature']:<16}{d['auc']:6.3f}")
    L.append("\nTrigger rules (positive = value really changed):")
    L.append(f"  {'rule':<36}{'flagged':>9}{'precision':>11}{'recall':>9}")
    for d in threshold_rows(rows):
        f = lambda x: "   -  " if math.isnan(x) else f"{100 * x:5.1f}%"
        L.append(f"  {d['rule']:<36}{f(d['flagged']):>9}{f(d['precision']):>11}{f(d['recall']):>9}")
    feat = model["feature"]
    L.append(f"\nP(changed | {feat}) and mean size of the change when it happens:")
    L.append(f"  {feat + ' bucket':<18}{'n':>6}{'P(changed)':>12}   {'95% CI':<12}{'mean shift':>11}")
    e = model["edges"]
    for i in range(len(e)):
        lab = f"{e[i]}-{e[i+1]-1}" if i + 1 < len(e) and e[i+1]-1 > e[i] else (
            f"{e[i]}" if i + 1 < len(e) else f"{e[i]}+")
        ci = ("" if math.isnan(model["ci_lo"][i]) else
              f"[{100*model['ci_lo'][i]:.0f}%, {100*model['ci_hi'][i]:.0f}%]")
        L.append(f"  {lab:<18}{model['n'][i]:>6}{100*model['p'][i]:>11.1f}%   {ci:<12}{model['shift'][i]:>9.0f}cp")
    return "\n".join(L)
