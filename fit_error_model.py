"""
Build the Russell-Wefald-style error model (engine/error_model.py) from data.

For each sampled frontier position P (outside check) and each quiet checking
move m available there:

    q0   = plain quiescence value of P, full window
    v    = value of P if m is also searched: max(q0, -quiescence(P after m))
    e    = v - q0                    (>= 0; clipped at 1000 cp so a mate is not "99,000 cp")
    cost = nodes searching m took
    evasions = legal replies the opponent has after m (0 = checkmate)

Frontier positions are not invented: they are the depth-0 nodes a real search
visits (reservoir-sampled). TRAIN split only, because the model is a tuned
parameter set and the final comparison must use held-out TEST positions.

Usage:
    python fit_error_model.py                       # default sample, a few minutes
    python fit_error_model.py --wac-n 40 --real-n 40 --per-root 8   # quick trial
Writes data/error_log.csv (raw rows) and data/error_model.json (the model).
"""

import argparse
import csv
import os
import random
import time

import chess

from engine.error_model import fit, DEFAULT_EDGES, DEFAULT_PATH
from engine.extensions import quiet_checks
from engine.positions import select_positions
from engine.search import SearchStats, quiescence, check_pass, search_root, INF, MATE_BOUND
from engine.tt import TranspositionTable

VALUE_CLIP = 1000
FIELDS = ["root_id", "source", "depth", "fen", "move", "evasions", "q0", "e", "cost"]


def sample_frontier_fens(board, depth, k, rng):
    """Up to k frontier positions (FENs) a plain depth-`depth` search visits,
    chosen uniformly at random (reservoir sampling)."""
    res, seen = [], [0]

    def hook(b, ply):
        seen[0] += 1
        if len(res) < k:
            res.append(b.fen())
        else:
            j = rng.randrange(seen[0])
            if j < k:
                res[j] = b.fen()

    search_root(board.copy(), depth, TranspositionTable(), SearchStats(frontier_hook=hook))
    return res


def measure(fen):
    """One data row per quiet check at a frontier position (empty list if not eligible)."""
    b = chess.Board(fen)
    if b.is_check() or b.is_game_over():
        return []
    checks = quiet_checks(b)
    if not checks:
        return []
    q0 = quiescence(b, -INF, INF, SearchStats())
    if abs(q0) >= MATE_BOUND:
        return []
    out = []
    for m in checks:
        b.push(m)
        evasions = b.legal_moves.count()
        b.pop()
        st = SearchStats()
        v = check_pass(b, [m], q0, -INF, INF, st, 0)
        out.append({"move": m.uci(), "evasions": evasions, "q0": q0,
                    "e": max(0, min(v - q0, VALUE_CLIP)), "cost": st.nodes})
    return out


def print_table(rows):
    """Per-bucket summary. The last column is Clune's (2007) stability quotient,
    used the way he uses it, OFFLINE: total variance of the value over the sampled
    states divided by the adjacent variance, here the mean squared change one
    step (searching the check) makes. Near 1 = the value jumps around, large = it moves
    only incrementally. Low-quotient buckets are where a fixed-depth cut-off is
    least trustworthy, which is the check that these buckets are the right ones."""
    print(f"\n{'evasions':>9} {'n':>5} {'P(e>0)':>7} {'P(e>=100)':>10} "
          f"{'mean e|e>0':>11} {'cost (nodes)':>13} {'Clune stability':>16}")
    edges = list(DEFAULT_EDGES) + [10 ** 9]
    for k in range(len(DEFAULT_EDGES)):
        g = [r for r in rows if edges[k] <= r["evasions"] < edges[k + 1]]
        if not g:
            continue
        pos = [r["e"] for r in g if r["e"] > 0]
        qs = [r["q0"] for r in g]
        mean = sum(qs) / len(qs)
        total_var = sum((x - mean) ** 2 for x in qs) / len(qs)
        adj_var = sum(r["e"] ** 2 for r in g) / len(g)
        stab = total_var / adj_var if adj_var else float("inf")
        print(f"{edges[k]:>8}+ {len(g):>5} {len(pos)/len(g):>7.2f} "
              f"{sum(r['e']>=100 for r in g)/len(g):>10.3f} "
              f"{(sum(pos)/len(pos) if pos else 0):>11.0f} "
              f"{sum(r['cost'] for r in g)/len(g):>13.1f} {stab:>16.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wac", default=os.path.join("data", "wac.epd"))
    ap.add_argument("--pgn", default=os.path.join("data", "real_games.pgn"))
    ap.add_argument("--wac-n", type=int, default=100)
    ap.add_argument("--real-n", type=int, default=100)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3],
                    help="depths of the searches that produce frontier nodes")
    ap.add_argument("--per-root", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=os.path.join("data", "error_log.csv"))
    ap.add_argument("--out", default=DEFAULT_PATH)
    args = ap.parse_args()

    roots = select_positions(args.wac, args.pgn, args.wac_n, args.real_n,
                             split="train", seed=args.seed)
    rows, seen, t0 = [], set(), time.time()
    for i, root in enumerate(roots, 1):
        for d in args.depths:
            rng = random.Random(f"{args.seed}:{root.id}:{d}")
            for fen in sample_frontier_fens(root.board(), d, args.per_root, rng):
                epd = chess.Board(fen).epd()
                if epd in seen:
                    continue
                seen.add(epd)
                for m in measure(fen):
                    rows.append({"root_id": root.id, "source": root.source,
                                 "depth": d, "fen": fen, **m})
        if i % 25 == 0 or i == len(roots):
            print(f"  {i}/{len(roots)} roots, {len(rows)} checks measured, "
                  f"{(time.time()-t0)/60:.1f} min")

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    model = fit(rows, DEFAULT_EDGES)
    model.save(args.out, meta={"rows": len(rows), "seed": args.seed, "depths": args.depths,
                               "split": "train"})
    print_table(rows)
    print(f"\nwrote {args.log} and {args.out}")


if __name__ == "__main__":
    main()
