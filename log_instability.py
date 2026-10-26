"""
Phase 3: log instability features and ground-truth labels for frontier nodes.

For each root position (WAC + real games) we run a short search, sample some
of the frontier positions it reaches, and for each one record its features
and whether one more ply of search changes its value (see engine/trigger_log.py).

Usage (from the project folder):
    python log_instability.py                       # TRAIN split, ~200 roots x 12 frontier nodes
    python log_instability.py --wac-n 30 --real-n 30 --per-root 8   # quick trial

The default is the TRAIN split on purpose: the trigger threshold and the
probability model are fitted on this data, and the final experiments must
use held-out TEST positions (run_baseline.py --split test).
The log is appended to as it goes and skipped roots are not repeated, so the
script can be stopped and restarted.

Next step: python analyze_instability.py
"""

import argparse
import csv
import os
import random
import time

import chess

from engine.positions import select_positions
from engine.trigger_log import (sample_frontier_fens, label_position, LOG_FIELDS,
                                DEFAULT_TAU)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wac", default=os.path.join("data", "wac.epd"))
    ap.add_argument("--pgn", default=os.path.join("data", "real_games.pgn"))
    ap.add_argument("--wac-n", type=int, default=100)
    ap.add_argument("--real-n", type=int, default=100)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--depth", type=int, default=2, help="depth of the search that produces frontier nodes")
    ap.add_argument("--per-root", type=int, default=12, help="frontier positions sampled per root")
    ap.add_argument("--tau", type=int, default=DEFAULT_TAU, help="value change (cp) that counts as 'changed'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "instability_log.csv"))
    args = ap.parse_args()

    if args.real_n and not os.path.exists(args.pgn):
        raise SystemExit(f"Real-game PGN not found: {args.pgn} (or use --real-n 0)")
    roots = select_positions(args.wac, args.pgn, args.wac_n, args.real_n,
                             split=args.split, seed=args.seed)

    done, seen_epd = set(), set()
    new_file = not (os.path.exists(args.out) and os.path.getsize(args.out) > 0)
    if not new_file:
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["root_id"])
                seen_epd.add(chess.Board(row["fen"]).epd())
        print(f"resuming: {len(done)} roots already logged in {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0, rows_written = time.time(), 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        for i, root in enumerate(roots, 1):
            # one RNG draw sequence per root keeps results independent of which
            # roots were skipped by a resume
            root_rng = random.Random(f"{args.seed}:{root.id}")
            if root.id in done:
                continue
            for fen in sample_frontier_fens(root.board(), args.depth, args.per_root, root_rng):
                epd = chess.Board(fen).epd()
                if epd in seen_epd:
                    continue
                seen_epd.add(epd)
                row = label_position(fen, args.tau)
                row.update(root_id=root.id, source=root.source)
                w.writerow(row)
                rows_written += 1
            f.flush()
            if i % 20 == 0 or i == len(roots):
                print(f"  {i}/{len(roots)} roots, {rows_written} new rows, "
                      f"{(time.time()-t0)/60:.1f} min")
    print(f"done: {args.out}  (tau={args.tau}cp, split={args.split})")


if __name__ == "__main__":
    main()
