"""
Comparison harness (Phase 2): run engine versions on test positions at fixed
NODE budgets, score every move against Stockfish, and summarise.

Pieces
  - ENGINES: name -> function(board, node_budget) -> (move, SearchStats).
    Adding a later engine version (e.g. with extensions) = adding one entry,
    so every version runs through exactly the same measurement code.
  - run_experiment(): loops position x budget x engine, one `Result` per run,
    appended to a CSV after EACH run (a crash or Ctrl-C loses nothing, and
    re-running with the same CSV skips finished runs).
  - summarize(): per (engine, source, budget): solve rate vs the book move
    (WAC only), match rate vs Stockfish, mean centipawn loss, big-miss rate,
    with 95% confidence intervals (Wilson for rates, bootstrap for the mean).

What is measured and why (see project-plan.md, "Evaluation methodology"):
  primary metric = centipawn loss vs Stockfish; cost = TOTAL nodes
  (alpha-beta + quiescence), compared at EQUAL node budgets.

Every run uses a fresh transposition table (iterative_deepening creates
one), so no information leaks from one position to the next.
"""

import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, fields, asdict
from typing import Callable, Dict, List, Optional

import chess

from .oracle import StockfishOracle
from .positions import Position
from .extensions import controller_factories
from .search import iterative_deepening, SEARCH_VERSION

BIG_MISS_CP = 200   # a move losing >= this many centipawns counts as a "big miss"


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def baseline_engine(board: chess.Board, budget: int):
    """Phase 1 engine: iterative-deepening alpha-beta + TT + quiescence."""
    move, _score, stats, _hist = iterative_deepening(board, 64, node_budget=budget)
    return move, stats


def controller_engine(factory: Callable):
    """Engine version = the Phase 1 search plus the extension controller that
    `factory()` builds. A FRESH controller is made per search, so no state
    leaks between positions."""
    def engine(board: chess.Board, budget: int):
        move, _score, stats, _hist = iterative_deepening(
            board, 64, node_budget=budget, controller=factory())
        return move, stats
    return engine


ENGINES: Dict[str, Callable] = {"baseline": baseline_engine}
for _name, _factory in controller_factories().items():
    ENGINES[_name] = controller_engine(_factory)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@dataclass
class Result:
    engine: str
    pos_id: str
    source: str
    budget: int
    move: str                       # UCI
    nodes: int                      # total nodes actually used (<= budget)
    depth: int                      # completed iterative-deepening depth; 0 = could not finish depth 1
    time_s: float
    oracle_best: str                # Stockfish's best move (UCI)
    oracle_match: bool              # engine move == Stockfish best move
    book_hit: Optional[bool]        # None when the position has no book answer
    cp_loss: int                    # centipawn loss vs Stockfish, >= 0
    extensions: int = 0             # times the controller added depth during this search


_FIELDS = [f.name for f in fields(Result)]


def _to_row(r: Result) -> dict:
    d = asdict(r)
    d["book_hit"] = "" if r.book_hit is None else r.book_hit
    return d


def _from_row(row: dict) -> Result:
    return Result(
        engine=row["engine"], pos_id=row["pos_id"], source=row["source"],
        budget=int(row["budget"]), move=row["move"], nodes=int(row["nodes"]),
        depth=int(row["depth"]), time_s=float(row["time_s"]),
        oracle_best=row["oracle_best"], oracle_match=row["oracle_match"] == "True",
        book_hit=None if row["book_hit"] == "" else row["book_hit"] == "True",
        cp_loss=int(row["cp_loss"]), extensions=int(row.get("extensions") or 0))


def load_results(path: str) -> List[Result]:
    with open(path, newline="") as f:
        return [_from_row(row) for row in csv.DictReader(f)]


def run_one(name: str, engine: Callable, pos: Position, budget: int,
            oracle: StockfishOracle) -> Result:
    board = pos.board()
    t0 = time.perf_counter()
    move, stats = engine(board, budget)
    dt = time.perf_counter() - t0
    best = oracle.analyse(pos.board()).best_move
    return Result(
        engine=name, pos_id=pos.id, source=pos.source, budget=budget,
        move=move.uci(), nodes=stats.nodes, depth=stats.completed_depth,
        time_s=round(dt, 3),
        oracle_best=best.uci() if best else "",
        oracle_match=(move == best),
        book_hit=(move in pos.best_moves) if pos.best_moves else None,
        cp_loss=oracle.move_loss(pos.board(), move),
        extensions=stats.extensions)


# ---------------------------------------------------------------------------
# Many runs
# ---------------------------------------------------------------------------

def settings_meta(oracle: StockfishOracle) -> dict:
    """Everything that must be identical for two runs to be comparable."""
    return {"search_version": SEARCH_VERSION, "oracle": oracle.version,
            "oracle_depth": oracle.depth, "fields": _FIELDS}


def _check_meta(out_csv: str, meta: dict):
    """Refuse to append to a results file made under different settings:
    mixing Stockfish versions, oracle depths or search versions in one file
    would silently corrupt every comparison made from it."""
    path = out_csv + ".meta.json"
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), meta[k]) for k in meta if old.get(k) != meta[k]}
        if diff:
            raise ValueError(
                f"{out_csv} was produced under different settings {diff}.\n"
                "Use a new --out file (or delete the old one) so results are not mixed.")
    else:
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)


def run_experiment(engines: Dict[str, Callable], positions: List[Position],
                   budgets: List[int], oracle: StockfishOracle,
                   out_csv: str = None, progress=print) -> List[Result]:
    """Run every engine on every position at every budget. If `out_csv`
    exists, finished runs (engine, position id, budget) are loaded and
    skipped, so an interrupted experiment can simply be started again.
    A `<out_csv>.meta.json` file records the settings (search version,
    Stockfish version, oracle depth, columns); resuming under different
    settings raises an error instead of mixing results."""
    results: List[Result] = []
    done = set()
    new_file = True
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        _check_meta(out_csv, settings_meta(oracle))
    if out_csv and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        results = load_results(out_csv)
        done = {(r.engine, r.pos_id, r.budget) for r in results}
        new_file = False
        progress(f"resuming: {len(done)} runs already in {out_csv}")

    todo = [(pos, b, n) for pos in positions for b in budgets for n in engines
            if (n, pos.id, b) not in done]
    progress(f"{len(todo)} runs to do ({len(positions)} positions x {len(budgets)} "
             f"budgets x {len(engines)} engines)")

    out = None
    writer = None
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        out = open(out_csv, "a", newline="")
        writer = csv.DictWriter(out, fieldnames=_FIELDS)
        if new_file:
            writer.writeheader()

    start = time.perf_counter()
    try:
        for i, (pos, budget, name) in enumerate(todo, 1):
            r = run_one(name, engines[name], pos, budget, oracle)
            results.append(r)
            if writer:
                writer.writerow(_to_row(r))
                out.flush()
            if i % 20 == 0 or i == len(todo):
                el = time.perf_counter() - start
                eta = el / i * (len(todo) - i)
                progress(f"  {i}/{len(todo)} done  ({el/60:.1f} min elapsed, "
                         f"~{eta/60:.1f} min left)")
    finally:
        if out:
            out.close()
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n (sane for small n and p near 0/1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: List[float], reps: int = 1000, seed: int = 0):
    """95% percentile-bootstrap interval for the mean (centipawn loss is
    heavily skewed, so a normal approximation would be misleading)."""
    n = len(values)
    if n == 0:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return (means[int(0.025 * reps)], means[int(0.975 * reps) - 1])


def summarize(results: List[Result]) -> List[dict]:
    groups: Dict[tuple, List[Result]] = {}
    for r in results:
        groups.setdefault((r.engine, r.source, r.budget), []).append(r)

    rows = []
    for (engine, source, budget), rs in sorted(groups.items()):
        n = len(rs)
        losses = [r.cp_loss for r in rs]
        book = [r for r in rs if r.book_hit is not None]
        solved = sum(r.book_hit for r in book)
        matched = sum(r.oracle_match for r in rs)
        big = sum(l >= BIG_MISS_CP for l in losses)
        lo, hi = bootstrap_mean_ci(losses)
        rows.append({
            "engine": engine, "source": source, "budget": budget, "n": n,
            "failed": sum(r.depth == 0 for r in rs),
            "mean_depth": statistics.mean(r.depth for r in rs),
            "mean_nodes": statistics.mean(r.nodes for r in rs),
            "mean_time_s": statistics.mean(r.time_s for r in rs),
            "mean_ext": statistics.mean(r.extensions for r in rs),
            "book_n": len(book),
            "book_solve": solved / len(book) if book else math.nan,
            "book_solve_lo": wilson(solved, len(book))[0],
            "book_solve_hi": wilson(solved, len(book))[1],
            "oracle_match": matched / n,
            "oracle_match_lo": wilson(matched, n)[0],
            "oracle_match_hi": wilson(matched, n)[1],
            "mean_loss": statistics.mean(losses),
            "mean_loss_lo": lo, "mean_loss_hi": hi,
            "median_loss": statistics.median(losses),
            "big_miss": big / n,
        })
    return rows


def write_summary_csv(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def format_summary(rows: List[dict]) -> str:
    pct = lambda x: "   -  " if math.isnan(x) else f"{100 * x:5.1f}%"
    lines = []
    head = (f"{'engine':<14}{'source':<7}{'budget':>7}{'n':>5}{'fail':>5}{'depth':>7}{'ext':>7}"
            f"{'book solve [95% CI]':>27}{'= SF move':>11}{'mean cp loss [95% CI]':>27}"
            f"{'median':>8}{'>=%dcp' % BIG_MISS_CP:>8}")
    lines += [head, "-" * len(head)]
    for r in rows:
        book = ("   -" if r["book_n"] == 0 else
                f"{pct(r['book_solve'])} [{100*r['book_solve_lo']:4.1f},{100*r['book_solve_hi']:5.1f}]")
        loss = f"{r['mean_loss']:6.1f} [{r['mean_loss_lo']:5.1f},{r['mean_loss_hi']:6.1f}]"
        lines.append(f"{r['engine']:<14}{r['source']:<7}{r['budget']:>7}{r['n']:>5}"
                     f"{r['failed']:>5}{r['mean_depth']:>7.1f}{r['mean_ext']:>7.1f}{book:>27}"
                     f"{pct(r['oracle_match']):>11}{loss:>27}"
                     f"{r['median_loss']:>8.0f}{pct(r['big_miss']):>8}")
    return "\n".join(lines)
