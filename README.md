# QUASAR — Quiescence-Aware Selective-search with Adaptive Reasoning

Project 16: Quiescence-Aware Minimax with Selective Search Extensions
(chess, Python, `python-chess`).

## Status

| Phase | What                                                            | Status      |
| ----- | --------------------------------------------------------------- | ----------- |
| 0     | Plain alpha-beta baseline (material + PST eval)                 | DONE        |
| 1     | Transposition table + quiescence search + iterative deepening   | DONE        |
| 2     | Evaluation harness (node budgets, Stockfish oracle, metrics)    | NEXT        |
| 3-6   | Instability signal, extension controller, experiments, write-up | Not started |

Phase 0 proved the foundation is correct. Phase 1 added the baseline
every later result is compared against: a TT, captures-only quiescence
search (the horizon-effect fix) and iterative deepening. No selective
extensions yet, on purpose.

## Files

- `engine/eval.py` — material + piece-square-table static evaluation
  (White-relative).
- `engine/search.py` — negamax alpha-beta with TT probing/storing,
  captures-only quiescence (stand-pat), `search_root` and
  `iterative_deepening`, plus a node counter that separates
  alpha-beta nodes from quiescence nodes.
- `engine/tt.py` — dict-backed transposition table, Zobrist-keyed,
  EXACT/LOWERBOUND/UPPERBOUND flags, no replacement policy yet.
- `play.py` — human-vs-engine command-line game (`--color`, `--depth`).
- `test_phase0.py` — perft, self-play legality, node-count baseline.
- `test_phase1.py` — legality, node counts vs Phase 0, quiescence
  recapture demo, TT consistency.

## Run it

```bash
pip install python-chess --break-system-packages
python3 test_phase0.py
python3 test_phase1.py
```

On Windows, create a virtual environment first
(`python -m venv .venv`, activate it, then `pip install python-chess`).

## Phase 0 results

1. **Perft (move generator correctness)** — matched known-correct
   values at depths 1–4 (20, 400, 8902, 197281). This confirms
   `python-chess`'s legal-move generation is trustworthy, so any bugs
   from here on are in _our_ code, not the library.
2. **Self-play** — the engine played 40 plies against itself at depth 3
   with zero illegal moves.
3. **Node-count baseline** — recorded node counts at depths 1–4 from the
   starting position (see table below). **Keep these numbers.** Every
   later phase (TT, quiescence search, extensions) should be compared
   against this baseline to check it's actually helping, not just
   adding complexity.

| Depth | Best move | Score (cp) | Nodes | Leaf nodes | Time  |
| ----- | --------- | ---------- | ----- | ---------- | ----- |
| 1     | Nf3       | 50         | 20    | 20         | 0.00s |
| 2     | Nf3       | 0          | 83    | 63         | 0.02s |
| 3     | Nf3       | 50         | 776   | 693        | 0.17s |
| 4     | Nf3       | 0          | 3,108 | 2,323      | 0.71s |

Node counts roughly grow with branching factor per ply — expected for
plain alpha-beta with decent move ordering (no TT yet, so no cross-branch
reuse).

## Phase 1 results (reported honestly)

- **Quiescence works:** in the test position, a naive snapshot after
  `Qxe5` reads +405cp for White; quiescence resolves the recapture and
  returns -500cp. That is the horizon-effect fix doing its job.
- **TT benefit is small at shallow depth.** Alpha-beta-only nodes
  (quiescence excluded) vs the Phase 0 baseline: depth 1: 20 vs 20,
  depth 2: 83 vs 83, depth 3: 785 vs 776 (slightly worse, probe
  overhead), depth 4: 2,917 vs 3,108 (about 6% fewer). The TT pays off
  mostly across iterative-deepening iterations.
- **Quiescence adds real nodes** (about +2,577 at depth 4, roughly as
  many as the alpha-beta nodes). That is the cost of fixing the horizon
  effect, and is why Phase 2 compares at equal node budgets.

## Known limitations

- Quiescence does not search check evasions (captures only).
- Mate scores are not ply-adjusted when stored in / read from the TT.
- TT has no replacement policy (always overwrite).
- `can_claim_draw()` is path-dependent and can interact with the TT
  (graph-history interaction); revisit if odd draw scores appear.
- Eval is material + PST only (no king safety or pawn structure).

## Fixed since Phase 1

- Quiescence now receives the real ply from the root (separate `q_ply`
  counter for its depth cap), so mate-distance scores are consistent
  between the main search and quiescence.

## Design notes for later phases

- **Score convention:** `evaluate()` in `eval.py` is always White-relative.
  `search.py` flips the sign for Black in the negamax convention — don't
  duplicate that flip elsewhere.
- **Terminal states** (checkmate/stalemate/draw) are handled in _both_
  `eval.py` and `search.py` right now — `search.py`'s checks fire first
  during search, `eval.py`'s only matter if `evaluate()` is called
  directly. Keep this in mind when quiescence search calls `evaluate()`
  at a non-terminal leaf.
- **`MATE_SCORE - ply`**: mate scores are offset by ply so the search
  prefers delivering mate sooner / being mated later. This matters once
  you add a TT (mate scores need adjusting relative to TT-stored depth)
  — flagged now so it's not a surprise in Phase 1.

## Next: Phase 2 — evaluation harness

1. Node-budgeted search (stop at N nodes, return last completed
   iteration's move).
2. Stockfish oracle via the python-chess UCI interface.
3. Decision-error metrics: move-match rate and centipawn loss.
4. Test set: WAC plus real-game positions.
5. Comparison runner: position x engine version -> move, nodes, time,
   error vs oracle.

Full plan and research framing: `project-plan.md`.
