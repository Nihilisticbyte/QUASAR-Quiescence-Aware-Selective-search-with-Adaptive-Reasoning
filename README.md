# Quiescence-Aware Minimax — Phase 0

## Status: DONE

Plain fixed-depth alpha-beta engine, verified correct. No TT, no
iterative deepening, no quiescence search, no extensions yet — that's
Phase 1 onward, on purpose. This phase exists only to prove the
foundation is solid before anything gets built on top of it.

## Files

- `engine/eval.py` — material + piece-square-table static evaluation
  (White-relative).
- `engine/search.py` — negamax alpha-beta with a node counter and
  simple MVV-LVA-style move ordering.
- `test_phase0.py` — the three sanity checks below.

## Run it

```bash
pip install python-chess --break-system-packages
python3 test_phase0.py
```

## What was verified (results from this run)

1. **Perft (move generator correctness)** — matched known-correct
   values at depths 1–4 (20, 400, 8902, 197281). This confirms
   `python-chess`'s legal-move generation is trustworthy, so any bugs
   from here on are in *our* code, not the library.
2. **Self-play** — the engine played 40 plies against itself at depth 3
   with zero illegal moves.
3. **Node-count baseline** — recorded node counts at depths 1–4 from the
   starting position (see table below). **Keep these numbers.** Every
   later phase (TT, quiescence search, extensions) should be compared
   against this baseline to check it's actually helping, not just
   adding complexity.

| Depth | Best move | Score (cp) | Nodes | Leaf nodes | Time |
|---|---|---|---|---|---|
| 1 | Nf3 | 50 | 20 | 20 | 0.00s |
| 2 | Nf3 | 0 | 83 | 63 | 0.02s |
| 3 | Nf3 | 50 | 776 | 693 | 0.17s |
| 4 | Nf3 | 0 | 3,108 | 2,323 | 0.71s |

Node counts roughly grow with branching factor per ply — expected for
plain alpha-beta with decent move ordering (no TT yet, so no cross-branch
reuse).

## Design notes for later phases

- **Score convention:** `evaluate()` in `eval.py` is always White-relative.
  `search.py` flips the sign for Black in the negamax convention — don't
  duplicate that flip elsewhere.
- **Terminal states** (checkmate/stalemate/draw) are handled in *both*
  `eval.py` and `search.py` right now — `search.py`'s checks fire first
  during search, `eval.py`'s only matter if `evaluate()` is called
  directly. Keep this in mind when quiescence search calls `evaluate()`
  at a non-terminal leaf.
- **`MATE_SCORE - ply`**: mate scores are offset by ply so the search
  prefers delivering mate sooner / being mated later. This matters once
  you add a TT (mate scores need adjusting relative to TT-stored depth)
  — flagged now so it's not a surprise in Phase 1.

## Next: Phase 1

Iterative deepening + Zobrist transposition table + captures-only
quiescence search baseline. This is the delegate-able part — the specs
should be tight enough to hand to your teammate.
