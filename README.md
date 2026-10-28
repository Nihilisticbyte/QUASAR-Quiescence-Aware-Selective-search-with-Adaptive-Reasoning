# QUASAR — Quiescence-Aware Selective-search with Adaptive Reasoning

Project 16: Quiescence-Aware Minimax with Selective Search Extensions
(chess, Python, `python-chess`).

**Thesis.** Under equal node budgets, a cost-benefit, budget-aware controller for checking moves at the search frontier reduces decision error relative to fixed-depth iterative-deepening alpha-beta
and captures-only quiescence search. Whether this holds is an empirical question: the
repository contains everything needed to measure it, and the plan/report record what was
actually found (see `project-plan.md`, `REPORT.md`).

## Status

| Phase | What                                                                                             | Status                                                      |
| ----- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 0     | Plain alpha-beta baseline (material + piece-square eval)                                         | done                                                        |
| 1     | Transposition table, quiescence, iterative deepening                                             | done                                                        |
| 2     | Evaluation harness: node budgets, Stockfish oracle, WAC + real-game positions, comparison runner | done                                                        |
| 3     | Error model: value gain and node cost of checking moves at frontier nodes (`fit_error_model.py`) | done                                                        |
| 4     | Controllers: check rule, blind checks (`qchk_all`), Russell-Wefald cost-benefit checks (`rw_*`)  | done                                                        |
| 5     | Paired comparison, ablations, engine-vs-engine matches                                           | tooling done, **full-scale experiments to be run**          |
| 6     | Write-up                                                                                         | skeleton in `REPORT.md`, tables via `make_report_tables.py` |

## Setup (Windows PowerShell, no virtual environment needed)

```powershell
pip install -r requirements.txt
# Stockfish: download from https://stockfishchess.org/download/ , extract, then:
$env:STOCKFISH_PATH = "C:\full\path\to\stockfish.exe"      # this window only
setx STOCKFISH_PATH "C:\full\path\to\stockfish.exe"        # permanent (open a new window afterwards)
python run_tests.py --fast                                 # everything except the slow Phase 0 self-play
```

Run every command from the project folder (the one containing `engine\`).
`py` works in place of `python` if `python` is not recognised.

## Files

| File                                                      | Purpose                                                                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `engine/eval.py`                                          | material + piece-square-table evaluation (White-relative)                                                                                                                 |
| `engine/tt.py`                                            | transposition table keyed by the exact position (python-chess position key)                                                                                               |
| `engine/search.py`                                        | alpha-beta, quiescence (with check evasions), iterative deepening, node budget, repetition tracking, extension hooks                                                      |
| `engine/error_model.py`, `fit_error_model.py`             | Russell-Wefald error model: per kind of check, quantiles of value gain and mean node cost, fit on TRAIN frontier nodes; also prints Clune's stability quotient per bucket |
| `engine/extensions.py`                                    | extension controllers and the engine registry (`controller_factories`)                                                                                                    |
| `engine/oracle.py`                                        | Stockfish oracle: best move, score, centipawn loss of a move                                                                                                              |
| `engine/positions.py`                                     | WAC / PGN position loading and the train/test split                                                                                                                       |
| `engine/harness.py`                                       | runs engines at node budgets, scores them against Stockfish, summarises with confidence intervals                                                                         |
| `engine/compare.py`, `engine/report.py`                   | paired comparison statistics and Markdown tables                                                                                                                          |
| `engine/match.py`                                         | engine-vs-engine matches                                                                                                                                                  |
| `run_baseline.py`                                         | main experiment runner (any set of engines)                                                                                                                               |
| `compare_engines.py`, `make_report_tables.py`, `match.py` | Phase 5/6 scripts                                                                                                                                                         |
| `play.py`                                                 | play against the engine in the terminal                                                                                                                                   |
| `run_tests.py`, `test_*.py`                               | tests (`python run_tests.py`)                                                                                                                                             |
| `data/wac.epd`                                            | Win At Chess suite, 300 tactical positions with book moves                                                                                                                |
| `data/real_games.pgn`                                     | real games to sample non-tactical-suite positions from                                                                                                                    |

## Engine versions (names for `--engines`)

| Name                | What it does                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`          | iterative deepening + TT + quiescence (captures and check evasions)                                                                                                     |
| `check`             | baseline + the classic rule: +1 ply when the move gives check                                                                                                           |
| `qchk_all`          | check rule + EVERY quiet check searched at every frontier node (no decision; the brute-force comparator)                                                                |
| `rw_l0.1 ... rw_l2` | check rule + each quiet check at a frontier node is searched only if expected gain inside the (alpha, beta) window >= lambda x its measured node cost x budget pressure |
| `rand_p33`          | CONTROL: each quiet check searched with probability 1/3 (same pass rate as `rw_l0.1`, no logic)                                                                         |
| ablations           | `rw_l0.1_nobudget` (constant price), `rw_l0.1_nowindow` (no relevance test), `rw_l0.1_nocheck` (no +1-ply check rule)                                                   |

Add a new variant by adding one line to `controller_factories()` in `engine/extensions.py`.

## Running the research (in this order)

```powershell
# Phase 3: fit the error model on TRAIN frontier nodes (~2 min, needs no Stockfish)
python fit_error_model.py

# Phase 4/5: tune on TRAIN positions, never on test
python run_baseline.py --split train --engines baseline check qchk_all rw_l0.1 rand_p33 --budgets 8000 --out results\train.csv

# Phase 5: final comparison on held-out TEST positions (use the same --oracle-depth and Stockfish for every file you compare)
python run_baseline.py --split test --engines baseline check qchk_all rw_l0.1 rand_p33 --out results\test.csv
python compare_engines.py --results results\test.csv
python make_report_tables.py --results results\test.csv

# Optional headline number (lowest rigour)
python match.py --a check --b baseline --budget 2000 --openings 20 --out results\match_check.csv
```

`run_baseline.py` saves after every run and can be stopped and restarted. It writes a
`.meta.json` next to each results file (search version, Stockfish version, oracle depth)
and refuses to append results made under different settings.

## Method in one page

- **Equal node budgets.** Every engine gets the same number of TOTAL nodes (alpha-beta +
  quiescence) per position; iterative deepening returns the best move found when the budget ends.
- **Primary metric: decision error.** Centipawn loss of the chosen move against Stockfish
  (fixed depth, one thread), plus move-match rate, book-move solve rate (WAC) and big-miss rate.
  Rates carry Wilson 95% intervals, mean loss a bootstrap interval.
- **Paired comparison.** Versions are compared position by position (same positions, same budgets):
  mean difference with bootstrap CI, exact sign test, McNemar for solves.
- **Cost beyond nodes.** Extension logic costs time that node counts hide; the tables report the
  wall-clock ratio and extensions per search.
  ply changes their value by >= 100 cp; `analyze_instability.py` reports which features predict that.
- **Held-out test.** WAC is shuffled and halved; real games are split by game. Tune on train, report on test.

## Known limitations

- Evaluation is material + piece-square tables only (no king safety, pawn structure, mobility).
  It is not the research contribution, but it caps absolute strength and makes tactics look
  different from a strong engine's.
- Draw detection inside the search is two-fold repetition on the current path and the fifty-move
  rule; scores stored in the transposition table can still embed path-dependent draws.
- Extensions that depend on the (alpha, beta) window can reach the same position at different
  depths along different paths; the TT stores the depth actually searched (see `engine/extensions.py`); the check decision is made at depth-0 nodes, which are never stored.
- Stockfish is depth-limited and version-dependent; the oracle is an approximation of the truth.
- The real-game set mixes eras and one source (McDonnell, 1830s) contributes a large share of games.
- Python speed: node budgets are measured in nodes, so conclusions do not depend on machine speed,
  but the extensions' extra work per node does cost wall-clock time.

## References

- S. Russell and E. Wefald, _On Optimal Game-Tree Search using Rational Meta-Reasoning_, IJCAI 1989.
- A. Plaat, J. Schaeffer, W. Pijls, A. de Bruin, _Best-First Fixed-Depth Game-Tree Search in Practice_, IJCAI 1995.
- J. Clune, _Heuristic Evaluation Functions for General Game Playing_, AAAI 2007.
- Win At Chess positions: `wacnew.epd` from github.com/fsmosca/EPD-Analyzer (GPL-3.0).
- Test games: Chessentials PGN collections (chessentials.com/chess-pgn-downloads/) and the Croatian championship files.
