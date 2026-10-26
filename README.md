# QUASAR — Quiescence-Aware Selective-search with Adaptive Reasoning

Project 16: Quiescence-Aware Minimax with Selective Search Extensions
(chess, Python, `python-chess`).

**Thesis.** Under equal node budgets, an instability-triggered, budget-aware extension
controller reduces decision error relative to fixed-depth iterative-deepening alpha-beta
and captures-only quiescence search. Whether this holds is an empirical question: the
repository contains everything needed to measure it, and the plan/report record what was
actually found (see `project-plan.md`, `REPORT.md`).

## Status

| Phase | What                                                                                             | Status                                                      |
| ----- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 0     | Plain alpha-beta baseline (material + piece-square eval)                                         | done                                                        |
| 1     | Transposition table, quiescence, iterative deepening                                             | done                                                        |
| 2     | Evaluation harness: node budgets, Stockfish oracle, WAC + real-game positions, comparison runner | done                                                        |
| 3     | Instability features, ground-truth labelling, trigger analysis, stability diagnostic             | done                                                        |
| 4     | Extension controllers (check rule, instability trigger, cost-weighted, checks-in-quiescence)     | done                                                        |
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

| File                                                                  | Purpose                                                                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `engine/eval.py`                                                      | material + piece-square-table evaluation (White-relative)                                                            |
| `engine/tt.py`                                                        | transposition table keyed by the exact position (python-chess position key)                                          |
| `engine/search.py`                                                    | alpha-beta, quiescence (with check evasions), iterative deepening, node budget, repetition tracking, extension hooks |
| `engine/extensions.py`                                                | extension controllers and the engine registry (`controller_factories`)                                               |
| `engine/features.py`                                                  | position-only instability features (`n_checks`, `threat_gain`, ...)                                                  |
| `engine/stability.py`                                                 | Clune-style stability diagnostic (total vs adjacent variance)                                                        |
| `engine/trigger_log.py`, `engine/trigger_analysis.py`                 | Phase 3: frontier sampling, ground-truth labels, trigger accuracy, probability model                                 |
| `engine/oracle.py`                                                    | Stockfish oracle: best move, score, centipawn loss of a move                                                         |
| `engine/positions.py`                                                 | WAC / PGN position loading and the train/test split                                                                  |
| `engine/harness.py`                                                   | runs engines at node budgets, scores them against Stockfish, summarises with confidence intervals                    |
| `engine/compare.py`, `engine/report.py`                               | paired comparison statistics and Markdown tables                                                                     |
| `engine/match.py`                                                     | engine-vs-engine matches                                                                                             |
| `run_baseline.py`                                                     | main experiment runner (any set of engines)                                                                          |
| `log_instability.py`, `analyze_instability.py`, `stability_report.py` | Phase 3 scripts                                                                                                      |
| `compare_engines.py`, `make_report_tables.py`, `match.py`             | Phase 5/6 scripts                                                                                                    |
| `play.py`                                                             | play against the engine in the terminal                                                                              |
| `run_tests.py`, `test_*.py`                                           | tests (`python run_tests.py`)                                                                                        |
| `data/wac.epd`                                                        | Win At Chess suite, 300 tactical positions with book moves                                                           |
| `data/real_games.pgn`                                                 | real games to sample non-tactical-suite positions from                                                               |
| `data/trigger_model.json`                                             | P(value changes \| number of checks), built by `analyze_instability.py`                                              |

## Engine versions (names for `--engines`)

| Name                        | What it does                                                                                                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`                  | iterative deepening + TT + quiescence (captures and check evasions)                                                                                                        |
| `check`                     | baseline + the classic rule: +1 ply when the move gives check                                                                                                              |
| `inst_c1/c2/c3`             | check rule + 1 extra ply at the search frontier when the side to move has at least 1/2/3 checking moves and the value is near the window                                   |
| `qchk_all`, `qchk_c1/c2/c3` | check rule + quiet checking moves searched inside quiescence at the frontier (always / when triggered)                                                                     |
| `costw_l10/20/40/80`        | frontier extension when P(change) x shift >= lambda x cost, from `data/trigger_model.json`                                                                                 |
| ablations                   | `inst_c2_nowin` (no window gate), `inst_c2_nocheck` (no check rule), `qchk_c2_nowin`, `inst_gain200` (first hypothesis: capture gain; kept to report that it did not work) |

Add a new variant by adding one line to `controller_factories()` in `engine/extensions.py`.

## Running the research (in this order)

```powershell
# Phase 3: which positions deserve extra search? (no Stockfish needed, a few minutes)
python log_instability.py                    # frontier positions + ground-truth labels, TRAIN split
python analyze_instability.py                # accuracy of each feature; writes data\trigger_model.json
python stability_report.py                   # Clune-style stability table

# Phase 4/5: tune on TRAIN positions, never on test
python run_baseline.py --split train --engines baseline check inst_c2 qchk_c2 costw_l40 --budgets 8000 --out results\train.csv

# Phase 5: final comparison on held-out TEST positions (use the same --oracle-depth and Stockfish for every file you compare)
python run_baseline.py --split test --engines baseline check inst_c2 qchk_c2 costw_l40 --out results\test.csv
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
- **Trigger design from data.** `log_instability.py` labels frontier positions by whether one more
  ply changes their value by >= 100 cp; `analyze_instability.py` reports which features predict that.
- **Held-out test.** WAC is shuffled and halved; real games are split by game. Tune on train, report on test.

## Known limitations

- Evaluation is material + piece-square tables only (no king safety, pawn structure, mobility).
  It is not the research contribution, but it caps absolute strength and makes tactics look
  different from a strong engine's.
- Draw detection inside the search is two-fold repetition on the current path and the fifty-move
  rule; scores stored in the transposition table can still embed path-dependent draws.
- Extensions that depend on the (alpha, beta) window can reach the same position at different
  depths along different paths; the TT stores the depth actually searched (see `engine/extensions.py`).
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
