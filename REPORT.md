# Quiescence-Aware Minimax with Selective Search Extensions (QUASAR)

> **How to use this file.** Sections 1-4 and 7 describe what was built and are ready to edit.
> Sections 5-6 are placeholders: fill them ONLY from your own full-scale runs
> (`python make_report_tables.py --results results/test.csv` gives the tables).
> Do not copy numbers from sandbox/trial runs. Every `[FILL]` marks something that needs a real number.

## Abstract

[FILL after results: one paragraph. Problem (horizon effect, fixed-depth cut-off), approach
(cost-benefit selection of checking moves at the search frontier under equal node budgets), main finding with numbers,
main limitation.]

## 1. Problem

A depth-limited alpha-beta search stops at a fixed horizon, so positions that are tactically
unsettled there (a capture pending, a check, a mating attack) are scored as if they were quiet.
Quiescence search repairs the capture case by continuing until captures run out. Selective
extension is a different tool: it spends additional depth on chosen branches earlier in the
tree. The question studied here is whether spending that extra effort _selectively_, where it is
most likely to change the minimax value, beats spending the same number of nodes uniformly, at
equal node budgets.

The two ideas must not be conflated: quiescence continues a line to a quiet position;
an extension adds depth to a branch that is still inside the nominal search.

## 2. Background and how we use it

- **Russell and Wefald (1989), rational meta-reasoning.** Search steps are decisions whose value is
  the expected improvement in the chosen move, weighed against the cost of computing. Their MGSS*
  is best-first, keeps the whole tree in memory and needs an error distribution learned offline
  from about 35,000 positions; it does not fit a depth-first alpha-beta engine directly. We borrow
  the *idea\* (expected gain versus cost), not the algorithm: the controller
  searches a check when `expected gain in window >= lambda x cost`. They themselves point to selective extension
  beyond the alpha-beta horizon (singular extensions) as the natural application.
- **Plaat et al. (1995), MTD(f).** Their experiments used iterative deepening and a transposition table
  on real programs and found MTD(f) beating aspiration NegaScout on leaf nodes, total nodes and time.
  They disabled search extensions in those experiments, so no result of theirs covers MTD(f) _with_
  extensions. They also warn that standard test suites alone are poor predictors of play, which is why
  we use real-game positions as well as WAC. We use plain alpha-beta with a TT as the driver; MTD(f)
  is a possible extension. [Note: it is SSS*/MTD(+inf), not MTD(f), that visits too many interior nodes.]
- **Clune (2007), stability.** Clune's stability quotient (total variance over adjacent variance of a
  feature) is an _offline_ tool for selecting evaluation features from sampled states. Our per-node
  trigger is our own design; `fit_error_model.py` reports his quotient offline per bucket
  as a diagnostic only.

## 3. Method

### 3.1 Baseline engine

Negamax alpha-beta with a transposition table (exact position key, depth-preferred probing with
EXACT/LOWER/UPPER bounds), MVV-LVA move ordering, iterative deepening, quiescence search
(stand-pat captures; all evasions when in check), two-fold repetition and fifty-move draw
detection, and mate scores stored relative to the node. Evaluation: material and piece-square tables.

### 3.2 Equal node budgets

Every engine receives the same number of total nodes per position. Iterative deepening stops when
the budget is exhausted and plays the best move found among the root moves whose search finished.

### 3.3 Extension controllers

`check`: +1 ply when a move gives check (capped at 4 per line). `qchk_all`: additionally, at every depth-0
node, plain quiescence runs first and then every quiet checking move is searched (no decision).
`rw_l*`: the same, but each quiet check is searched only if

    expected gain inside (alpha, beta)  >=  lambda x mean node cost x budget pressure.

Expected gain is the mean over K empirical quantiles e of the error distribution of
max(0, min(q + e, beta) - max(q, alpha)), where q is the plain quiescence value. Because extra moves can only
raise the value of the side to move, a node with q >= beta is skipped exactly, and value movement outside the
window is worth nothing (Russell and Wefald's "relevant node" idea in alpha-beta form). Budget pressure is
1 / (fraction of nodes remaining), capped at 20. Checks are searched most forcing first. `rand_p33` searches each
check with probability 1/3 and is a control. All decisions are made at depth-0 nodes, which are never stored in the
TT, so window-dependent logic cannot change what the table stores; this is the answer to the TT risk in the plan.
Not implemented: MGSS\* (best-first), MTD(f) with extensions (untested in the literature).

### 3.4 Learning the error model

`fit_error_model.py` samples frontier nodes visited by real depth-2 and depth-3 searches from TRAIN positions and,
for each quiet check there, records e = value gain from searching it (clipped at 0 and 1000 cp), the nodes it took,
and the number of replies the opponent has. Replies were chosen as the bucket feature because it predicted a useful
check better than the number of checks at the node (AUC 0.76 vs 0.67 on the training sample). The same script
prints Clune's stability quotient per bucket (total variance of the value over adjacent-step variance), used
offline as he uses it; it is a summary of the same data, not independent evidence. [FILL: bucket table from your run.]

## 4. Experimental setup

- **Positions.** WAC (300 tactical positions with book moves) and positions sampled from real games
  [FILL: number of games, sources, share of the largest source]. WAC is shuffled and halved; real games
  are split by game number into train and test. Thresholds and lambda are tuned on train only;
  all reported numbers are from test.
- **Oracle.** Stockfish [FILL: version] at fixed depth [FILL], one thread.
- **Budgets.** [FILL: node budgets used]. **Sample sizes.** [FILL: positions per source].
- **Metrics, in priority order.** (1) decision error: mean centipawn loss against Stockfish, move-match rate,
  book solve rate, rate of big misses (>= 200 cp); (2) nodes, at equal budgets, plus extensions per search and
  wall-clock ratio; (3) selection quality: does `rw_*` beat the random control `rand_p33` at the same pass rate;
  (4) playing strength from engine-vs-engine matches (lowest rigour).
- **Statistics.** Wilson intervals for rates, bootstrap intervals for mean loss, paired differences per
  position, exact sign test and McNemar test. [Note the number of comparisons made if reporting p-values.]

## 5. Results

### 5.1 Trigger analysis (Phase 3)

[FILL: feature AUC table, threshold-rule table, P(change | n_checks) table.]

### 5.2 Tuning on the train split

[FILL: which settings were tried, which was chosen and why.]

### 5.3 Test-split comparison (Phase 5)

[FILL: tables from make_report_tables.py. Report ALL engines tried, not only the best.]

### 5.4 Ablations

[FILL: `rw_l0.1_nowindow`, `rw_l0.1_nobudget`, `rw_l0.1_nocheck`, and the `rand_p33` control.]

### 5.5 Playing strength

[FILL if run: match.py output with the confidence interval and number of games.]

## 6. Discussion

[FILL: Did the selective controller beat (a) the baseline and (b) the fixed check rule at equal nodes?
How large is the effect compared with its confidence interval? What does it cost in wall-clock time?
Is the effect different on WAC and on real games? Which ablation matters? Be explicit about negative results.]

## 7. Limitations

1. Material + piece-square evaluation only; absolute strength and tactical judgement are limited by it.
2. Window-dependent extensions can reach the same position at different depths; the TT stores the depth actually searched.
3. Path-dependent draw scores can be stored in the TT (graph-history interaction).
4. Stockfish is depth-limited and version-specific; centipawn loss is relative to it, not to the truth.
5. Sample sizes bound what can be detected; intervals are reported so this is visible.
6. The real-game set is not representative of modern play (mixed eras, one dominant source, famous
   games are tactically rich). WAC is a puzzle suite whose book moves are not always unique.
7. Many engine variants were compared, so a few "significant" p-values are expected by chance.
8. Python speed: all conclusions are at equal NODE counts; the extensions' extra per-node work is reported as wall-clock ratio.

## 8. Conclusion

[FILL.]

## References

See `README.md` (Russell and Wefald 1989; Plaat et al. 1995; Clune 2007; WAC source; PGN sources).
