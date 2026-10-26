# Quiescence-Aware Minimax with Selective Search Extensions (QUASAR)

> **How to use this file.** Sections 1-4 and 7 describe what was built and are ready to edit.
> Sections 5-6 are placeholders: fill them ONLY from your own full-scale runs
> (`python make_report_tables.py --results results/test.csv` gives the tables).
> Do not copy numbers from sandbox/trial runs. Every `[FILL]` marks something that needs a real number.

## Abstract

[FILL after results: one paragraph. Problem (horizon effect, fixed-depth cut-off), approach
(instability-triggered extensions under equal node budgets), main finding with numbers,
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
  the *idea\* (expected gain versus cost), not the algorithm: the cost-weighted controller
  extends when `P(change) x shift >= lambda x cost`. They themselves point to selective extension
  beyond the alpha-beta horizon (singular extensions) as the natural application.
- **Plaat et al. (1995), MTD(f).** Their experiments used iterative deepening and a transposition table
  on real programs and found MTD(f) beating aspiration NegaScout on leaf nodes, total nodes and time.
  They disabled search extensions in those experiments, so no result of theirs covers MTD(f) _with_
  extensions. They also warn that standard test suites alone are poor predictors of play, which is why
  we use real-game positions as well as WAC. We use plain alpha-beta with a TT as the driver; MTD(f)
  is a possible extension. [Note: it is SSS*/MTD(+inf), not MTD(f), that visits too many interior nodes.]
- **Clune (2007), stability.** Clune's stability quotient (total variance over adjacent variance of a
  feature) is an _offline_ tool for selecting evaluation features from sampled states. Our per-node
  trigger is our own design; `stability_report.py` applies Clune's diagnostic offline to our features
  as context only.

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

[Describe each registered engine, see README table. State what is path-dependent and the TT trade-off.]

### 3.4 Choosing the trigger from data

`log_instability.py` samples frontier positions (where the depth-limited search hands over to
quiescence) and labels each by whether one extra ply changes its value by at least 100 cp
(one pawn; smaller thresholds mix in the ~50 cp tempo effect of a piece-square table).
`analyze_instability.py` reports, per feature, the AUC and the precision/recall of threshold rules.
[FILL: the table from your run. Two things to state from the data: (a) whether the capture-gain
hypothesis held, (b) which feature predicted change best and by how much.]

## 4. Experimental setup

- **Positions.** WAC (300 tactical positions with book moves) and positions sampled from real games
  [FILL: number of games, sources, share of the largest source]. WAC is shuffled and halved; real games
  are split by game number into train and test. Thresholds and lambda are tuned on train only;
  all reported numbers are from test.
- **Oracle.** Stockfish [FILL: version] at fixed depth [FILL], one thread.
- **Budgets.** [FILL: node budgets used]. **Sample sizes.** [FILL: positions per source].
- **Metrics, in priority order.** (1) decision error: mean centipawn loss against Stockfish, move-match rate,
  book solve rate, rate of big misses (>= 200 cp); (2) nodes, at equal budgets, plus extensions per search and
  wall-clock ratio; (3) trigger accuracy (precision/recall against the ground-truth labels);
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

[FILL: `inst_c2_nowin` (window gate), `inst_c2_nocheck` (check rule), `inst_gain200` (first hypothesis).]

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
