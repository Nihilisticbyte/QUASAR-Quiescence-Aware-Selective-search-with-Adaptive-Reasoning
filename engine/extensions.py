"""
Search-extension controllers (Phase 4).

A controller decides, for each move the search is about to play, whether the
child position deserves extra search depth. The search (engine/search.py)
calls

    controller.extension(board_after_move, move, depth_after, ply, alpha, beta, stats)

with the child position already on the board, and adds the returned number of
plies (0 or more) to that child's remaining depth.

Controllers, in increasing sophistication:

  NoExtension              does nothing (== the plain Phase 1 search)
  CheckExtension           classic fixed rule: +1 ply when the move gives check.
                           A real engine already has this, so it is the honest
                           baseline: beating plain alpha-beta alone would be a
                           strawman (project-plan.md, loophole 5).
  InstabilityExtension     check rule PLUS +1 ply at the search FRONTIER
                           (depth_after == 0) when the position is "unstable"
                           and its value could matter for the window.
  CostWeightedExtension    same gate, but the trigger weighs expected benefit
                           against cost:  P(change) * shift  >=  lambda * cost
  QuiescenceChecks         check rule PLUS, at triggered frontier nodes, a
                           quiescence that also tries the quiet CHECKING moves
                           (instead of a full-width extra ply). Same trigger,
                           much cheaper "extension": only the forcing moves that
                           captures-only quiescence cannot see are searched.

What counts as "unstable" was chosen from DATA, not guessed (Phase 3,
log_instability.py / analyze_instability.py). Our first hypothesis, "the side
to move has a big capture available", turned out to be anti-predictive: the
frontier hands over to quiescence, which already searches captures, so one
more ply rarely changes those nodes. What quiescence does NOT see is forcing
quiet moves, and the strongest predictor of "one more ply changes the value"
was the number of checking moves available (n_checks). The capture-gain rule
is kept as the ablation `inst_gain` so the report can show both.

The window gate: a node only matters for the minimax value if its value can
land inside the (alpha, beta) window; far outside it the parent cuts off no
matter what (project-plan.md, "Core research framing"). We approximate the
node's value by its static evaluation and require
    alpha - margin <= static <= beta + margin.

Table safety (loophole 7): every decision uses only the child POSITION, its
remaining depth and ply, and alpha/beta. The only path-dependent quantity is
how many plies the line has already been extended, and that equals
ply + depth - root_depth (a function of the node's own ply and depth). The TT
stores the depth actually searched and an entry is reused only if it was
searched at least as deep as needed. Two paths reaching the same position with
different windows can still extend differently; that is the same trade-off real
engines accept, and it is listed as a limitation in the report.
"""

import json
import os
from functools import lru_cache
from typing import Callable, Dict, List

import chess

from .eval import evaluate
from .features import max_capture_gain, count_checks

DEFAULT_MODEL_PATH = os.path.join("data", "trigger_model.json")
CHECK_MAX_EXT = 4      # cap on plies one line may gain from the check rule


class Controller:
    """Base class: never extends. Also knows how many plies the current line
    has already been extended, which is how the per-path cap is enforced."""
    name = "none"

    def __init__(self):
        self.root_depth = 0

    def start_search(self, depth: int):
        """Called by search_root at the start of each iterative-deepening iteration."""
        self.root_depth = depth

    def path_extensions(self, depth_after: int, ply: int) -> int:
        """Plies already added on the line leading to this child. Without
        extensions ply + remaining depth always equals the root depth, so any
        excess is exactly the number of plies extended so far."""
        return ply + depth_after - self.root_depth

    def extension(self, board: chess.Board, move: chess.Move, depth_after: int,
                  ply: int, alpha: int, beta: int, stats) -> int:
        return 0

    def checks_at_frontier(self, board: chess.Board, alpha: int, beta: int) -> bool:
        """Called at every depth-0 node: should quiescence also try the quiet
        checking moves here? Default: no."""
        return False


class NoExtension(Controller):
    name = "none"


class CheckExtension(Controller):
    """+1 ply when the move just played gives check (the child is in check)."""
    name = "check"

    def __init__(self, max_ext: int = CHECK_MAX_EXT):
        super().__init__()
        self.max_ext = max_ext

    def extension(self, board, move, depth_after, ply, alpha, beta, stats):
        if board.is_check() and self.path_extensions(depth_after, ply) < self.max_ext:
            return 1
        return 0


def _static_stm(board: chess.Board) -> int:
    v = evaluate(board, terminal_checks=False)
    return -v if board.turn == chess.BLACK else v


class InstabilityExtension(Controller):
    """
    Check rule (optional) + a frontier extension when the position is unstable.

    The instability test, all enabled conditions must hold:
      min_checks  side to move has at least this many checking moves (0 = ignore)
      min_gain    side to move has a capture winning at least this many
                  centipawns on the first step (0 = ignore; the first
                  hypothesis, kept as an ablation)
    Then, unless use_window is False, the window gate applies.

    margin      slack (cp) around the (alpha, beta) window in the gate.
    max_ext     cap on plies one line may gain from the instability rule.
    use_window  False = ablation: extend on instability alone, ignoring alpha/beta.
    use_check   False = ablation: instability trigger only, no check rule.
    """
    name = "inst"

    def __init__(self, min_checks: int = 2, min_gain: int = 0, margin: int = 150,
                 max_ext: int = 2, use_window: bool = True, use_check: bool = True):
        super().__init__()
        if min_checks <= 0 and min_gain <= 0:
            raise ValueError("an instability trigger needs min_checks or min_gain > 0")
        self.min_checks = min_checks
        self.min_gain = min_gain
        self.margin = margin
        self.max_ext = max_ext
        self.use_window = use_window
        self.use_check = use_check

    def _window_relevant(self, board, alpha, beta) -> bool:
        static = _static_stm(board)
        return alpha - self.margin <= static <= beta + self.margin

    def _unstable(self, board, alpha, beta) -> bool:
        if self.min_gain > 0 and max_capture_gain(board) < self.min_gain:
            return False
        if self.min_checks > 0 and count_checks(board, self.min_checks) < self.min_checks:
            return False
        return (not self.use_window) or self._window_relevant(board, alpha, beta)

    def extension(self, board, move, depth_after, ply, alpha, beta, stats):
        used = self.path_extensions(depth_after, ply)
        if board.is_check():
            if self.use_check and used < CHECK_MAX_EXT:
                return 1
            return 0                       # in check: quiescence searches the evasions
        if depth_after == 0 and used < self.max_ext and self._unstable(board, alpha, beta):
            return 1
        return 0


class QuiescenceChecks(InstabilityExtension):
    """
    Check rule (optional) + "checks in quiescence" at unstable frontier nodes.

    min_checks = 0 means ALWAYS search checks at the frontier (no trigger): the
    unconditional version, used to ask whether the instability trigger earns its
    keep or whether always doing it is just as good.
    With min_checks >= 1 the trigger is the same as InstabilityExtension:
    at least that many checking moves AND the window gate.
    """
    name = "qchk"

    def __init__(self, min_checks: int = 2, margin: int = 150,
                 use_window: bool = True, use_check: bool = True):
        super().__init__(min_checks=max(min_checks, 1), margin=margin,
                         use_window=use_window, use_check=use_check)
        self.unconditional = min_checks <= 0

    def extension(self, board, move, depth_after, ply, alpha, beta, stats):
        # only the check rule extends depth; the instability part acts in quiescence
        if board.is_check() and self.use_check and \
                self.path_extensions(depth_after, ply) < CHECK_MAX_EXT:
            return 1
        return 0

    def checks_at_frontier(self, board, alpha, beta) -> bool:
        if board.is_check():
            return False              # quiescence already searches every evasion
        return self.unconditional or self._unstable(board, alpha, beta)


class TriggerModel:
    """
    P(value changes when searched one ply deeper | feature bucket), and the mean
    size of that change, estimated from data (log_instability.py +
    analyze_instability.py) and stored as JSON:

        {"feature": "n_checks", "edges": [0, 1, 2, 3, 5],
         "p": [...], "shift": [...], "n": [...], ...}

    Bucket i covers edges[i] <= value < edges[i+1]; the last bucket is open-ended.
    """

    def __init__(self, feature: str, edges: List[int], p: List[float], shift: List[float]):
        if not (len(edges) == len(p) == len(shift)) or not edges:
            raise ValueError("edges, p and shift must be non-empty and the same length")
        self.feature, self.edges, self.p, self.shift = feature, list(edges), list(p), list(shift)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "TriggerModel":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trigger model not found: {path}. Build it first:\n"
                "  python log_instability.py\n  python analyze_instability.py")
        with open(path) as f:
            d = json.load(f)
        return cls(d["feature"], d["edges"], d["p"], d["shift"])

    def _bucket(self, value: int) -> int:
        i = 0
        for k, e in enumerate(self.edges):
            if value >= e:
                i = k
        return i

    def p_change(self, value: int) -> float:
        return self.p[self._bucket(value)]

    def expected_shift(self, value: int) -> float:
        """P(change) * mean size of the change, in centipawns."""
        i = self._bucket(value)
        return self.p[i] * self.shift[i]


class CostWeightedExtension(InstabilityExtension):
    """
    Extend at the frontier when

        P(change | n_checks) * shift   >=   lam * cost,     cost = n_legal / 30

    i.e. the expected value shift an extension promises must beat lam times its
    estimated price. cost is the number of legal moves over 30 (one extra ply
    costs roughly in proportion to the branching factor; 30 is typical for a
    middlegame). `lam` is in centipawn units. Tune it on the TRAIN split
    (run_baseline.py --split train) and report on the test split.
    Needs data/trigger_model.json built on n_checks.
    """
    name = "costw"

    def __init__(self, model: TriggerModel, lam: float = 40.0, margin: int = 150,
                 max_ext: int = 2, use_check: bool = True):
        super().__init__(min_checks=1, margin=margin, max_ext=max_ext,
                         use_window=True, use_check=use_check)
        if model.feature != "n_checks":
            raise ValueError(f"CostWeightedExtension needs a model on n_checks, got {model.feature}")
        self.model = model
        self.lam = lam

    def _unstable(self, board, alpha, beta) -> bool:
        checks = count_checks(board)
        if checks < 1:
            return False
        cost = max(1, sum(1 for _ in board.legal_moves)) / 30.0
        if self.model.expected_shift(checks) < self.lam * cost:
            return False
        return self._window_relevant(board, alpha, beta)


# ---------------------------------------------------------------------------
# Registry: name -> zero-argument factory. The harness turns each entry into an
# engine version, so adding a variant here makes it runnable everywhere
# (run_baseline.py --engines ..., compare_engines.py, match.py).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cached_model(path: str) -> TriggerModel:
    return TriggerModel.load(path)


def controller_factories(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Callable[[], Controller]]:
    f: Dict[str, Callable[[], Controller]] = {
        "check":        lambda: CheckExtension(),
        "inst_c1":      lambda: InstabilityExtension(min_checks=1),
        "inst_c2":      lambda: InstabilityExtension(min_checks=2),
        "inst_c3":      lambda: InstabilityExtension(min_checks=3),
        # ablations of inst_c2:
        "inst_c2_nowin":   lambda: InstabilityExtension(min_checks=2, use_window=False),
        "inst_c2_nocheck": lambda: InstabilityExtension(min_checks=2, use_check=False),
        # the first hypothesis (capture gain), kept to report the negative result:
        "inst_gain200": lambda: InstabilityExtension(min_checks=0, min_gain=200),
    }
    f["qchk_all"] = lambda: QuiescenceChecks(min_checks=0)
    for k in (1, 2, 3):
        f[f"qchk_c{k}"] = (lambda k=k: QuiescenceChecks(min_checks=k))
    f["qchk_c2_nowin"] = lambda: QuiescenceChecks(min_checks=2, use_window=False)
    for lam in (10, 20, 40, 80):   # ~ shift of 1 check / 2+ checks / 5+ checks, see analyze_instability.py
        f[f"costw_l{lam}"] = (lambda lam=lam: CostWeightedExtension(
            _cached_model(model_path), lam=float(lam)))
    return f
