"""
Search-extension controllers.

A controller decides where the search spends extra effort. The search
(engine/search.py) asks it two questions:

  extension(...)        for each move about to be played: add plies to the child?
  frontier_checks(...)  at each depth-0 node, AFTER plain quiescence has run:
                        should the quiet checking moves also be searched?

Engines (names for run_baseline.py --engines):

  NoExtension      baseline: nothing extra.
  CheckExtension   classic fixed rule, +1 ply when the move gives check. Any real
                   engine has this, so it is the honest minimum (beating plain
                   alpha-beta alone would be a strawman).
  BlindChecks      "qchk_all": the check pass at EVERY frontier node. No logic;
                   it is the brute-force comparator the selective controller
                   has to beat on cost-effectiveness, not a design to keep.
  RWChecks         "rw_*": searches a quiet checking move at a frontier node only
                   if its expected value, inside the alpha-beta window, beats its
                   price in nodes. See below.

RWChecks: what is taken from Russell & Wefald (1989), and what is not
---------------------------------------------------------------------
R&W: a search step is worth taking if the expected improvement in the decision
it can cause exceeds the cost of computing, and they choose among individual
leaf expansions. We use that decision rule for one step, "search this quiet
checking move at a frontier node", with:

  expected gain  = E[change of the window-clipped value | step]   (error model,
                   engine/error_model.py, learned offline from sampled frontier
                   nodes like R&W's error distributions, bucketed by how forcing
                   the check is: the number of replies the opponent has)
  price          = lambda * measured mean nodes of the step * budget pressure
  rule           = take the step iff gain >= price; best (most forcing) first

Two further R&W ideas appear in exact form:
  * Relevance. Extra moves can only RAISE the value of the side to move, so a
    node whose plain value already reaches beta cannot change (search.py skips
    it), and movement outside (alpha, beta) cannot change the search result;
    the clipped-gain formula gives both for free. This replaces the old
    hand-set "margin" around the window.
  * Time cost grows as time runs out. R&W set the cost as an inverse function
    of the time remaining per move; here, of the node budget remaining.

Single-step assumption (R&W's own): each check is valued as if it were the only
one searched; interactions between checks are ignored.

Not taken: MGSS* itself (best-first, whole tree in memory, normal error curves
from 35,000 samples) does not fit a depth-first alpha-beta engine, and its
single-step assumption is known to fail as trees grow. Only the idea is used.

Transposition-table safety: the decision is made at depth-0 nodes, which are
never stored in the TT, and uses only the position, alpha and beta. A
window-dependent decision therefore cannot change what the table stores for
any node (the path-dependence risk that worried the plan, loophole 7). The
check pass can still change the VALUE a depth-1 parent stores; that is the same
trade-off any quiescence extension makes.
"""

from functools import lru_cache
from typing import Callable, Dict, List

import chess
import chess.polyglot

from .error_model import ErrorModel, DEFAULT_PATH
from .eval import evaluate

CHECK_MAX_EXT = 4      # cap on plies one line may gain from the check rule
INF = 10 ** 9
MAX_PRESSURE = 20.0    # budget pressure never prices a node above 20x


def quiet_checks(board: chess.Board) -> List[chess.Move]:
    """Checking moves that are not captures (captures are already searched by quiescence)."""
    return [m for m in board.legal_moves
            if not board.is_capture(m) and board.gives_check(m)]


def _static_stm(board: chess.Board) -> int:
    v = evaluate(board, terminal_checks=False)
    return -v if board.turn == chess.BLACK else v


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

    def frontier_checks(self, board: chess.Board, q: int, alpha: int, beta: int,
                        stats) -> List[chess.Move]:
        """Quiet checking moves to search at this frontier node (empty = none).
        `q` is the plain quiescence value already computed (side-to-move relative)."""
        return []


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


class BlindChecks(CheckExtension):
    """Check rule + the check pass at every frontier node (no decision at all)."""
    name = "qchk_all"

    def frontier_checks(self, board, q, alpha, beta, stats):
        return quiet_checks(board)


class RandomChecks(CheckExtension):
    """CONTROL, not a design: searches each quiet check with fixed probability p
    (deterministic: decided by a hash of the position and move). Compare with
    RWChecks at the same number of check searches to see whether its selection
    carries any information beyond "do less of what qchk_all does"."""
    name = "rand"

    def __init__(self, p: float = 0.33):
        super().__init__()
        self.p = p

    def frontier_checks(self, board, q, alpha, beta, stats):
        key = chess.polyglot.zobrist_hash(board)
        return [m for m in quiet_checks(board)
                if ((key ^ (m.from_square * 64 + m.to_square) * 0x9E3779B97F4A7C15) >> 7) % 100 < self.p * 100]


class RWChecks(CheckExtension):
    """
    Check rule + search each quiet check whose  expected gain >= lambda * cost * pressure.

    lam           centipawns of expected value a check must promise per node of
                  search it costs. Tuned on the TRAIN split.
    budget_aware  False = ablation: constant price, ignoring the remaining budget.
    use_window    False = ablation: pretend the window is infinite, i.e. drop the
                  relevance test and count any value change as useful.
    use_check     False = ablation: no +1-ply check rule in the main search.
    """
    name = "rw"

    def __init__(self, model: ErrorModel, lam: float = 1.0, budget_aware: bool = True,
                 use_window: bool = True, use_check: bool = True):
        super().__init__()
        self.model, self.lam = model, lam
        self.budget_aware, self.use_window, self.use_check = budget_aware, use_window, use_check

    def extension(self, board, move, depth_after, ply, alpha, beta, stats):
        if not self.use_check:
            return 0
        return super().extension(board, move, depth_after, ply, alpha, beta, stats)

    @staticmethod
    def pressure(stats) -> float:
        """Price multiplier from the budget left: 1 with a full budget, rising as
        it is used up (R&W: time cost is an inverse function of time remaining)."""
        if stats.node_limit is None:
            return 1.0
        remaining = 1.0 - stats.nodes / stats.node_limit
        return MAX_PRESSURE if remaining <= 1.0 / MAX_PRESSURE else 1.0 / remaining

    def frontier_checks(self, board, q, alpha, beta, stats):
        checks = quiet_checks(board)
        if not checks:
            return []
        # q <= alpha means plain quiescence only told us "at most alpha"; its
        # stand-pat floor (static value) is then the safe (low) estimate.
        base = q if q > alpha else min(_static_stm(board), alpha)
        a, b = (alpha, beta) if self.use_window else (-INF, INF)
        rate = self.lam * (self.pressure(stats) if self.budget_aware else 1.0)
        chosen = []
        for m in checks:
            board.push(m)
            replies = board.legal_moves.count()
            board.pop()
            if self.model.expected_gain(replies, base, a, b) >= rate * self.model.cost(replies):
                chosen.append((replies, m))
        chosen.sort(key=lambda t: t[0])         # most forcing first: raises alpha early
        return [m for _, m in chosen]


@lru_cache(maxsize=None)
def _cached_model(path: str) -> ErrorModel:
    return ErrorModel.load(path)


def controller_factories(model_path: str = DEFAULT_PATH) -> Dict[str, Callable[[], Controller]]:
    """name -> zero-argument factory. The harness turns each entry into an engine
    version, so one line here makes a variant runnable from run_baseline.py,
    compare_engines.py and match.py."""
    m = lambda: _cached_model(model_path)
    f: Dict[str, Callable[[], Controller]] = {
        "check":    lambda: CheckExtension(),
        "qchk_all": lambda: BlindChecks(),
    }
    for lam in (0.1, 0.25, 0.5, 1, 2):
        f[f"rw_l{lam}"] = (lambda lam=lam: RWChecks(m(), lam=lam))
    f["rand_p33"] = lambda: RandomChecks(0.33)   # control for rw_*: same pass rate, no logic
    # ablations of one setting (re-point them at the tuned lambda once chosen)
    f["rw_l0.1_nobudget"] = lambda: RWChecks(m(), lam=0.1, budget_aware=False)
    f["rw_l0.1_nowindow"] = lambda: RWChecks(m(), lam=0.1, use_window=False)
    f["rw_l0.1_nocheck"]  = lambda: RWChecks(m(), lam=0.1, use_check=False)
    return f
