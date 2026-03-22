import logging
import random
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SolverConfig
from .replay_buffer import ReplayBuffer
from .a2c_trainer import TORCH_AVAILABLE, TorchHRM, A2CTrainer
from .reflexion_engine import ReflexionEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Puzzle constants
# ---------------------------------------------------------------------------
GOAL_STATE = np.array(
    [1, 2, 3, 4,
     5, 6, 7, 8,
     9, 10, 11, 12,
     13, 14, 15, 0],
    dtype=np.int32,
)
GOAL_POS: Dict[int, int] = {v: i for i, v in enumerate(GOAL_STATE)}
ACTION_DIRS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # up
    1: ( 1, 0),  # down
    2: ( 0,-1),  # left
    3: ( 0, 1),  # right
}


# ---------------------------------------------------------------------------
# Puzzle utilities
# ---------------------------------------------------------------------------

def is_goal(state: np.ndarray) -> bool:
    return np.array_equal(state, GOAL_STATE)


def blank_index(state: np.ndarray) -> int:
    return int(np.where(state == 0)[0][0])


def apply_move(state: np.ndarray, action: int) -> np.ndarray:
    s = state.copy()
    b = blank_index(s)
    r, c = divmod(b, 4)
    dr, dc = ACTION_DIRS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < 4 and 0 <= nc < 4:
        ni = nr * 4 + nc
        s[b], s[ni] = s[ni], s[b]
    return s


def legal_moves(state: np.ndarray) -> List[int]:
    return [a for a in range(4) if not np.array_equal(apply_move(state, a), state)]


def heuristic(state: np.ndarray) -> int:
    """Manhattan distance + linear conflict — admissible lower bound on moves."""
    d = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        r1, c1 = divmod(i, 4)
        r2, c2 = divmod(GOAL_POS[v], 4)
        d += abs(r1 - r2) + abs(c1 - c2)
    return d + 2 * (_row_conflicts(state) + _col_conflicts(state))


def _row_conflicts(state: np.ndarray) -> int:
    c = 0
    for row in range(4):
        tiles = [
            state[row * 4 + col]
            for col in range(4)
            if state[row * 4 + col] != 0 and GOAL_POS[state[row * 4 + col]] // 4 == row
        ]
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if GOAL_POS[tiles[i]] % 4 > GOAL_POS[tiles[j]] % 4:
                    c += 1
    return c


def _col_conflicts(state: np.ndarray) -> int:
    c = 0
    for col in range(4):
        tiles = [
            state[row * 4 + col]
            for row in range(4)
            if state[row * 4 + col] != 0 and GOAL_POS[state[row * 4 + col]] % 4 == col
        ]
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if GOAL_POS[tiles[i]] // 4 > GOAL_POS[tiles[j]] // 4:
                    c += 1
    return c


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    def __init__(
        self,
        state: np.ndarray,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.unexplored = legal_moves(state)

    def Q(self) -> float:
        return 0.0 if self.visits == 0 else (self.value_sum / self.visits)

    def U(self, c_puct: float) -> float:
        if self.parent is None or self.prior == 0.0:
            return 0.0
        return c_puct * self.prior * (sqrt(self.parent.visits) / (1 + self.visits))

    def best_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda ch: ch.Q() + ch.U(c_puct))


class PUCTMCTS:
    def __init__(
        self,
        predictor,
        rb,
        config: SolverConfig,
        reflexion=None,
    ):
        self.sims = config.sims
        self.c_puct = config.c_puct
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_epsilon = config.dirichlet_epsilon
        self.discount_gamma = config.discount_gamma
        self.prior_floor = config.prior_floor
        self.priority_floor = config.priority_floor
        self.predictor = predictor
        self.rb = rb
        self.reflexion = reflexion

    def search(self, root_state: np.ndarray) -> Tuple[int, List[float], float]:
        root = MCTSNode(root_state)
        root_priors, _ = self.predictor.predict(root_state)

        legal = legal_moves(root_state)
        if legal:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal)).tolist()
            rp = {a: root_priors.get(a, 0.0) for a in legal}
            rp = {
                a: (1 - self.dirichlet_epsilon) * rp[a] + self.dirichlet_epsilon * n
                for a, n in zip(legal, noise)
            }
            total = sum(rp.values())
            if total > 0:
                rp = {a: v / total for a, v in rp.items()}
            root_priors = rp

        for _ in range(self.sims):
            node = root
            while not node.unexplored and node.children:
                node = node.best_child(self.c_puct)

            if node.unexplored:
                a = node.unexplored.pop(0)
                next_state = apply_move(node.state, a)
                parent_priors, _ = self.predictor.predict(node.state)
                if self.reflexion is not None:
                    parent_priors = self.reflexion.bias_priors(node.state, [], parent_priors)
                child_prior = max(parent_priors.get(a, 0.0), self.prior_floor)
                child = MCTSNode(next_state, parent=node, action=a, prior=child_prior)
                node.children[a] = child
                node = child

            V_leaf = 1.0 if is_goal(node.state) else self.predictor.predict(node.state)[1]

            depth = 0
            backup = node
            while backup is not None:
                backup.visits += 1
                backup.value_sum += (self.discount_gamma ** depth) * V_leaf
                backup = backup.parent
                depth += 1

        if not root.children:
            # No simulations could expand the root (terminal or fully pruned).
            acts = legal_moves(root_state)
            best = acts[0] if acts else 0
            uniform = 1.0 / max(len(acts), 1)
            pi = [uniform if a in acts else 0.0 for a in range(4)]
            logger.debug("MCTS search: no children at root — returning heuristic fallback.")
            return best, pi, 0.0

        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        visit_counts = [
            root.children[a].visits if a in root.children else 0 for a in range(4)
        ]
        total_visits = sum(visit_counts)
        pi_mcts = [c / total_visits if total_visits > 0 else 0.0 for c in visit_counts]
        v_mcts = root.Q()

        self.rb.update_target(tuple(root_state.tolist()), pi_mcts, v_mcts)
        if self.rb.replay_buffer is not None:
            priority = abs(v_mcts) + self.priority_floor
            self.rb.replay_buffer.add(root_state.copy(), pi_mcts, v_mcts, priority=priority)

        return best_action, pi_mcts, v_mcts


# ---------------------------------------------------------------------------
# Heuristic-only predictor (no torch dependency)
# ---------------------------------------------------------------------------

class HRMPredictorNP:
    """Pure-NumPy fallback predictor using the Manhattan + linear-conflict
    heuristic.  Used automatically when PyTorch is unavailable."""

    def __init__(self, heuristic_scale: float = 60.0, prior_floor: float = 1e-6):
        self.heuristic_scale = heuristic_scale
        self.prior_floor = prior_floor

    def predict(self, state: np.ndarray) -> Tuple[Dict[int, float], float]:
        acts = legal_moves(state)
        h = heuristic(state)
        V = 1.0 if is_goal(state) else -h / self.heuristic_scale

        scores = np.array(
            [-float(heuristic(apply_move(state, a))) for a in acts],
            dtype=np.float64,
        )
        scores -= scores.max()
        exps = np.exp(scores)
        total = exps.sum()
        probs = (exps / total).tolist() if total > 0 else [1.0 / len(acts)] * len(acts)
        probs = [p + self.prior_floor for p in probs]
        z = sum(probs)
        probs = [p / z for p in probs]

        return {a: p for a, p in zip(acts, probs)}, V


# ---------------------------------------------------------------------------
# Reasoning bank (replay buffer + target store)
# ---------------------------------------------------------------------------

class ReasoningBank:
    def __init__(self, config: SolverConfig):
        self.state_targets: Dict[Tuple, Tuple[List[float], float]] = {}
        self.reflexion_traces: List[Dict] = []
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            prioritized=True,
            alpha=config.buffer_alpha,
        )

    def get_target(self, state_tuple: Tuple) -> Tuple[List[float], float]:
        """Return stored (pi, v) target or a zero-initialised default."""
        return self.state_targets.get(state_tuple, ([0.0] * 4, 0.0))

    def update_target(self, state_tuple: Tuple, pi_mcts: List[float], v_mcts: float):
        self.state_targets[state_tuple] = (pi_mcts, v_mcts)

    def store_reflexion(self, trace: List[int], reason: str, correction):
        self.reflexion_traces.append(
            {"trace": trace, "reason": reason, "correction": correction}
        )


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

class Solver:
    """15-puzzle solver combining PUCT-MCTS, A2C training, and reflexion.

    Parameters
    ----------
    config:
        Hyperparameter bundle.  Defaults to :class:`SolverConfig` with all
        default values when not provided.
    """

    def __init__(self, config: SolverConfig = None):
        self.cfg = config or SolverConfig()
        self.rb = ReasoningBank(self.cfg)

        if TORCH_AVAILABLE:
            import torch
            if self.cfg.device == "auto":
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                dev = torch.device(self.cfg.device)

            self.hrm = TorchHRM(
                hidden=self.cfg.hrm_hidden,
                state_norm=self.cfg.state_norm,
            ).to(dev)
            self.trainer = A2CTrainer(
                self.hrm,
                lr=self.cfg.lr,
                entropy_coef=self.cfg.entropy_coef,
                value_coef=self.cfg.value_coef,
                max_grad_norm=self.cfg.max_grad_norm,
                device=dev,
            )
            predictor = self.hrm
            logger.info("Solver using TorchHRM on device %s.", dev)
        else:
            self.hrm = HRMPredictorNP(
                heuristic_scale=self.cfg.heuristic_scale,
                prior_floor=self.cfg.prior_floor,
            )
            self.trainer = None
            predictor = self.hrm
            logger.info("Solver using heuristic-only predictor (torch unavailable).")

        self.reflexion = ReflexionEngine(
            rb=self.rb,
            add_to_buffer_fn=lambda s, pi, v, priority=None: self.rb.replay_buffer.add(
                s, pi, v, priority
            ),
            config=self.cfg,
        )
        self.rb.reflexion_engine = self.reflexion

        self.mcts = PUCTMCTS(
            predictor=predictor,
            rb=self.rb,
            config=self.cfg,
            reflexion=self.reflexion,
        )

    def step(self, state: np.ndarray) -> int:
        action, _, _ = self.mcts.search(state)
        if self.trainer is not None:
            result = self.trainer.train_step_from_buffer(
                self.rb, batch_size=self.cfg.train_batch_size
            )
            if not result.get("skipped") and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "train step — loss=%.4f policy=%.4f value=%.4f entropy=%.4f",
                    result["loss"],
                    result["policy_loss"],
                    result["value_loss"],
                    result["entropy"],
                )
        return action

    def run_reflexion_analysis(
        self,
        trace: List[int],
        heuristics: List[int],
        final_state: np.ndarray,
    ) -> None:
        self.reflexion.reflect_and_patch(
            final_state=final_state, trace=trace, heuristics=heuristics
        )

    def solve(
        self, start: np.ndarray, max_steps: int = 200
    ) -> Tuple[bool, int, List[int]]:
        """Run the solver from *start* for up to *max_steps*.

        Returns ``(solved, steps_taken, heuristics_per_step)``.
        """
        state = start.copy()
        move_trace: List[int] = []
        path_h: List[int] = []

        for t in range(max_steps):
            if is_goal(state):
                self.run_reflexion_analysis(move_trace, path_h, state)
                return True, t, path_h
            action = self.step(state)
            state = apply_move(state, action)
            move_trace.append(action)
            path_h.append(heuristic(state))

        self.run_reflexion_analysis(move_trace, path_h, state)
        return False, max_steps, path_h


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def random_scramble(depth: int = 20, seed: int = None) -> np.ndarray:
    """Generate a random reachable puzzle state by applying *depth* random
    moves from the goal, avoiding immediate back-tracking."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    state = GOAL_STATE.copy()
    last_a: Optional[int] = None
    inv_action = {0: 1, 1: 0, 2: 3, 3: 2}
    for _ in range(depth):
        moves = legal_moves(state)
        if last_a is not None:
            inv = inv_action[last_a]
            if len(moves) > 1 and inv in moves:
                moves.remove(inv)
        a = random.choice(moves)
        state = apply_move(state, a)
        last_a = a
    return state
