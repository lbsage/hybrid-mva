
import numpy as np
import random
from math import sqrt
from typing import List, Tuple, Dict

from .replay_buffer import ReplayBuffer
from .a2c_trainer_stub import TORCH_AVAILABLE, TorchHRM, A2CTrainer
from .reflexion_engine import ReflexionEngine

GOAL_STATE = np.array([1, 2, 3, 4,
                       5, 6, 7, 8,
                       9,10,11,12,
                       13,14,15,0], dtype=np.int32)
GOAL_POS: Dict[int,int] = {v:i for i,v in enumerate(GOAL_STATE)}
ACTION_DIRS = {0: (-1, 0), 1: ( 1, 0), 2: ( 0,-1), 3: ( 0, 1)}

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
        ni = nr*4 + nc
        s[b], s[ni] = s[ni], s[b]
    return s

def legal_moves(state: np.ndarray) -> List[int]:
    return [a for a in range(4) if not np.array_equal(apply_move(state, a), state)]

def _row_conflicts(state: np.ndarray) -> int:
    c = 0
    for row in range(4):
        tiles = []
        for col in range(4):
            v = state[row*4 + col]
            if v == 0: continue
            if (GOAL_POS[v] // 4) == row:
                tiles.append(v)
        for i in range(len(tiles)):
            g1 = GOAL_POS[tiles[i]] % 4
            for j in range(i+1, len(tiles)):
                g2 = GOAL_POS[tiles[j]] % 4
                if g1 > g2: c += 1
    return c

def _col_conflicts(state: np.ndarray) -> int:
    c = 0
    for col in range(4):
        tiles = []
        for row in range(4):
            v = state[row*4 + col]
            if v == 0: continue
            if (GOAL_POS[v] % 4) == col:
                tiles.append(v)
        for i in range(len(tiles)):
            g1 = GOAL_POS[tiles[i]] // 4
            for j in range(i+1, len(tiles)):
                g2 = GOAL_POS[tiles[j]] // 4
                if g1 > g2: c += 1
    return c

def heuristic(state: np.ndarray) -> int:
    d = 0
    for i, v in enumerate(state):
        if v == 0: continue
        r1, c1 = divmod(i, 4)
        r2, c2 = divmod(GOAL_POS[v], 4)
        d += abs(r1 - r2) + abs(c1 - c2)
    return d + 2*(_row_conflicts(state) + _col_conflicts(state))

class MCTSNode:
    def __init__(self, state: np.ndarray, parent=None, action=None, prior: float = 0.0):
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
    def __init__(self, predictor, rb, sims: int = 100, c_puct: float = 1.4, reflexion=None):
        self.sims = sims
        self.c_puct = c_puct
        self.predictor = predictor
        self.rb = rb
        self.reflexion = reflexion

    def search(self, root_state: np.ndarray):
        root = MCTSNode(root_state)
        root_priors, _ = self.predictor.predict(root_state)

        alpha = 0.3
        epsilon = 0.25
        legal = legal_moves(root_state)
        if legal:
            noise = np.random.dirichlet([alpha] * len(legal)).tolist()
            rp = {a: root_priors.get(a, 0.0) for a in legal}
            rp = {a: (1 - epsilon) * rp[a] + epsilon * n for a, n in zip(legal, noise)}
            z = sum(rp.values())
            if z > 0:
                rp = {a: v / z for a, v in rp.items()}
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
                child_prior = max(parent_priors.get(a, 0.0), 1e-6)
                child = MCTSNode(next_state, parent=node, action=a, prior=child_prior)
                node.children[a] = child
                node = child

            if is_goal(node.state):
                V_leaf = 1.0
            else:
                _, V_leaf = self.predictor.predict(node.state)

            depth = 0
            while node is not None:
                node.visits += 1
                node.value_sum += (0.99 ** depth) * V_leaf
                node = node.parent
                depth += 1

        if not root.children:
            return 0, [0.0]*4, 0.0
        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]

        visit_counts = [root.children[a].visits if a in root.children else 0 for a in range(4)]
        total_visits = sum(visit_counts)
        pi_mcts = [c / total_visits if total_visits > 0 else 0.0 for c in visit_counts]
        v_mcts = root.Q()

        self.rb.update_target(tuple(root_state.tolist()), pi_mcts, v_mcts)
        if self.rb.replay_buffer is not None:
            priority = abs(v_mcts) + 1e-3
            self.rb.replay_buffer.add(root_state.copy(), pi_mcts, v_mcts, priority=priority)

        return best_action, pi_mcts, v_mcts

class HRMPredictorNP:
    def predict(self, state: np.ndarray):
        acts = legal_moves(state)
        h = heuristic(state)
        V = -h/60.0 if not np.array_equal(state, GOAL_STATE) else 1.0
        scores = [-float(heuristic(apply_move(state, a))) for a in acts]
        xs = np.array(scores, dtype=np.float64)
        xs -= xs.max()
        exps = np.exp(xs)
        s = exps.sum()
        probs = (exps/s).tolist() if s > 0 else [1.0/len(acts)]*len(acts)
        eps = 1e-6
        probs = [p+eps for p in probs]
        z = sum(probs)
        probs = [p/z for p in probs]
        P = {a:p for a,p in zip(acts, probs)}
        return P, V

class ReasoningBank:
    def __init__(self):
        self.state_targets: Dict[Tuple, Tuple[List[float], float]] = {}
        self.reflexion_traces: List[Dict] = []
        self.replay_buffer = ReplayBuffer(capacity=200_000, prioritized=True, alpha=0.6)

    def get_target(self, state_tuple: Tuple):
        return self.state_targets.get(state_tuple, ([0.0]*4, 0.0))

    def update_target(self, state_tuple: Tuple, pi_mcts: List[float], v_mcts: float):
        self.state_targets[state_tuple] = (pi_mcts, v_mcts)

    def store_reflexion(self, trace, reason, correction):
        self.reflexion_traces.append({"trace": trace, "reason": reason, "correction": correction})

class Solver:
    def __init__(self, sims: int = 200, c_puct: float = 1.2, device: str = "auto"):
        self.rb = ReasoningBank()

        if TORCH_AVAILABLE:
            import torch
            dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
            self.hrm = TorchHRM(hidden=128).to(dev)
            self.trainer = A2CTrainer(self.hrm, lr=3e-4, entropy_coef=1e-3, value_coef=0.5,
                                      max_grad_norm=1.0, device=dev)
            predictor = self.hrm
        else:
            self.hrm = HRMPredictorNP()
            self.trainer = None
            predictor = self.hrm

        self.reflexion = ReflexionEngine(
            rb=self.rb,
            add_to_buffer_fn=lambda s, pi, v, priority=None: self.rb.replay_buffer.add(s, pi, v, priority)
        )
        self.rb.reflexion_engine = self.reflexion

        self.mcts = PUCTMCTS(predictor=predictor, rb=self.rb, sims=sims, c_puct=c_puct, reflexion=self.reflexion)

    def step(self, state: np.ndarray) -> int:
        action, _, _ = self.mcts.search(state)
        if self.trainer is not None:
            _ = self.trainer.train_step_from_buffer(self.rb, batch_size=128)
        return action

    def run_reflexion_analysis(self, trace: List[int], heuristics: List[int], final_state: np.ndarray):
        self.reflexion.reflect_and_patch(final_state=final_state, trace=trace, heuristics=heuristics)

    def solve(self, start: np.ndarray, max_steps: int = 200) -> Tuple[bool, int, List[int]]:
        state = start.copy()
        move_trace = []
        path_h = []

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

def random_scramble(depth: int = 20, seed: int = None) -> np.ndarray:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    state = GOAL_STATE.copy()
    last_a = None
    for _ in range(depth):
        moves = legal_moves(state)
        if last_a is not None:
            inv = {0:1,1:0,2:3,3:2}[last_a]
            if len(moves) > 1 and inv in moves: moves.remove(inv)
        a = random.choice(moves)
        state = apply_move(state, a)
        last_a = a
    return state
