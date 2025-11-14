
import numpy as np
import random
from typing import List, Tuple, Optional

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.pis:    List[np.ndarray] = []
        self.vals:   List[float] = []
        self.prioritized = prioritized
        self.alpha = alpha
        self.priorities: List[float] = []

        self.ptr = 0
        self.full = False

    def __len__(self):
        return len(self.states)

    def add(self, state: np.ndarray, pi: List[float], v: float, priority: Optional[float] = None):
        if not self.full:
            self.states.append(state.astype(np.int32))
            self.pis.append(np.array(pi, dtype=np.float32))
            self.vals.append(float(v))
            self.priorities.append(float(priority) if priority is not None else 1.0)
            if len(self.states) >= self.capacity:
                self.full = True
                self.ptr = 0
        else:
            self.states[self.ptr] = state.astype(np.int32)
            self.pis[self.ptr]    = np.array(pi, dtype=np.float32)
            self.vals[self.ptr]   = float(v)
            self.priorities[self.ptr] = float(priority) if priority is not None else 1.0
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.states)
        if n == 0:
            return None

        if not self.prioritized:
            idxs = random.sample(range(n), min(batch_size, n))
            probs = None
            isw = None
        else:
            pr = np.array(self.priorities[:n], dtype=np.float64)
            pr = np.maximum(pr, 1e-6)
            weights = pr ** self.alpha
            probs = weights / weights.sum()
            idxs = np.random.choice(n, size=min(batch_size, n), replace=False, p=probs)
            isw = (1.0 / (n * probs[idxs])).astype(np.float32)
            isw = isw / isw.max()

        S  = np.stack([self.states[i] for i in idxs], axis=0)
        PI = np.stack([self.pis[i]    for i in idxs], axis=0)
        V  = np.array([self.vals[i]   for i in idxs], dtype=np.float32)
        return (S, PI, V, idxs, (isw if self.prioritized else None))

    def update_priorities(self, idxs, new_priorities: List[float]):
        if not self.prioritized:
            return
        for i, p in zip(idxs, new_priorities):
            self.priorities[i] = float(max(p, 1e-6))
