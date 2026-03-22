import logging
import random
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for a sampled mini-batch.
# (states, pi_targets, v_targets, indices, importance_sampling_weights)
SampleBatch = Tuple[
    np.ndarray,          # states       (B, 16) int32
    np.ndarray,          # pi_targets   (B, 4)  float32
    np.ndarray,          # v_targets    (B,)    float32
    np.ndarray,          # indices      (B,)    int
    Optional[np.ndarray],# IS weights   (B,)    float32 or None
]


class ReplayBuffer:
    """Circular replay buffer with optional prioritised sampling.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.  Oldest transitions are
        overwritten once the buffer is full.
    prioritized:
        If ``True``, transitions are sampled proportionally to their stored
        priority raised to the power *alpha*.
    alpha:
        Priority exponent (0 = uniform sampling, 1 = fully prioritised).
    """

    def __init__(
        self,
        capacity: int = 100_000,
        prioritized: bool = False,
        alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha

        self.states: List[np.ndarray] = []
        self.pis: List[np.ndarray] = []
        self.vals: List[float] = []
        self.priorities: List[float] = []

        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: np.ndarray,
        pi: List[float],
        v: float,
        priority: Optional[float] = None,
    ) -> None:
        """Add a single transition.  Overwrites the oldest entry when full."""
        p = float(priority) if priority is not None else 1.0
        entry = (state.astype(np.int32), np.array(pi, dtype=np.float32), float(v), p)

        if not self._full:
            self.states.append(entry[0])
            self.pis.append(entry[1])
            self.vals.append(entry[2])
            self.priorities.append(entry[3])
            if len(self.states) >= self.capacity:
                self._full = True
                self._ptr = 0
        else:
            idx = self._ptr
            self.states[idx] = entry[0]
            self.pis[idx] = entry[1]
            self.vals[idx] = entry[2]
            self.priorities[idx] = entry[3]
            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[SampleBatch]:
        """Sample a mini-batch of *batch_size* transitions.

        Returns ``None`` when the buffer is empty so callers can gate
        training steps with a simple ``if sample is None`` check.
        """
        n = len(self.states)
        if n == 0:
            logger.debug("ReplayBuffer.sample: buffer is empty.")
            return None

        effective_size = min(batch_size, n)

        if not self.prioritized:
            idxs = np.array(random.sample(range(n), effective_size), dtype=np.intp)
            isw: Optional[np.ndarray] = None
        else:
            pr = np.maximum(np.array(self.priorities[:n], dtype=np.float64), 1e-6)
            weights = pr ** self.alpha
            probs = weights / weights.sum()
            idxs = np.random.choice(n, size=effective_size, replace=False, p=probs)
            isw = (1.0 / (n * probs[idxs])).astype(np.float32)
            isw = isw / isw.max()

        S = np.stack([self.states[i] for i in idxs], axis=0)
        PI = np.stack([self.pis[i] for i in idxs], axis=0)
        V = np.array([self.vals[i] for i in idxs], dtype=np.float32)
        return S, PI, V, idxs, isw

    def update_priorities(self, idxs, new_priorities: List[float]) -> None:
        """Update stored priorities for a batch of transitions (PER only)."""
        if not self.prioritized:
            return
        for i, p in zip(idxs, new_priorities):
            self.priorities[int(i)] = float(max(p, 1e-6))
