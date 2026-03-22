import logging
from typing import Callable, Dict, List

import numpy as np

from .config import SolverConfig

logger = logging.getLogger(__name__)

# Inverse-action map for the 15-puzzle: up↔down, left↔right
_INVERSE_ACTION: Dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2}
_NUM_ACTIONS = 4


class PlaybookRule:
    """A reactive rule that biases MCTS priors while it has remaining TTL."""

    def __init__(
        self,
        name: str,
        matcher: Callable[[np.ndarray, List[int]], bool],
        bias: Callable[[np.ndarray, List[int]], List[float]],
        ttl_steps: int = 500,
    ):
        self.name = name
        self.matcher = matcher
        self.bias = bias
        self.ttl_steps = ttl_steps

    def apply(self, state: np.ndarray, trace: List[int]) -> List[float]:
        return self.bias(state, trace)


class ReflexionEngine:
    """Post-episode analysis that injects corrective rules and replay data."""

    def __init__(self, rb, add_to_buffer_fn: Callable, config: SolverConfig = None):
        self.rb = rb
        self.add_to_buffer = add_to_buffer_fn
        self.rules: List[PlaybookRule] = []
        self.cfg = config or SolverConfig()

    # ------------------------------------------------------------------
    # Failure-mode detectors
    # ------------------------------------------------------------------

    def detect_oscillation(self, trace: List[int], window: int = None) -> bool:
        """Return True when the last *window* actions form a 2-cycle."""
        w = window if window is not None else self.cfg.oscillation_window
        if len(trace) < w:
            return False
        segment = trace[-w:]
        return all(segment[i] == segment[i % 2] for i in range(len(segment)))

    def detect_plateau(self, heuristics: List[int], k: int = None) -> bool:
        """Return True when the heuristic has not improved by more than 1 in
        the last *k* steps (agent is stuck in a local minimum)."""
        k = k if k is not None else self.cfg.plateau_window
        if len(heuristics) < k:
            return False
        window = heuristics[-k:]
        return (max(window) - min(window)) <= 1

    def detect_corner_deadlock(self, state: np.ndarray) -> bool:
        """Return True when the bottom row contains the deadlock tile set
        {13, 14, 15, 0} (blank in bottom row — hard to escape with standard
        moves)."""
        bottom = state[12:].tolist()
        return set(bottom) == {13, 14, 15, 0}

    # ------------------------------------------------------------------
    # Rule factories
    # ------------------------------------------------------------------

    def make_oscillation_rule(self) -> PlaybookRule:
        """Suppress the back-tracking action that causes a 2-cycle."""
        inv_bias = self.cfg.osc_inverse_bias
        window = self.cfg.oscillation_window
        ttl = self.cfg.ttl_break_cycle

        def matcher(state: np.ndarray, trace: List[int]) -> bool:
            return self.detect_oscillation(trace, window=window)

        def bias(state: np.ndarray, trace: List[int]) -> List[float]:
            if not trace:
                return [1.0] * _NUM_ACTIONS
            b = [1.0] * _NUM_ACTIONS
            b[_INVERSE_ACTION[trace[-1]]] = inv_bias
            return b

        return PlaybookRule("break_2cycle", matcher, bias, ttl_steps=ttl)

    def make_plateau_rule(self) -> PlaybookRule:
        """Boost up/down moves to escape a heuristic plateau.

        The rule is only *created* when a plateau is detected, so the matcher
        returns ``True`` unconditionally — the TTL governs how long it stays
        active.
        """
        up_down_bias = self.cfg.shake_bias_updown
        left_right_bias = self.cfg.shake_bias_leftright
        ttl = self.cfg.ttl_shake_plateau

        def matcher(state: np.ndarray, trace: List[int]) -> bool:
            # Rule lifetime is managed by TTL; always apply while active.
            return True

        def bias(state: np.ndarray, trace: List[int]) -> List[float]:
            # Actions: 0=up, 1=down, 2=left, 3=right
            return [up_down_bias, up_down_bias, left_right_bias, left_right_bias]

        return PlaybookRule("shake_plateau", matcher, bias, ttl_steps=ttl)

    def make_bottom_right_rule(self) -> PlaybookRule:
        """Apply directional bias to unjam the bottom-right corner deadlock."""
        corner_bias = list(self.cfg.corner_bias)  # copy to avoid shared mutation
        ttl = self.cfg.ttl_unjam_corner

        def matcher(state: np.ndarray, trace: List[int]) -> bool:
            return self.detect_corner_deadlock(state)

        def bias(state: np.ndarray, trace: List[int]) -> List[float]:
            return corner_bias

        return PlaybookRule("unjam_bottom_right", matcher, bias, ttl_steps=ttl)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def reflect_and_patch(
        self,
        final_state: np.ndarray,
        trace: List[int],
        heuristics: List[int],
    ) -> None:
        """Analyse a completed episode and inject corrective rules + replay
        transitions."""
        new_rules = []
        if self.detect_oscillation(trace):
            new_rules.append(self.make_oscillation_rule())
            logger.debug("ReflexionEngine: oscillation detected — added break_2cycle rule.")
        if self.detect_plateau(heuristics):
            new_rules.append(self.make_plateau_rule())
            logger.debug("ReflexionEngine: plateau detected — added shake_plateau rule.")
        if self.detect_corner_deadlock(final_state):
            new_rules.append(self.make_bottom_right_rule())
            logger.debug("ReflexionEngine: corner deadlock detected — added unjam_bottom_right rule.")

        self.rules.extend(new_rules)

        # Patch replay buffer with corrected transitions from the tail of the
        # episode so the network learns to avoid the failure mode.
        if (
            self.rb is not None
            and hasattr(self.rb, "replay_buffer")
            and self.rb.replay_buffer is not None
        ):
            window = self.cfg.reflexion_patch_window
            inv_bias = self.cfg.reflexion_inv_bias
            heuristic_scale = self.cfg.heuristic_scale
            num_actions = _NUM_ACTIONS

            for i in range(max(0, len(trace) - window), len(trace)):
                uniform = 1.0 / num_actions
                pi = [uniform] * num_actions
                if i > 0:
                    inv = _INVERSE_ACTION[trace[i - 1]]
                    pi[inv] *= inv_bias
                total = sum(pi)
                pi = [p / total for p in pi]
                v = -heuristics[i] / heuristic_scale
                self.add_to_buffer(
                    final_state.copy(),
                    pi,
                    v,
                    priority=abs(v) + self.cfg.priority_floor,
                )

    def bias_priors(
        self,
        state: np.ndarray,
        trace: List[int],
        priors: Dict[int, float],
    ) -> Dict[int, float]:
        """Apply all active rules to *priors* and return the biased distribution."""
        if not self.rules:
            return priors

        mult = [1.0] * _NUM_ACTIONS
        still_active = []
        for rule in self.rules:
            if rule.ttl_steps <= 0:
                continue
            if rule.matcher(state, trace):
                b = rule.apply(state, trace)
                mult = [m * b[i] for i, m in enumerate(mult)]
                rule.ttl_steps -= 1
                still_active.append(rule)
        self.rules = still_active

        out = {a: priors.get(a, 0.0) * mult[a] for a in range(_NUM_ACTIONS)}
        total = sum(out.values())
        if total > 0:
            out = {a: out[a] / total for a in out}
        return out
