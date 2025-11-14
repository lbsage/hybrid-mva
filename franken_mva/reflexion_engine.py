
import numpy as np
from typing import List, Dict, Callable

class PlaybookRule:
    def __init__(self, name: str,
                 matcher: Callable[[np.ndarray, List[int]], bool],
                 bias: Callable[[np.ndarray, List[int]], List[float]],
                 ttl_steps: int = 500):
        self.name = name
        self.matcher = matcher
        self.bias = bias
        self.ttl_steps = ttl_steps

    def apply(self, state: np.ndarray, trace: List[int]) -> List[float]:
        return self.bias(state, trace)

class ReflexionEngine:
    def __init__(self, rb, add_to_buffer_fn):
        self.rb = rb
        self.add_to_buffer = add_to_buffer_fn
        self.rules: List[PlaybookRule] = []

    def detect_oscillation(self, trace: List[int], window: int = 6) -> bool:
        if len(trace) < window: return False
        segment = trace[-window:]
        return all(segment[i] == segment[i % 2] for i in range(len(segment)))

    def detect_plateau(self, heuristics: List[int], k: int = 10) -> bool:
        if len(heuristics) < k: return False
        window = heuristics[-k:]
        return (max(window) - min(window)) <= 1

    def detect_corner_deadlock(self, state: np.ndarray) -> bool:
        bottom = state[12:].tolist()
        return set(bottom) == set([13,14,15,0])

    def make_oscillation_rule(self, inv_map: Dict[int,int]):
        def matcher(state, trace):
            return self.detect_oscillation(trace, window=6)
        def bias(state, trace):
            if not trace: return [1,1,1,1]
            last = trace[-1]
            inv = inv_map[last]
            b = [1,1,1,1]
            b[inv] = 0.2
            return b
        return PlaybookRule("break_2cycle", matcher, bias, ttl_steps=800)

    def make_plateau_rule(self):
        def matcher(state, trace): return True
        def bias(state, trace):
            return [1.15, 1.15, 1.0, 1.0]
        return PlaybookRule("shake_plateau", matcher, bias, ttl_steps=400)

    def make_bottom_right_rule(self):
        def matcher(state, trace): return self.detect_corner_deadlock(state)
        def bias(state, trace):
            return [1.2, 0.9, 1.3, 0.8]
        return PlaybookRule("unjam_bottom_right", matcher, bias, ttl_steps=600)

    def reflect_and_patch(self, final_state: np.ndarray, trace: List[int], heuristics: List[int]):
        add_rules = []
        if self.detect_oscillation(trace): add_rules.append(self.make_oscillation_rule({0:1,1:0,2:3,3:2}))
        if self.detect_plateau(heuristics): add_rules.append(self.make_plateau_rule())
        if self.detect_corner_deadlock(final_state): add_rules.append(self.make_bottom_right_rule())

        for r in add_rules:
            self.rules.append(r)

        if self.rb is not None and hasattr(self.rb, "replay_buffer") and self.rb.replay_buffer is not None:
            for i in range(max(0, len(trace)-6), len(trace)):
                pi = [0.25,0.25,0.25,0.25]
                if i > 0:
                    inv = {0:1,1:0,2:3,3:2}[trace[i-1]]
                    pi[inv] *= 0.2
                z = sum(pi); pi = [p/z for p in pi]
                v = -heuristics[i] / 60.0
                self.add_to_buffer(final_state.copy(), pi, v, priority=abs(v)+1e-3)

    def bias_priors(self, state: np.ndarray, trace: List[int], priors: Dict[int,float]) -> Dict[int,float]:
        if not self.rules: return priors
        mult = [1.0,1.0,1.0,1.0]
        kept = []
        for r in self.rules:
            if r.ttl_steps <= 0: continue
            try_match = r.matcher(state, trace)
            if try_match:
                b = r.apply(state, trace)
                mult = [m*b[i] for i,m in enumerate(mult)]
                r.ttl_steps -= 1
                kept.append(r)
        self.rules = kept
        out = {a: priors.get(a, 0.0) * mult[a] for a in range(4)}
        z = sum(out.values())
        if z > 0: out = {a: out[a]/z for a in out}
        return out
