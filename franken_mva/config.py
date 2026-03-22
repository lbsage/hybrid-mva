from dataclasses import dataclass, field
from typing import List


@dataclass
class SolverConfig:
    """Central configuration for all solver hyperparameters.

    Pass an instance to ``Solver`` to override defaults, e.g.::

        cfg = SolverConfig(sims=400, lr=1e-4)
        solver = Solver(config=cfg)
    """

    # ------------------------------------------------------------------ MCTS
    sims: int = 200
    """Number of MCTS simulations per step."""

    c_puct: float = 1.2
    """PUCT exploration constant."""

    dirichlet_alpha: float = 0.3
    """Dirichlet noise concentration parameter applied at the root."""

    dirichlet_epsilon: float = 0.25
    """Fraction of Dirichlet noise mixed into root priors."""

    discount_gamma: float = 0.99
    """Discount factor applied during MCTS backup."""

    prior_floor: float = 1e-6
    """Minimum prior probability assigned to any child node."""

    # -------------------------------------------------------- Neural network
    hrm_hidden: int = 128
    """Hidden layer width of TorchHRM."""

    state_norm: float = 15.0
    """Divisor used to normalise tile values into [0, 1]."""

    heuristic_scale: float = 60.0
    """Divisor used to convert Manhattan-distance heuristic to a value in
    roughly [-1, 0]; 60 is an empirical upper-bound for the 15-puzzle."""

    # ------------------------------------------------------------ A2C trainer
    lr: float = 3e-4
    """Adam learning rate."""

    entropy_coef: float = 1e-3
    """Entropy regularisation coefficient."""

    value_coef: float = 0.5
    """Value loss coefficient."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    train_batch_size: int = 128
    """Number of transitions sampled per training step."""

    # --------------------------------------------------------- Replay buffer
    buffer_capacity: int = 200_000
    """Maximum number of transitions stored in the replay buffer."""

    buffer_alpha: float = 0.6
    """Priority exponent for prioritised experience replay."""

    priority_floor: float = 1e-3
    """Minimum priority added to |V| when storing a transition."""

    # ----------------------------------------------------------- Reflexion
    oscillation_window: int = 6
    """Trace window for oscillation (2-cycle) detection."""

    plateau_window: int = 10
    """Heuristic window for plateau detection."""

    ttl_break_cycle: int = 800
    """Rule lifetime (steps) for the break-2-cycle rule."""

    ttl_shake_plateau: int = 400
    """Rule lifetime (steps) for the shake-plateau rule."""

    ttl_unjam_corner: int = 600
    """Rule lifetime (steps) for the unjam-bottom-right rule."""

    osc_inverse_bias: float = 0.2
    """Prior multiplier applied to the inverse of the last action to break
    oscillations (< 1 discourages back-tracking)."""

    shake_bias_updown: float = 1.15
    """Prior multiplier for up/down moves during plateau shaking."""

    shake_bias_leftright: float = 1.0
    """Prior multiplier for left/right moves during plateau shaking."""

    corner_bias: List[float] = field(default_factory=lambda: [1.2, 0.9, 1.3, 0.8])
    """Per-action prior multipliers [up, down, left, right] for unjamming the
    bottom-right corner deadlock."""

    reflexion_patch_window: int = 6
    """Number of trailing transitions to add to the replay buffer when
    reflexion fires."""

    reflexion_inv_bias: float = 0.2
    """Prior multiplier applied to the inverse action when patching the
    replay buffer after reflexion."""

    # --------------------------------------------------------------- Device
    device: str = "auto"
    """Compute device: ``"auto"`` picks CUDA when available, otherwise CPU."""
