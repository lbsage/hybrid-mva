import logging
from typing import Dict, Optional

import numpy as np

from .config import SolverConfig

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils as nn_utils
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "PyTorch is not installed. Neural-network predictor will be disabled; "
        "falling back to HRMPredictorNP (heuristic-only)."
    )
    TORCH_AVAILABLE = False

# Always export TorchHRM so ``from .a2c_trainer import TorchHRM`` works even
# when PyTorch is absent.  solver.py guards usage with ``if TORCH_AVAILABLE``.
TorchHRM = None  # overwritten below when torch is available

# ---------------------------------------------------------------------------
# Shared puzzle utility (avoids circular import with solver.py)
# ---------------------------------------------------------------------------
_ACTION_DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _legal_moves_np(state: np.ndarray):
    """Return list of legal action indices for *state* (16-element int array)."""
    blank = int(np.where(state == 0)[0][0])
    r, c = divmod(blank, 4)
    moves = []
    for a, (dr, dc) in _ACTION_DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            moves.append(a)
    return moves


# ---------------------------------------------------------------------------
# Neural-network model (only defined when torch is available)
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class TorchHRM(nn.Module):  # type: ignore[no-redef]
        """Two-headed MLP: shared backbone → policy logits + scalar value."""

        def __init__(self, hidden: int = 128, state_norm: float = 15.0):
            super().__init__()
            self.state_norm = state_norm
            self.backbone = nn.Sequential(
                nn.Linear(16, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden, 4)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x: "torch.Tensor"):
            h = self.backbone(x)
            logits = self.policy_head(h)
            v = self.value_head(h).squeeze(-1)
            return logits, v

        @torch.no_grad()
        def predict(self, state_np: np.ndarray) -> tuple:
            """Return (policy_dict, value_float) for a single board state."""
            device = self.value_head.weight.device
            x = (
                torch.tensor(state_np, dtype=torch.float32, device=device)
                / self.state_norm
            ).view(1, 16)
            logits, v = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            legal = _legal_moves_np(state_np)
            masked = [probs[a] if a in legal else 0.0 for a in range(4)]
            total = sum(masked)
            if total <= 0:
                n = len(legal)
                masked = [1.0 / n if a in legal else 0.0 for a in range(4)]
            else:
                masked = [p / total for p in masked]

            return {a: masked[a] for a in range(4)}, float(v.item())


# ---------------------------------------------------------------------------
# A2C trainer
# ---------------------------------------------------------------------------

#: Type alias for the dict returned by :meth:`A2CTrainer.train_step_from_buffer`.
TrainResult = Dict[str, Optional[float]]

_SKIPPED_RESULT: TrainResult = {
    "loss": None,
    "policy_loss": None,
    "value_loss": None,
    "entropy": None,
    "adv_mean": None,
    "skipped": True,
}


class A2CTrainer:
    """Advantage Actor-Critic update step driven by a :class:`ReplayBuffer`.

    When PyTorch is unavailable, or when *hrm_torch* is ``None``, the trainer
    is disabled and :meth:`train_step_from_buffer` is a no-op that returns a
    result dict with ``skipped=True``.
    """

    def __init__(
        self,
        hrm_torch,
        lr: float = 3e-4,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device=None,
    ):
        self.enabled = TORCH_AVAILABLE and (hrm_torch is not None)
        # Always initialise every attribute so attribute access never raises
        # AttributeError regardless of whether the trainer is enabled.
        self.hrm = hrm_torch if self.enabled else None
        self.device = None
        self.opt = None
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        if not self.enabled:
            logger.debug("A2CTrainer disabled (torch unavailable or no model provided).")
            return

        self.device = device or next(self.hrm.parameters()).device
        self.opt = Adam(self.hrm.parameters(), lr=lr)

    def train_step_from_buffer(
        self,
        rb,
        batch_size: int = 64,
        clip_value_loss: bool = True,
    ) -> TrainResult:
        """Sample a mini-batch from *rb* and perform one A2C gradient step.

        Returns a :data:`TrainResult` dict.  When the step is skipped (trainer
        disabled, buffer empty, or sample unavailable) every metric field is
        ``None`` and ``"skipped"`` is ``True``.
        """
        if not self.enabled:
            return _SKIPPED_RESULT

        if rb.replay_buffer is None or len(rb.replay_buffer) == 0:
            logger.debug("train_step_from_buffer: replay buffer empty, skipping.")
            return _SKIPPED_RESULT

        sample = rb.replay_buffer.sample(batch_size)
        if sample is None:
            logger.debug("train_step_from_buffer: buffer.sample() returned None, skipping.")
            return _SKIPPED_RESULT

        states_np, pi_targets_np, v_targets_np, idxs, isw = sample

        x = torch.tensor(states_np, dtype=torch.float32, device=self.device) / self.hrm.state_norm
        pi_t = torch.tensor(pi_targets_np, dtype=torch.float32, device=self.device)
        v_t = torch.tensor(v_targets_np, dtype=torch.float32, device=self.device)

        logits, v_pred = self.hrm(x)
        logp = F.log_softmax(logits, dim=-1)
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * logp).sum(dim=-1)

        advantage = (v_t - v_pred).detach()
        policy_logp = (pi_t * logp).sum(dim=-1)
        policy_loss = -(policy_logp * advantage)

        if clip_value_loss:
            value_loss = F.smooth_l1_loss(v_pred, v_t, reduction="none")
        else:
            value_loss = F.mse_loss(v_pred, v_t, reduction="none")

        if isw is not None:
            w = torch.tensor(isw, dtype=torch.float32, device=self.device)
            policy_loss = (policy_loss * w).mean()
            value_loss = (value_loss * w).mean()
            ent_loss = (entropy * w).mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss = value_loss.mean()
            ent_loss = entropy.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn_utils.clip_grad_norm_(self.hrm.parameters(), self.max_grad_norm)
        self.opt.step()

        if isw is not None:
            td_errors = (v_t - v_pred).abs().detach().cpu().numpy().tolist()
            rb.replay_buffer.update_priorities(idxs, td_errors)

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent_loss.item()),
            "adv_mean": float(advantage.mean().item()),
            "skipped": False,
        }
