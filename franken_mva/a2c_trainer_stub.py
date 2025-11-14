
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class TorchHRM(nn.Module):
        def __init__(self, hidden=128):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(16, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy = nn.Linear(hidden, 4)
            self.value  = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.backbone(x)
            logits = self.policy(h)
            v = self.value(h).squeeze(-1)
            return logits, v

        @torch.no_grad()
        def predict(self, state_np: np.ndarray):
            state = torch.tensor(state_np, dtype=torch.float32, device=self.value.weight.device)
            x = (state / 15.0).view(1, 16)
            logits, v = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy().tolist()

            def _legal_moves(st):
                moves = []
                for a in range(4):
                    s = st.copy()
                    b = int(np.where(s == 0)[0][0]); r, c = divmod(b, 4)
                    drdc = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}[a]
                    nr, nc = r+drdc[0], c+drdc[1]
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        moves.append(a)
                return moves

            acts = _legal_moves(state_np)
            masked = [probs[a] if a in acts else 0.0 for a in range(4)]
            s = sum(masked)
            if s <= 0:
                masked = [1.0/len(acts) if a in acts else 0.0 for a in range(4)]
            else:
                masked = [p/s for p in masked]

            P = {a: masked[a] for a in range(4)}
            V = float(v.item())
            return P, V

class A2CTrainer:
    def __init__(self, hrm_torch, lr=3e-4, entropy_coef=1e-3, value_coef=0.5,
                 max_grad_norm=1.0, device=None):
        self.enabled = TORCH_AVAILABLE and (hrm_torch is not None)
        if not self.enabled:
            self.hrm = None
            return
        self.hrm = hrm_torch
        self.device = device or next(self.hrm.parameters()).device
        self.opt = Adam(self.hrm.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def train_step_from_buffer(self, rb, batch_size=64, clip_value_loss=True) -> dict:
        if not self.enabled or rb.replay_buffer is None or len(rb.replay_buffer) == 0:
            return {"empty": True}

        sample = rb.replay_buffer.sample(batch_size)
        if sample is None:
            return {"empty": True}
        states_np, pi_targets_np, v_targets_np, idxs, isw = sample

        import torch
        import torch.nn.functional as F

        x = torch.tensor(states_np, dtype=torch.float32, device=self.device) / 15.0
        pi_t = torch.tensor(pi_targets_np, dtype=torch.float32, device=self.device)
        v_t  = torch.tensor(v_targets_np, dtype=torch.float32, device=self.device)

        logits, v_pred = self.hrm(x)
        logp = F.log_softmax(logits, dim=-1)
        p    = torch.softmax(logits, dim=-1)
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
            value_loss  = (value_loss  * w).mean()
            ent_loss    = (entropy     * w).mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss  = value_loss.mean()
            ent_loss    = entropy.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        import torch.nn.utils as utils
        utils.clip_grad_norm_(self.hrm.parameters(), self.max_grad_norm)
        self.opt.step()

        if isw is not None:
            td_err = (v_t - v_pred).abs().detach().cpu().numpy().tolist()
            rb.replay_buffer.update_priorities(idxs, td_err)

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent_loss.item()),
            "adv_mean": float(advantage.mean().item()),
        }
