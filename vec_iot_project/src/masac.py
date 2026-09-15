"""Discrete Multi-Agent Soft Actor-Critic with centralized critics.

Each vehicle is an agent with shared actor parameters.  The critics are
centralized: they receive the global state plus the acting agent's one-hot
action.  The implementation automatically uses CUDA when available (for
Colab GPU runtimes) and otherwise falls back to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class MASACConfig:
    gamma: float = 0.98
    tau: float = 0.01
    alpha: float = 0.15
    lr: float = 3e-4
    hidden: int = 256
    batch_size: int = 128
    replay_size: int = 100_000
    warmup: int = 1_000
    updates_per_step: int = 1
    seed: int = 42
    # Automatic GPU support. Set to 'cpu' to force CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size
        self.data = []
        self.ptr = 0

    def add(self, s, a, r, ns, done, mask, next_mask):
        item = (
            np.asarray(s, np.float32), int(a), float(r),
            np.asarray(ns, np.float32), float(done),
            np.asarray(mask, bool), np.asarray(next_mask, bool)
        )
        if len(self.data) < self.size:
            self.data.append(item)
        else:
            self.data[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.size

    def __len__(self):
        return len(self.data)

    def sample(self, batch: int, device: torch.device):
        items = random.sample(self.data, batch)
        s, a, r, ns, d, m, nm = zip(*items)
        return (
            torch.as_tensor(np.stack(s), dtype=torch.float32, device=device),
            torch.as_tensor(a, dtype=torch.long, device=device),
            torch.as_tensor(r, dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(ns), dtype=torch.float32, device=device),
            torch.as_tensor(d, dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(m), dtype=torch.bool, device=device),
            torch.as_tensor(np.stack(nm), dtype=torch.bool, device=device),
        )


class DiscreteMASAC:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: MASACConfig = MASACConfig()):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        requested = str(cfg.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.actor = MLP(obs_dim, cfg.hidden, action_dim).to(self.device)
        critic_in = state_dim + action_dim
        self.q1 = MLP(critic_in, cfg.hidden, 1).to(self.device)
        self.q2 = MLP(critic_in, cfg.hidden, 1).to(self.device)
        self.tq1 = MLP(critic_in, cfg.hidden, 1).to(self.device)
        self.tq2 = MLP(critic_in, cfg.hidden, 1).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.oa = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.o1 = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.o2 = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)

    @staticmethod
    def masked_logits(logits, mask):
        mask = mask.bool()
        # The environment should always leave at least LOCAL valid.
        return logits.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits = self.masked_logits(self.actor(x), m)
        if deterministic:
            return int(logits.argmax(-1).item())
        return int(Categorical(logits=logits).sample().item())

    def _dist(self, obs, mask):
        logits = self.masked_logits(self.actor(obs), mask)
        dist = Categorical(logits=logits)
        probs = dist.probs
        logp = torch.log(probs.clamp_min(1e-8))
        return probs, logp

    def update(self):
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup):
            return None

        stats = None
        for _ in range(self.cfg.updates_per_step):
            s, a, r, ns, d, mask, next_mask = self.replay.sample(self.cfg.batch_size, self.device)
            onehot = torch.nn.functional.one_hot(a, self.action_dim).float()
            q1 = self.q1(torch.cat([s, onehot], dim=-1)).squeeze(-1)
            q2 = self.q2(torch.cat([s, onehot], dim=-1)).squeeze(-1)

            with torch.no_grad():
                p2, logp2 = self._dist(ns[:, :self.obs_dim], next_mask)
                vals = []
                for ai in range(self.action_dim):
                    aidx = torch.full((ns.size(0),), ai, dtype=torch.long, device=self.device)
                    oh = torch.nn.functional.one_hot(aidx, self.action_dim).float()
                    qa = torch.min(
                        self.tq1(torch.cat([ns, oh], -1)).squeeze(-1),
                        self.tq2(torch.cat([ns, oh], -1)).squeeze(-1),
                    )
                    # Invalid next actions are masked by p2=0.
                    vals.append(qa)
                qmat = torch.stack(vals, dim=1)
                v = torch.sum(p2 * (qmat - self.cfg.alpha * logp2), dim=1)
                target = r + self.cfg.gamma * (1.0 - d) * v

            lq1 = ((q1 - target) ** 2).mean()
            lq2 = ((q2 - target) ** 2).mean()
            self.o1.zero_grad(set_to_none=True)
            lq1.backward()
            self.o1.step()
            self.o2.zero_grad(set_to_none=True)
            lq2.backward()
            self.o2.step()

            p, logp = self._dist(s[:, :self.obs_dim], mask)
            vals = []
            for ai in range(self.action_dim):
                aidx = torch.full((s.size(0),), ai, dtype=torch.long, device=self.device)
                oh = torch.nn.functional.one_hot(aidx, self.action_dim).float()
                qa = torch.min(
                    self.q1(torch.cat([s, oh], -1)).squeeze(-1),
                    self.q2(torch.cat([s, oh], -1)).squeeze(-1),
                )
                vals.append(qa)
            qmat = torch.stack(vals, dim=1)
            actor_loss = torch.sum(p * (self.cfg.alpha * logp - qmat), dim=1).mean()
            self.oa.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.oa.step()

            with torch.no_grad():
                for tgt, src in ((self.tq1, self.q1), (self.tq2, self.q2)):
                    for tp, sp in zip(tgt.parameters(), src.parameters()):
                        tp.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * sp)
            stats = {
                "q_loss": float((lq1 + lq2).item()),
                "actor_loss": float(actor_loss.item()),
            }
        return stats

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
