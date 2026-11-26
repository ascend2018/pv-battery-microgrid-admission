"""Shared actor network and parameter vector utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple 2-layer MLP used for actor parameterization."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dout),
        )

    def forward(self, x):
        return self.net(x)


def flatten_actor_params(actor: nn.Module) -> np.ndarray:
    """Flatten actor parameters into a 1-D numpy vector."""

    params = [p.detach().cpu().numpy().reshape(-1) for p in actor.parameters()]
    return np.concatenate(params) if params else np.array([], dtype=np.float32)


def load_actor_params(actor: nn.Module, theta: np.ndarray) -> None:
    """Load flattened parameter vector into actor."""

    vec = np.asarray(theta, dtype=np.float32).ravel()
    total = sum(p.numel() for p in actor.parameters())
    if vec.size != total:
        raise ValueError(f"theta size {vec.size} does not match actor params {total}")
    offset = 0
    for p in actor.parameters():
        num = p.numel()
        vals = vec[offset : offset + num].reshape(p.data.shape)
        p.data.copy_(torch.as_tensor(vals, dtype=p.data.dtype, device=p.data.device))
        offset += num


def actor_param_dim(obs_dim: int) -> int:
    """Return flattened actor parameter dimension for the given obs dim."""

    dummy = MLP(obs_dim, 1)
    return sum(p.numel() for p in dummy.parameters())
