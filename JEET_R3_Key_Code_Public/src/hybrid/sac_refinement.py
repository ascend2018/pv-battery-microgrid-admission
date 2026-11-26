"""SAC refinement core for the hybrid EA–AC method."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim

from src.instrumentation import record_policy_update

from src.models.actor_mlp import MLP, flatten_actor_params, load_actor_params


Transition = Tuple[np.ndarray, float, np.ndarray, np.ndarray, bool, np.ndarray]
FROZEN_AC_GAMMA = 0.99
DISCOUNT_INTERPRETATION = "discounted_local_refinement"
EPISODE_OBJECTIVE_AGGREGATION = "undiscounted_sum"


def _squashed_gaussian_sample(
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized tanh-Gaussian action and corrected log probability."""

    bounded_log_std = torch.clamp(log_std, -5.0, 2.0).expand_as(mean)
    distribution = torch.distributions.Normal(mean, bounded_log_std.exp())
    pre_tanh = distribution.rsample()
    action = torch.tanh(pre_tanh)
    log_prob = distribution.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
    return action, log_prob.sum(dim=-1, keepdim=True)


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    """Replay buffer that stores the operating preference with each transition."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.storage: list[Transition] = []
        self.ptr = 0

    def add(self, transition: Transition) -> None:
        s, a, r, s2, done, w = transition
        frozen_arrays: list[np.ndarray] = []
        for value in (s, r, s2, w):
            array = np.array(value, dtype=float, copy=True)
            array.setflags(write=False)
            frozen_arrays.append(array)
        stored = (
            frozen_arrays[0],
            float(a),
            frozen_arrays[1],
            frozen_arrays[2],
            bool(done),
            frozen_arrays[3],
        )
        if len(self.storage) < self.capacity:
            self.storage.append(stored)
        else:
            self.storage[self.ptr] = stored
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int, *, device: torch.device):
        idx = np.random.choice(len(self.storage), batch_size, replace=False)
        batch = [self.storage[i] for i in idx]
        s, a, r, s2, d, w = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(a), dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.array(r), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(s2), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(d), dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.array(w), dtype=torch.float32, device=device),
        )


@dataclass
class SACConfig:
    """Minimal SAC config."""

    gamma: float = FROZEN_AC_GAMMA
    alpha: float = 0.2
    lr: float = 3e-4
    batch_size: int = 32
    actor_warmup_transitions: int = 32
    steps_per_episode: int = 24
    device: str = "cuda"

    def __post_init__(self) -> None:
        if float(self.gamma) != FROZEN_AC_GAMMA:
            raise ValueError(f"The SAC discount factor must be {FROZEN_AC_GAMMA}.")


class ScalarizedSAC:
    """Scalarized SAC agent."""

    def __init__(self, obs_dim: int, cfg: SACConfig, weight: np.ndarray, seed: int, init_theta: np.ndarray | None = None):
        set_seed(seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAC refinement requires an available CUDA device.")
        self.weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        self.obs_dim = obs_dim
        self.actor = MLP(obs_dim, 1).to(self.device)
        self.log_std = torch.nn.Parameter(torch.tensor([-1.0], device=self.device))
        critic_dim = obs_dim + 1 + 3
        self.critic1 = MLP(critic_dim, 1).to(self.device)
        self.critic2 = MLP(critic_dim, 1).to(self.device)
        self.critic1_target = MLP(critic_dim, 1).to(self.device)
        self.critic2_target = MLP(critic_dim, 1).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.opt_actor = optim.Adam([*self.actor.parameters(), self.log_std], lr=cfg.lr)
        self.opt_critic = optim.Adam(
            [*self.critic1.parameters(), *self.critic2.parameters()], lr=cfg.lr
        )
        self.buffer = ReplayBuffer()
        if init_theta is not None:
            load_actor_params(self.actor, init_theta)

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> float:
        """Return a squashed-Gaussian training action or deterministic mean action."""

        with torch.no_grad():
            mean = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).reshape(1, 1)
            action = torch.tanh(mean) if deterministic else _squashed_gaussian_sample(mean, self.log_std)[0]
        return float(action.squeeze().item())

    def update(self):
        return self.update_with_actor_control(update_actor=True)

    def update_with_actor_control(self, *, update_actor: bool):
        if len(self.buffer.storage) < self.cfg.batch_size:
            return {}
        s, a, r_vec, s2, done, w = self.buffer.sample(self.cfg.batch_size, device=self.device)
        # scalarize rewards
        r = torch.sum(w * r_vec, dim=1, keepdim=True)  # Scalarize the reward vector using its preference weights.
        # Critic loss
        with torch.no_grad():
            a2, next_log_prob = _squashed_gaussian_sample(self.actor(s2), self.log_std)
            q1_target = self.critic1_target(torch.cat([s2, a2, w], dim=1))
            q2_target = self.critic2_target(torch.cat([s2, a2, w], dim=1))
            soft_target = torch.minimum(q1_target, q2_target) - self.cfg.alpha * next_log_prob
            y = r + self.cfg.gamma * (1 - done) * soft_target
        critic_input = torch.cat([s, a, w], dim=1)
        q1 = self.critic1(critic_input)
        q2 = self.critic2(critic_input)
        critic_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        # Actor loss
        actor_loss = None
        entropy_term = None
        if update_actor:
            a_pi, log_prob = _squashed_gaussian_sample(self.actor(s), self.log_std)
            q1_pi = self.critic1(torch.cat([s, a_pi, w], dim=1))
            q2_pi = self.critic2(torch.cat([s, a_pi, w], dim=1))
            actor_loss = (self.cfg.alpha * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            entropy_term = float((-log_prob.mean()).item())
        # target update
        tau = 0.01
        with torch.no_grad():
            for critic, target in (
                (self.critic1, self.critic1_target),
                (self.critic2, self.critic2_target),
            ):
                for p, tp in zip(critic.parameters(), target.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
        if actor_loss is not None:
            record_policy_update()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": None if actor_loss is None else actor_loss.item(),
            "entropy_term": entropy_term,
        }

    def get_theta(self) -> np.ndarray:
        """Flatten actor parameters to EA-compatible vector."""

        return flatten_actor_params(self.actor)

    def set_theta(self, theta: np.ndarray) -> None:
        """Load flattened parameters into actor."""

        load_actor_params(self.actor, theta)

    def reset_refinement_actor(self, theta: np.ndarray) -> None:
        """Load one EA elite while retaining shared replay and twin critics."""

        self.set_theta(theta)
        with torch.no_grad():
            self.log_std.fill_(-1.0)
        self.opt_actor = optim.Adam([*self.actor.parameters(), self.log_std], lr=self.cfg.lr)
