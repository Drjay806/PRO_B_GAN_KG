from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class EvidenceStep:
    head: int
    rel: int
    tail: int
    reward: float
    log_prob: float = 0.0


@dataclass
class Rollout:
    steps: List[EvidenceStep] = field(default_factory=list)
    total_reward: float = 0.0
    reached_target: bool = False


class EvidenceEnv:
    def __init__(
        self,
        neighbors: Dict[Tuple[int, int], List[int]],
        target_tail: int,
        entity_emb: Optional[torch.Tensor] = None,
        hub_penalty: float = 0.1,
        proximity_bonus: float = 0.1,
    ) -> None:
        self.neighbors = neighbors
        self.target_tail = target_tail
        self.entity_emb = entity_emb
        self.hub_penalty = hub_penalty
        self.proximity_bonus = proximity_bonus
        self.visited: set = set()
        self.done = False

    def available_actions(self, current: int) -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = []
        for (h, r), tails in self.neighbors.items():
            if h == current:
                for t in tails:
                    if t not in self.visited:
                        actions.append((r, t))
        return actions

    def step(self, current: int, r_id: int, t_id: int) -> float:
        if t_id == self.target_tail:
            self.done = True
            return 1.0

        reward = 0.0

        if self.entity_emb is not None:
            with torch.no_grad():
                sim = torch.cosine_similarity(
                    self.entity_emb[t_id].unsqueeze(0),
                    self.entity_emb[self.target_tail].unsqueeze(0),
                ).item()
            reward += self.proximity_bonus * max(sim, 0.0)

        if t_id in self.visited:
            reward -= self.hub_penalty

        self.visited.add(t_id)
        return reward


class EvidencePolicy(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, current: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([current, target, candidate], dim=-1)).squeeze(-1)


def run_evidence_rollout(
    policy: EvidencePolicy,
    entity_emb: torch.Tensor,
    neighbors: Dict[Tuple[int, int], List[int]],
    query: Tuple[int, int],
    target_tail: int,
    budget: int = 3,
) -> List[EvidenceStep]:
    h_id, r_id = query
    env = EvidenceEnv(neighbors, target_tail, entity_emb=entity_emb)
    steps: List[EvidenceStep] = []
    current = h_id
    target_vec = entity_emb[target_tail]

    for _ in range(budget):
        actions = env.available_actions(current)
        if not actions or env.done:
            break
        action_r = [a[0] for a in actions]
        action_t = [a[1] for a in actions]
        t_ids = torch.tensor(action_t, device=entity_emb.device)
        t_vec = entity_emb[t_ids]
        cur_vec = entity_emb[current]

        scores = policy(
            cur_vec.unsqueeze(0).expand(len(actions), -1),
            target_vec.unsqueeze(0).expand(len(actions), -1),
            t_vec,
        )
        probs = torch.softmax(scores, dim=0).detach().cpu().numpy()
        idx = int(np.random.choice(len(actions), p=probs))
        chosen_r, chosen_t = action_r[idx], action_t[idx]
        reward = env.step(current, chosen_r, chosen_t)
        steps.append(EvidenceStep(head=current, rel=chosen_r, tail=chosen_t, reward=reward))
        current = chosen_t

    return steps


def collect_rollout(
    policy: EvidencePolicy,
    entity_emb: torch.Tensor,
    neighbors: Dict[Tuple[int, int], List[int]],
    h_id: int,
    target_tail: int,
    budget: int = 3,
    hub_penalty: float = 0.1,
    proximity_bonus: float = 0.1,
) -> Rollout:
    env = EvidenceEnv(
        neighbors, target_tail, entity_emb=entity_emb,
        hub_penalty=hub_penalty, proximity_bonus=proximity_bonus,
    )
    rollout = Rollout()
    current = h_id
    target_vec = entity_emb[target_tail]

    for _ in range(budget):
        actions = env.available_actions(current)
        if not actions or env.done:
            break

        action_r = [a[0] for a in actions]
        action_t = [a[1] for a in actions]
        t_ids = torch.tensor(action_t, device=entity_emb.device)
        t_vec = entity_emb[t_ids]
        cur_vec = entity_emb[current]

        scores = policy(
            cur_vec.unsqueeze(0).expand(len(actions), -1),
            target_vec.unsqueeze(0).expand(len(actions), -1),
            t_vec,
        )
        probs = torch.softmax(scores, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        chosen_r = action_r[idx.item()]
        chosen_t = action_t[idx.item()]
        reward = env.step(current, chosen_r, chosen_t)

        rollout.steps.append(EvidenceStep(
            head=current, rel=chosen_r, tail=chosen_t,
            reward=reward, log_prob=log_prob.item(),
        ))
        current = chosen_t

    rollout.reached_target = env.done
    rollout.total_reward = sum(s.reward for s in rollout.steps)
    return rollout


def reinforce_loss(
    rollouts: List[Rollout],
    gamma: float = 0.99,
    baseline: float = 0.0,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    policy_losses: List[torch.Tensor] = []
    entropy_bonus: List[torch.Tensor] = []

    for rollout in rollouts:
        if not rollout.steps:
            continue

        returns: List[float] = []
        G = 0.0
        for step in reversed(rollout.steps):
            G = step.reward + gamma * G
            returns.insert(0, G)

        for step, ret in zip(rollout.steps, returns):
            advantage = ret - baseline
            log_p = torch.tensor(step.log_prob, requires_grad=True)
            policy_losses.append(-log_p * advantage)
            prob = torch.exp(log_p)
            entropy_bonus.append(-(prob * log_p))

    if not policy_losses:
        return torch.tensor(0.0, requires_grad=True)

    loss = torch.stack(policy_losses).mean()
    if entropy_bonus:
        loss = loss - entropy_coef * torch.stack(entropy_bonus).mean()
    return loss


def train_evidence_policy(
    policy: EvidencePolicy,
    entity_emb: torch.Tensor,
    neighbors: Dict[Tuple[int, int], List[int]],
    train_triples: List[Tuple[int, int, int]],
    optimizer: torch.optim.Optimizer,
    budget: int = 3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    baseline_decay: float = 0.95,
    batch_size: int = 256,
    hub_penalty: float = 0.1,
    proximity_bonus: float = 0.1,
    logger=None,
) -> Dict[str, float]:
    policy.train()
    baseline = 0.0
    total_reward = 0.0
    total_reached = 0
    total_steps = 0
    total_loss = 0.0
    num_batches = 0

    indices = np.random.permutation(len(train_triples))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        rollouts: List[Rollout] = []

        for i in batch_idx:
            h, r, t = train_triples[i]
            rollout = collect_rollout(
                policy=policy,
                entity_emb=entity_emb,
                neighbors=neighbors,
                h_id=h,
                target_tail=t,
                budget=budget,
                hub_penalty=hub_penalty,
                proximity_bonus=proximity_bonus,
            )
            rollouts.append(rollout)

        loss = reinforce_loss(
            rollouts, gamma=gamma, baseline=baseline, entropy_coef=entropy_coef,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        batch_reward = sum(r.total_reward for r in rollouts)
        batch_reached = sum(1 for r in rollouts if r.reached_target)
        baseline = baseline_decay * baseline + (1 - baseline_decay) * (batch_reward / max(len(rollouts), 1))

        total_reward += batch_reward
        total_reached += batch_reached
        total_steps += len(batch_idx)
        total_loss += loss.item()
        num_batches += 1

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "avg_reward": total_reward / max(total_steps, 1),
        "hit_rate": total_reached / max(total_steps, 1),
        "num_triples": total_steps,
    }

    if logger:
        logger.info(
            "  RL epoch: loss=%.4f  avg_reward=%.4f  hit_rate=%.3f  (%d triples)",
            metrics["loss"], metrics["avg_reward"], metrics["hit_rate"], metrics["num_triples"],
        )

    return metrics
