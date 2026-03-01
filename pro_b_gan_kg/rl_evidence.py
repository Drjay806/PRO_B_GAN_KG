from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class EvidenceStep:
    head: int
    rel: int
    tail: int
    reward: float


class EvidenceEnv:
    def __init__(
        self,
        neighbors: Dict[Tuple[int, int], List[int]],
        target_tail: int,
        hub_penalty: float = 0.1,
    ) -> None:
        self.neighbors = neighbors
        self.target_tail = target_tail
        self.hub_penalty = hub_penalty
        self.visited = set()

    def available_actions(self, h_id: int, r_id: int) -> List[int]:
        return self.neighbors.get((h_id, r_id), [])

    def step(self, h_id: int, r_id: int, t_id: int) -> float:
        reward = 1.0 if t_id == self.target_tail else 0.0
        reward -= self.hub_penalty if t_id in self.visited else 0.0
        self.visited.add(t_id)
        return reward


class EvidencePolicy(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, t], dim=-1)).squeeze(-1)


def run_evidence_rollout(
    policy: EvidencePolicy,
    entity_emb: torch.Tensor,
    neighbors: Dict[Tuple[int, int], List[int]],
    query: Tuple[int, int],
    target_tail: int,
    budget: int = 3,
) -> List[EvidenceStep]:
    h_id, r_id = query
    env = EvidenceEnv(neighbors, target_tail)
    steps: List[EvidenceStep] = []
    h_vec = entity_emb[h_id]

    for _ in range(budget):
        actions = env.available_actions(h_id, r_id)
        if not actions:
            break
        t_vec = entity_emb[torch.tensor(actions, device=entity_emb.device)]
        scores = policy(h_vec.repeat(len(actions), 1), t_vec)
        probs = torch.softmax(scores, dim=0).detach().cpu().numpy()
        choice = int(np.random.choice(actions, p=probs))
        reward = env.step(h_id, r_id, choice)
        steps.append(EvidenceStep(head=h_id, rel=r_id, tail=choice, reward=reward))

    return steps
