from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass
class PatchSample:
    patch: torch.Tensor
    target: torch.Tensor


class PatchDataset(Dataset):
    def __init__(self, patches: List[PatchSample]) -> None:
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> PatchSample:
        return self.patches[idx]


class PatchInpaintingModel(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        self.size = size

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)


def build_patch(adj: torch.Tensor, nodes: List[int], size: int) -> torch.Tensor:
    patch = adj[nodes][:, nodes]
    if patch.shape[0] < size:
        pad = size - patch.shape[0]
        patch = torch.nn.functional.pad(patch, (0, pad, 0, pad))
    return patch[:size, :size]


def rerank_candidates(
    adj: torch.Tensor,
    h_id: int,
    candidates: List[int],
    model: PatchInpaintingModel,
    size: int,
) -> List[Tuple[int, float]]:
    scores = []
    for cand in candidates:
        nodes = [h_id, cand]
        patch = build_patch(adj, nodes, size).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(patch)
            score = float(pred.mean().item())
        scores.append((cand, score))
    scores.sort(key=lambda x: -x[1])
    return scores
