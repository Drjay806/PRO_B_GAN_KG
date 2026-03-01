from typing import List, Optional, Tuple

import torch
from torch import nn


class ContextAttention(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.key = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_emb: torch.Tensor,
        r_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hub_bias: Optional[torch.Tensor] = None,
        type_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(torch.cat([h_emb, r_emb], dim=-1))
        k = self.key(neighbor_emb)
        scores = torch.sum(q.unsqueeze(1) * k, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        if hub_bias is not None:
            scores = scores + hub_bias
        if type_bias is not None:
            scores = scores + type_bias

        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        context = torch.sum(alpha.unsqueeze(-1) * neighbor_emb, dim=1)
        return context, alpha


def batch_neighbors(
    batch_pairs: List[List[int]],
    entity_emb: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(n) for n in batch_pairs), default=1)
    batch_size = len(batch_pairs)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    neighbor_emb = torch.zeros(batch_size, max_len, entity_emb.shape[1], device=device)

    for i, neighbors in enumerate(batch_pairs):
        if not neighbors:
            continue
        length = len(neighbors)
        mask[i, :length] = 1
        neighbor_emb[i, :length] = entity_emb[torch.tensor(neighbors, device=device)]

    return neighbor_emb, mask
