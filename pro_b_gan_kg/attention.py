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

    # Build padded index and mask tensors (no in-place ops on graph-tracked tensors)
    padded_ids = []
    mask_rows = []
    for neighbors in batch_pairs:
        length = len(neighbors)
        if length == 0:
            padded_ids.append([0] * max_len)
            mask_rows.append([False] * max_len)
        else:
            padded_ids.append(neighbors + [0] * (max_len - length))
            mask_rows.append([True] * length + [False] * (max_len - length))

    idx = torch.tensor(padded_ids, dtype=torch.long, device=device)   # [B, max_len]
    mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)    # [B, max_len]

    # Single vectorised gather — gradient-safe, no in-place assignment
    neighbor_emb = entity_emb[idx]  # [B, max_len, dim]

    return neighbor_emb, mask
