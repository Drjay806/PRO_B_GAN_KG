from typing import Tuple

import torch
from torch import nn


def comp_op(lhs: torch.Tensor, rhs: torch.Tensor, op: str) -> torch.Tensor:
    if op == "mul":
        return lhs * rhs
    if op == "sub":
        return lhs - rhs
    if op == "add":
        return lhs + rhs
    raise ValueError("Unsupported comp op")


class CompGCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float, op: str) -> None:
        super().__init__()
        self.op = op
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        entity_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0]
        dst = edge_index[1]
        rel = rel_emb[edge_type]
        msg = comp_op(entity_emb[src], rel, self.op)

        out = torch.zeros_like(entity_emb)
        out.index_add_(0, dst, msg)

        out = self.lin(out)
        out = self.dropout(out)
        out = self.norm(out + entity_emb)

        return out, rel_emb


class CompGCN(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float, op: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CompGCNLayer(dim, dropout, op) for _ in range(layers)])

    def forward(
        self,
        entity_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = entity_emb
        r = rel_emb
        for layer in self.layers:
            h, r = layer(h, r, edge_index, edge_type)
        return h
