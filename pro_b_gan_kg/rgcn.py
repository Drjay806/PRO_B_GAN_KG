from typing import Tuple

import torch
from torch import nn


class RGCNLayer(nn.Module):
    def __init__(self, dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.rel_lin = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_relations)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        entity_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        out = torch.zeros_like(entity_emb)
        for rel_id, lin in enumerate(self.rel_lin):
            mask = edge_type == rel_id
            if mask.sum() == 0:
                continue
            rel_src = src[mask]
            rel_dst = dst[mask]
            msg = lin(entity_emb[rel_src])
            out.index_add_(0, rel_dst, msg)

        out = self.dropout(out)
        out = self.norm(out + entity_emb)
        return out


class RGCN(nn.Module):
    def __init__(self, dim: int, num_relations: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RGCNLayer(dim, num_relations, dropout) for _ in range(layers)]
        )

    def forward(
        self,
        entity_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        h = entity_emb
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return h
