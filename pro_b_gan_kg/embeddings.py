from typing import Tuple

import torch
from torch import nn


class EntityEmbedding(nn.Module):
    def __init__(self, num_entities: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    def weight(self) -> torch.Tensor:
        return self.embedding.weight


class RelationEmbedding(nn.Module):
    def __init__(self, num_relations: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    def weight(self) -> torch.Tensor:
        return self.embedding.weight


class DistMultScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.rel = nn.Embedding(1, dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(h * r * t, dim=-1)


def distmult_score(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.sum(h * r * t, dim=-1)


def split_embeddings(
    entity_emb: torch.Tensor, relation_emb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return entity_emb, relation_emb
