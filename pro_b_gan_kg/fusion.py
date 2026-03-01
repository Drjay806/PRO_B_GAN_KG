import torch
from torch import nn


class FusionConcat(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_sem = nn.LayerNorm(dim)
        self.norm_struct = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, sem: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        sem = self.norm_sem(sem)
        struct = self.norm_struct(struct)
        out = self.proj(torch.cat([sem, struct], dim=-1))
        return self.norm_out(out)


class FusionGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_sem = nn.LayerNorm(dim)
        self.norm_struct = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, sem: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        sem = self.norm_sem(sem)
        struct = self.norm_struct(struct)
        gate = torch.sigmoid(self.gate(torch.cat([sem, struct], dim=-1)))
        return gate * sem + (1 - gate) * struct
