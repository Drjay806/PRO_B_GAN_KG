import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, dim: int, hidden: int, noise_dim: int) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        context: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([h, r, context, noise], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([h, r, t, context], dim=-1)).squeeze(-1)
