"""Ordinary (non-branch) networks with matched parameter counts."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..data import normalize_cifar


class MLP(nn.Module):
    is_branch = False

    def __init__(self, depth: int, width: int, seed: int = 0, d_in: int = 784):
        super().__init__(); torch.manual_seed(seed); dims = [d_in] + [width] * depth
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:])]); self.head = nn.Linear(width, 10)

    def forward(self, x, mode=None, routes=None):
        for layer in self.layers: x = layer(x).tanh()
        return self.head(x)


def mlp_width(depth: int, n_params: int) -> int:
    """Hidden width whose tanh MLP has closest to n_params parameters."""
    return min(range(16, 400), key=lambda w: abs(785 * w + (depth - 1) * (w * w + w) + 11 * w - n_params))


class CNN(nn.Module):
    """tanh CNN 32/64/128 (3x3, stride 2) -> LayerNorm -> dense 32 -> head (159,146 params)."""
    is_branch = False

    def __init__(self, seed: int = 0, n_out: int = 10):
        super().__init__(); torch.manual_seed(seed)
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1) for a, b in zip([3, 32, 64], [32, 64, 128])])
        self.fc = nn.Linear(2048, 32); self.head = nn.Linear(32, n_out)

    def forward(self, x, mode=None, routes=None):
        h = normalize_cifar(x)
        for conv in self.convs: h = conv(h).tanh()
        return self.head(self.fc(F.layer_norm(h.flatten(1), (2048,))).tanh())
