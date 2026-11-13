"""Dendritic branch layers.

Every neuron (or conv channel) has B branches. A branch is a tanh unit over its own
inputs. Each layer keeps an ``error`` trace per (neuron, branch): a running estimate of
the loss if that branch drove the neuron. How the B branch outputs are combined:

  'trace'  training: one branch per neuron, the one with the lowest error trace
  'route'  a stored route (one branch index per neuron), per sample: (N, n_neurons)
  'soft'   image-only inference: softmax(|response|) mixture of all branches
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

INITIAL_ERROR = math.log(10)  # cross-entropy of a uniform 10-class guess


def patch_indices(n_branches: int, gen: torch.Generator, side: int = 28, patch: int = 14) -> torch.Tensor:
    """One compact patch x patch window of a side x side image per branch (random position),
    as flat pixel indices. Same input count as a 196-pixel random subset when patch = 14."""
    r = torch.randint(0, side - patch + 1, (n_branches, 2), generator=gen)
    rows = r[:, :1, None] + torch.arange(patch)[None, :, None]; cols = r[:, 1:, None] + torch.arange(patch)[None, None, :]
    return (rows * side + cols).reshape(n_branches, -1)


class SparseBranchLayer(nn.Module):
    """n neurons x B branches; each branch reads k fixed random inputs out of d_in."""

    def __init__(self, d_in: int, n: int, B: int, k: int, gen: torch.Generator, bias: str = 'uniform'):
        super().__init__(); self.n, self.B = n, B
        self.weight = nn.Parameter(torch.empty(n * B, k)); nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias == 'uniform':
            bound = 1 / math.sqrt(k); self.bias = nn.Parameter(torch.empty(n * B).uniform_(-bound, bound))
        else:
            self.bias = nn.Parameter(torch.zeros(n * B))
        self.register_buffer('indices', torch.stack([torch.randperm(d_in, generator=gen)[:k] for _ in range(n * B)]))
        self.register_buffer('error', torch.full((n, B), INITIAL_ERROR))
        self.override = None  # set by a GateRule to replace error-trace selection (ablations only)
        self.k_frac = None    # global k-winners-take-all over the layer's branches (APL-style; experiment only)

    @property
    def n_units(self) -> int:
        return self.n

    def selected(self) -> torch.Tensor:
        """Branch index per neuron used in training: lowest error trace unless overridden."""
        return self.error.argmin(-1) if self.override is None else self.override

    def responses(self, x: torch.Tensor) -> torch.Tensor:
        """Raw branch pre-activations, (N, n, B)."""
        return ((x[:, self.indices] * self.weight).sum(-1) + self.bias).reshape(-1, self.n, self.B)

    def mix(self, raw: torch.Tensor, mode: str, route: torch.Tensor | None = None):
        z = raw.tanh()
        if mode == 'soft': w = raw.abs().softmax(-1)
        elif mode == 'route': w = F.one_hot(route, self.B).float()
        else: w = F.one_hot(self.selected(), self.B).float()
        if self.k_frac:  # only the layer's top k_frac branches by |response| may be active, per input
            w = (w > 0).float() * self.active_mask(raw) if mode != 'soft' else self.active_mask(raw)
        return (z * w).sum(-1), z

    @torch.no_grad()
    def active_mask(self, raw: torch.Tensor) -> torch.Tensor:
        """APL-style global inhibition: 1 for the k_frac most strongly driven branches of the
        whole layer (all neurons together), 0 elsewhere. In 'soft' mode a neuron outputs the
        sum of its active branches (0 if none); in 'trace'/'route' mode its routed branch only
        if that branch is active, so a branch drives its neuron, and learns, only when active."""
        a = raw.abs().flatten(1); k = max(1, round(self.k_frac * a.shape[1]))
        return (a >= a.topk(k, 1).values[:, -1:]).float().view_as(raw)


class ConvBranchLayer(nn.Module):
    """3x3 stride-2 conv where every output channel has B filter sets (branches).

    frac < 1 gives each branch a fixed random subset of the input channels.
    With B = 1 this is an ordinary tanh conv layer (no routing).
    """

    def __init__(self, cin: int, cout: int, B: int, frac: float, gen: torch.Generator):
        super().__init__(); self.c, self.B = cout, B
        self.conv = nn.Conv2d(cin, cout * B, 3, 2, 1)
        if frac < 1:
            k = math.ceil(frac * cin); mask = torch.zeros(cout * B, cin, 1, 1)
            for o in range(cout * B): mask[o, torch.randperm(cin, generator=gen)[:k]] = 1
            self.register_buffer('mask', mask)
        self.register_buffer('error', torch.full((cout, B), INITIAL_ERROR))
        self.override = None

    @property
    def n_units(self) -> int:
        return self.c

    def selected(self) -> torch.Tensor:
        return self.error.argmin(-1) if self.override is None else self.override

    def responses(self, x: torch.Tensor) -> torch.Tensor:
        """Raw branch maps, (N, C, B, H, W)."""
        w = self.conv.weight * self.mask if hasattr(self, 'mask') else self.conv.weight
        r = F.conv2d(x, w, self.conv.bias, 2, 1); return r.reshape(len(x), self.c, self.B, *r.shape[-2:])

    def mix(self, raw: torch.Tensor, mode: str, route: torch.Tensor | None = None):
        z = raw.tanh()
        if self.B == 1: return z[:, :, 0], z
        if mode == 'soft': w = raw.abs().mean((3, 4)).softmax(-1)[..., None, None]
        elif mode == 'route': w = F.one_hot(route, self.B).float()[..., None, None]
        else: w = F.one_hot(self.selected(), self.B).float()[None, :, :, None, None]
        return (z * w).sum(2), z
