"""Frozen-input dendritic pilot with online branch-local direction protection."""
import math
import torch
import torch.nn.functional as F
from torch import nn


class ProtectedBranches(nn.Module):
    def __init__(self, d=128, neurons=128, branches=4, rank=16, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(neurons, branches, d, generator=gen) / math.sqrt(d))
        self.register_buffer('readout', torch.randn(neurons, 10, generator=gen) / math.sqrt(neurons))
        self.register_buffer('enabled', torch.arange(branches)[None].expand(neurons, -1).clone() < branches-1)
        self.register_buffer('basis', torch.zeros(neurons, branches, d, rank))
        self.register_buffer('ranks', torch.zeros(neurons, branches, dtype=torch.long))
        self.rank = rank
        self.loss_ema = None
        self.stalled_steps = 0
        self.recruited = 0

    def responses(self, x):
        pre = torch.einsum('bd,nkd->bnk', x, self.weight)
        a = F.relu(pre).masked_fill(~self.enabled[None], -torch.inf)
        soma, winners = a.max(-1)
        return pre, soma, winners

    def forward(self, x):
        return self.responses(x)[1] @ self.readout * 5.

    def project(self, delta):
        coords = torch.einsum('nkdr,nkd->nkr', self.basis, delta)
        return delta - torch.einsum('nkdr,nkr->nkd', self.basis, coords)

    @torch.no_grad()
    def consolidate(self, x, winners, eligible, active=None):
        """One routed mean direction per branch per minibatch. Two-pass MGS.

        Statistics are written after the update, for examples correct/confident
        before it. No task boundaries, stored examples, or basis replacement.
        """
        n, k, d = self.weight.shape
        assignments = F.one_hot(winners, k).float() * eligible[:, None, None]
        if active is not None:
            assignments *= active[..., None]
        counts = assignments.sum(0)
        means = torch.einsum('bnk,bd->nkd', assignments, x) / counts.clamp_min(1)[..., None]
        mask = (counts > 0) & (self.ranks < self.rank) & self.enabled
        for neuron, branch in mask.nonzero().tolist():
            r = int(self.ranks[neuron, branch])
            u = means[neuron, branch]
            v = u.clone()
            q = self.basis[neuron, branch, :, :r]
            for _ in range(2):
                v -= q @ (q.T @ v)
            norm = v.norm()
            # Discard near-redundant directions; compact basis has a hard cap.
            if norm > .05 * u.norm().clamp_min(1e-12):
                self.basis[neuron, branch, :, r] = v / norm
                self.ranks[neuron, branch] += 1

    @torch.no_grad()
    def maybe_recruit(self, loss, freedom, raw):
        self.loss_ema = loss if self.loss_ema is None else .95*self.loss_ema + .05*loss
        self.stalled_steps = self.stalled_steps+1 if self.loss_ema > 1.0 and freedom < .2 else 0
        events = []
        if self.stalled_steps < 20:
            return events
        candidates = (~self.enabled[:, -1]).nonzero().flatten()
        if len(candidates):
            # Recruit up to eight neuron-local reserve branches at a time.
            importance = raw[candidates].flatten(1).norm(dim=1)
            chosen = candidates[importance.topk(min(8, len(candidates))).indices]
            self.enabled[chosen, -1] = True
            self.recruited += len(chosen)
            events = chosen.tolist()
        self.stalled_steps = 0
        return events
