"""Branch networks: an MLP for Split MNIST and a CNN for Split CIFAR-10.

Both subclass :class:`BranchNetwork`, which implements the training rule:

* each neuron is driven by the branch with the lowest error trace;
* after each batch, every branch's trace moves towards its estimated substitution
  loss CE + g * (z_branch - h), with g the gradient of CE w.r.t. the neuron output
  (first-order estimate of the loss if that branch had been used);
* loss = 1.5 CE + 0.5 * mean over layers of the same first-order substitution term;
* at a task boundary, the per-neuron winners are stored as that task's route.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..data import normalize_cifar
from .layers import ConvBranchLayer, SparseBranchLayer


class BranchNetwork(nn.Module):
    is_branch = True

    def branch_layers(self) -> list[nn.Module]:
        raise NotImplementedError

    def forward(self, x, mode: str = 'soft', routes=None, internals: bool = False):
        """routes: one (N, n_units) tensor per branch layer (needed for mode='route').
        internals=True returns (logits, hs, zs, raws) for the training rule and keys."""
        raise NotImplementedError

    # ---- training rule
    def substitution_loss(self, x: torch.Tensor, y: torch.Tensor, gate_weight: float = 0., gate_mode: str = 'full'):
        logits, hs, zs, raws = self(x, 'trace', internals=True)
        ce_ex = F.cross_entropy(logits, y, reduction='none'); ce = ce_ex.mean()
        grads = torch.autograd.grad(ce, hs, retain_graph=True); aux = 0; feedback = []
        for h, z, g in zip(hs, zs, grads):
            g = g.detach() * len(x); prod = g.unsqueeze(2) * (z - h.unsqueeze(2))
            if prod.dim() == 5: prod = prod.sum((3, 4))  # conv: sum over space
            aux = aux + prod.mean()
            with torch.no_grad(): feedback.append((ce_ex.detach()[:, None, None] + prod.detach()).mean(0))
        loss = 1.5 * ce + .5 * aux / len(hs)
        if gate_weight: loss = loss + gate_weight * self.gate_loss(raws, gate_mode)
        return loss, feedback

    def gate_loss(self, raws, mode: str = 'full') -> torch.Tensor:
        """Cross-entropy of the image-only gate softmax(|response|) towards the branch that is
        currently learning (lowest error trace), averaged over samples, neurons and layers.

        It raises route mass: the share of the test-time soft mixture on the route's branch.
        mode 'full': gradients reach every branch (selected one up, the others down).
        mode 'up'  : only the selected branch's |response| is pushed up; the others are
                     detached, so branches used by earlier tasks are not changed by this term."""
        total = 0.
        for layer, raw in zip(self.branch_layers(), raws):
            a = raw.abs()
            if a.dim() == 5: a = a.mean((3, 4))  # conv: same spatial average as soft mixing
            sel = layer.selected().expand(len(a), -1)[..., None]  # (N, n, 1)
            if mode == 'up':
                is_sel = torch.zeros_like(a, dtype=torch.bool).scatter_(2, sel, True); a = torch.where(is_sel, a, a.detach())
            total = total - a.log_softmax(-1).gather(2, sel).mean()
        return total / len(raws)

    @torch.no_grad()
    def route_mass(self, x, routes, chunk: int = 500) -> list[float]:
        """Per layer: mean soft weight on the route's branch, upstream layers mixing softly
        (image-only inference). routes: per layer (N, n_units). 1/B = blind, 1 = soft == route."""
        was = self.training; self.eval(); sums = [0.] * len(routes)
        for j in range(0, len(x), chunk):
            _, _, _, raws = self(x[j:j + chunk], 'soft', internals=True)
            for i, (raw, r) in enumerate(zip(raws, routes)):
                a = raw.abs()
                if a.dim() == 5: a = a.mean((3, 4))
                sums[i] += float(a.softmax(-1).gather(2, r[j:j + chunk, :, None]).mean(2).sum(0).mean())
        self.train(was); return [s / len(x) for s in sums]

    @torch.no_grad()
    def update_traces(self, feedback, decay: float = .95):
        for layer, f in zip(self.branch_layers(), feedback): layer.error.mul_(decay).add_(f, alpha=1 - decay)

    def current_route(self) -> list[torch.Tensor]:
        return [layer.selected().clone() for layer in self.branch_layers()]

    def empty_routes(self, n_tasks: int) -> list[torch.Tensor]:
        return [torch.zeros(n_tasks, layer.n_units, dtype=torch.long) for layer in self.branch_layers()]


class BranchMLP(BranchNetwork):
    """depth branch layers of 168 neurons x 4 branches, linear head.
    Layer 1: 196 random pixels per branch. Deeper layers: 42 of the previous 168 outputs."""

    def __init__(self, depth: int = 1, seed: int = 0, n: int = 168, B: int = 4, d_in: int = 784):
        """d_in: input width (784 pixels; other values, e.g. frozen backbone features, give each
        first-layer branch d_in // 4 random inputs)."""
        super().__init__(); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList([SparseBranchLayer(d_in, n, B, d_in // 4, g)] + [SparseBranchLayer(n, n, B, n // 4, g) for _ in range(depth - 1)])
        self.head = nn.Linear(n, 10)

    def branch_layers(self): return list(self.layers)

    def forward(self, x, mode='soft', routes=None, internals=False):
        h, hs, zs, raws = x, [], [], []
        for i, layer in enumerate(self.layers):
            raw = layer.responses(h); h, z = layer.mix(raw, mode, routes[i] if routes is not None else None)
            hs.append(h); zs.append(z); raws.append(raw)
        logits = self.head(h)
        return (logits, hs, zs, raws) if internals else logits


CNN_ARMS = {
    # name: (channels, branches per channel, fraction of input channels each branch sees)
    'conv_b4_sparse': ([16, 32, 64], 4, [1, .5, .5]),
    'conv_b4': ([16, 32, 64], 4, [1, 1, 1]),
    'conv_b1_wide': ([32, 64, 128], 1, [1, 1, 1]),  # ordinary convs, same parameter count
}


class BranchCNN(BranchNetwork):
    """Three stride-2 conv layers with branched channels -> LayerNorm -> 64x4 sparse
    branch layer (256 features per branch) -> linear head."""

    def __init__(self, arm: str = 'conv_b4_sparse', seed: int = 0):
        super().__init__(); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        ch, B, frac = CNN_ARMS[arm]
        self.convs = nn.ModuleList([ConvBranchLayer(a, b, B, f, g) for a, b, f in zip([3] + ch[:-1], ch, frac)])
        self.d = ch[-1] * 16; self.top = SparseBranchLayer(self.d, 64, 4, 256, g, bias='zeros'); self.head = nn.Linear(64, 10)

    def branch_layers(self): return [c for c in self.convs if c.B > 1] + [self.top]

    def forward(self, x, mode='soft', routes=None, internals=False):
        h = normalize_cifar(x); hs, zs, raws = [], [], []; i = 0
        for conv in self.convs:
            raw = conv.responses(h)
            h, z = conv.mix(raw, mode, routes[i] if (routes is not None and conv.B > 1) else None)
            if conv.B > 1: hs.append(h); zs.append(z); raws.append(raw); i += 1
        f = F.layer_norm(h.flatten(1), (self.d,)); raw = self.top.responses(f)
        h, z = self.top.mix(raw, mode, routes[i] if routes is not None else None); hs.append(h); zs.append(z); raws.append(raw)
        logits = self.head(h)
        return (logits, hs, zs, raws) if internals else logits
