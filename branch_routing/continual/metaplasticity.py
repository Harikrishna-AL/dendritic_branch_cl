"""Control: random intermixed per-synapse stability (Cho, Lee, Baek & Paik, Neural Networks 2025).

Each weight gets a fixed random flexibility f ~ U(0, 1). Its update is scaled by
S = 1 - tanh^2(alpha * (1 - f) / f * |dw|), where dw = (weight after the previous task)
- (initial weight), refreshed at each task boundary. Applied to the optimiser step:
w <- w_old + S * (w_new - w_old).
"""
from __future__ import annotations

import torch


class Metaplasticity:
    def __init__(self, model: torch.nn.Module, alpha: float, seed: int):
        g = torch.Generator().manual_seed(seed + 555); self.params = list(model.parameters()); self.alpha = alpha
        self.f = [torch.rand(p.shape, generator=g).clamp(1e-3, 1) for p in self.params]
        self.init = [p.detach().clone() for p in self.params]; self.task_boundary()

    def task_boundary(self):
        self.dw = [(p.detach() - w0).abs() for p, w0 in zip(self.params, self.init)]

    def _stability(self, f, dw):
        return 1 - torch.tanh(self.alpha * (1 - f) / f * dw) ** 2

    def before_step(self):
        self.old = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def after_step(self):
        for p, old, f, dw in zip(self.params, self.old, self.f, self.dw): p.copy_(old + self._stability(f, dw) * (p - old))

    def frozen_fraction(self, threshold: float = .05) -> float:
        n = sum(p.numel() for p in self.params)
        return sum(float((self._stability(f, dw) < threshold).sum()) for f, dw in zip(self.f, self.dw)) / n
