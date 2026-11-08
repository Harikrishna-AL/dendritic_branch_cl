"""Training loop for HAT (official recipe: task-reset SGD, s annealed from 1/smax to smax over
each epoch, mask-sparsity regulariser lambda = .75, embedding-gradient compensation,
embedding clamp +-6, cumulative used-unit masks after each task)."""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F

from ..data import SplitData, task_of
from .trainer import ContinualRun, TrainConfig


class HATTrainer:
    def __init__(self, model, cfg: TrainConfig, seed: int, lr: float = .1, smax: float = 400., lamb: float = .75):
        self.model, self.cfg, self.seed, self.lr, self.smax, self.lamb = model, cfg, seed, lr, smax, lamb
        self.used = None  # cumulative masks of earlier tasks

    def _regularizer(self, masks):
        if self.used is None: return sum(m.sum() for m in masks) / sum(m.numel() for m in masks)
        free = [1 - u for u in self.used]
        return sum((m * f).sum() for m, f in zip(masks, free)) / sum(f.sum() for f in free).clamp_min(1e-8)

    def _step(self, opt, x, y, task, s):
        m = self.model; t = torch.full((len(x),), task, dtype=torch.long); opt.zero_grad(set_to_none=True)
        masks = [q[0] for q in m.masks(t[:1], s)]
        loss = F.cross_entropy(m(x, t, s), y) + self.lamb * self._regularizer(masks); loss.backward()
        with torch.no_grad():
            if self.used is not None:
                views = m.protection_views(self.used)
                for name, p in m.named_parameters():
                    if name in views and p.grad is not None: p.grad.mul_(1 - views[name])
            for e in m.embedding_parameters():
                if e.grad is not None: e.grad.mul_(self.smax / s * (torch.cosh((s * e).clamp(-50, 50)) + 1) / (torch.cosh(e) + 1))
        nn.utils.clip_grad_norm_(m.parameters(), self.cfg.clip); opt.step()
        with torch.no_grad():
            for e in m.embedding_parameters(): e.clamp_(-6, 6)

    def fit(self, data: SplitData, order: list[int]) -> ContinualRun:
        from .evaluation import predict_logits, summarize
        t0 = time.perf_counter(); run = ContinualRun(self.model, None); bs = self.cfg.batch_size
        for stage, task in enumerate(order):
            ix = torch.where(task_of(data.y) == task)[0]; opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            for epoch in range(self.cfg.epochs):
                g = torch.Generator().manual_seed(self.seed * self.cfg.shuffle_seed_scale + stage * 100 + epoch)
                perm = ix[torch.randperm(len(ix), generator=g)]
                for b, ids in enumerate(perm.split(bs)):
                    s = (self.smax - 1 / self.smax) * (b * bs / len(perm)) + 1 / self.smax
                    self._step(opt, data.x[ids], data.y[ids], task, s)
            with torch.no_grad():
                new = [q[0] for q in self.model.masks(torch.tensor([task]), self.smax)]
                self.used = new if self.used is None else [torch.maximum(u, n) for u, n in zip(self.used, new)]
            v = task_of(data.val_y) == task
            s_ = summarize(predict_hat(self.model, data.val_x[v], torch.full((int(v.sum()),), task)), data.val_y[v])
            run.acquisition[task] = dict(cil=s_['cil'], til=s_['til'])
        run.seconds = time.perf_counter() - t0; return run


@torch.no_grad()
def predict_hat(model, x, tasks, chunk=500):
    was = model.training; model.eval()
    out = torch.cat([model(x[j:j + chunk], tasks[j:j + chunk]) for j in range(0, len(x), chunk)]); model.train(was); return out
