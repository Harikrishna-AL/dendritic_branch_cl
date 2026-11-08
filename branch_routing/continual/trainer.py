"""Task-incremental training loop shared by every experiment.

Tasks arrive one at a time (known boundaries, no replay). For branch networks the
per-neuron winning branches are stored as the task's route at each boundary.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..data import N_TASKS, SplitData, task_of
from .evaluation import evaluate, predict_logits, route_disagreement, summarize
from .metaplasticity import Metaplasticity


@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 5e-4
    batch_size: int = 128
    clip: float = 5.
    trace_decay: float = .95
    shuffle_seed_scale: int = 100000  # batch order seed = seed * scale + stage * 100 + epoch


BENCH_CONFIG = {'mnist': TrainConfig(batch_size=128, shuffle_seed_scale=100000),
                'cifar10': TrainConfig(batch_size=64, shuffle_seed_scale=10000)}


@dataclass
class ContinualRun:
    model: nn.Module
    routes: list | None
    acquisition: dict = field(default_factory=dict)  # task -> {'cil', 'til'} on validation, right after learning it
    seconds: float = 0.


class ContinualTrainer:
    """Train ``model`` on the tasks of ``data`` in ``order``.

    metaplasticity_alpha > 0 wraps the optimiser step with the Cho et al. control.
    on_task_end(task, trainer) is called after each task (e.g. to fit task keys).
    gate_rule (ablation only) replaces error-trace branch selection, see gate_rules.py.
    """

    def __init__(self, model: nn.Module, cfg: TrainConfig, seed: int, metaplasticity_alpha: float = 0,
                 on_task_end: Callable | None = None, gate_rule=None):
        self.model, self.cfg, self.seed = model, cfg, seed
        self.branch = getattr(model, 'is_branch', False)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.meta = Metaplasticity(model, metaplasticity_alpha, seed) if metaplasticity_alpha else None
        self.on_task_end = on_task_end
        self.routes = model.empty_routes(N_TASKS) if self.branch else None
        self.gate_rule = gate_rule
        if gate_rule: gate_rule.start(model)

    def _batches(self, ix, stage, epoch):
        g = torch.Generator().manual_seed(self.seed * self.cfg.shuffle_seed_scale + stage * 100 + epoch)
        return ix[torch.randperm(len(ix), generator=g)].split(self.cfg.batch_size)

    def train_step(self, x, y):
        m = self.model; self.opt.zero_grad(set_to_none=True)
        if self.gate_rule: self.gate_rule.before_step(m, x)
        if self.branch: loss, feedback = m.substitution_loss(x, y)
        else: loss = F.cross_entropy(m(x), y)
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), self.cfg.clip)
        if self.meta: self.meta.before_step()
        self.opt.step()
        if self.meta: self.meta.after_step()
        if self.branch: m.update_traces(feedback, self.cfg.trace_decay)
        return loss.item()

    def fit(self, data: SplitData, order: list[int]) -> ContinualRun:
        t0 = time.perf_counter(); run = ContinualRun(self.model, self.routes)
        for stage, task in enumerate(order):
            ix = torch.where(task_of(data.y) == task)[0]
            if self.meta: self.meta.task_boundary()
            if self.gate_rule: self.gate_rule.task_start(self.model, task)
            for epoch in range(self.cfg.epochs):
                for ids in self._batches(ix, stage, epoch): self.train_step(data.x[ids], data.y[ids])
            if self.branch:
                for layer_routes, r in zip(self.routes, self.model.current_route()): layer_routes[task] = r
            v = task_of(data.val_y) == task
            s = summarize(predict_logits(self.model, data.val_x[v], 'trace' if self.branch else None), data.val_y[v])
            run.acquisition[task] = dict(cil=s['cil'], til=s['til'])
            if self.on_task_end: self.on_task_end(task, self)
        run.seconds = time.perf_counter() - t0; return run


def run_record(run: ContinualRun, data: SplitData, order: list[int], meta: Metaplasticity | None = None, **info) -> dict:
    """Evaluate a finished run on validation and test and return a JSON-ready record."""
    m = run.model
    if getattr(m, 'uses_task', False):  # HAT: the true task's mask is the 'route'
        from .hat import predict_hat
        val, test = ({'route': summarize(predict_hat(m, x, task_of(y)), y)} for x, y in [(data.val_x, data.val_y), (data.test_x, data.test_y)])
    else:
        val = evaluate(m, data.val_x, data.val_y, run.routes); test = evaluate(m, data.test_x, data.test_y, run.routes)
    key = 'route' if 'route' in val else 'plain'
    forgetting = statistics.mean(run.acquisition[t]['til'] - val[key]['til_tasks'][t] for t in order[:-1])
    return dict(**info, order=order, params=sum(p.numel() for p in m.parameters()),
                acquisition_til=[run.acquisition[t]['til'] for t in range(N_TASKS)], acquisition_cil=[run.acquisition[t]['cil'] for t in range(N_TASKS)],
                val=val, test=test, forgetting_til_val=forgetting, seconds=run.seconds,
                route_disagreement=route_disagreement(run.routes) if run.routes is not None else None,
                frozen_fraction=meta.frozen_fraction() if meta else None)
