"""Metrics.

CIL: argmax over all 10 classes (image only).
TIL: argmax restricted to the true task's two classes (task given at test).
For branch networks, 'soft' uses no task information; 'route' restores the true task's
stored route (diagnostic: separates retention from access).
"""
from __future__ import annotations

import itertools
import statistics

import torch

from ..data import N_TASKS, task_of


@torch.no_grad()
def predict_logits(model, x, mode='soft', routes=None, tasks=None, chunk=500):
    """routes: per-layer (n_tasks, n_units) tensors; tasks: per-sample task index used to pick the route."""
    was_training = model.training; model.eval(); out = []
    for j in range(0, len(x), chunk):
        r = None if routes is None else [q[tasks[j:j + chunk]] for q in routes]
        out.append(model(x[j:j + chunk], mode, r))
    model.train(was_training); return torch.cat(out)


def til_predictions(logits, y):
    t = task_of(y); idx = torch.arange(len(y))
    pair = torch.stack([logits[idx, 2 * t], logits[idx, 2 * t + 1]], 1); return 2 * t + pair.argmax(1)


def summarize(logits, y) -> dict:
    t = task_of(y); cil = logits.argmax(1) == y; til = til_predictions(logits, y) == y
    return dict(cil=float(cil.float().mean()), til=float(til.float().mean()),
                cil_tasks=[float(cil[t == k].float().mean()) for k in range(N_TASKS)],
                til_tasks=[float(til[t == k].float().mean()) for k in range(N_TASKS)])


def evaluate(model, x, y, routes=None) -> dict:
    if not getattr(model, 'is_branch', False):
        return {'plain': summarize(predict_logits(model, x, None), y)}
    out = {'soft': summarize(predict_logits(model, x, 'soft'), y)}
    if routes is not None: out['route'] = summarize(predict_logits(model, x, 'route', routes, task_of(y)), y)
    return out


def route_disagreement(routes) -> list[float]:
    """Per layer: fraction of neurons whose branch differs between two task routes (chance = 1 - 1/B)."""
    return [statistics.mean(float((r[i] != r[j]).float().mean()) for i, j in itertools.combinations(range(len(r)), 2)) for r in routes]
