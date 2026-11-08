"""Ablation: how the training branch is chosen for each neuron.

  error_trace       the project's rule: lowest running substitution-error trace (default)
  fixed_random      one random branch per neuron, never changes (all tasks share it)
  random_per_batch  a fresh random branch per neuron at every batch
  most_active       the branch with the largest mean |response| on the current batch
  task_assigned     a random branch per neuron for each task, chosen with the task identity
                    (upper bound: perfect task-specific allocation, uses task ID in training)
Everything else (model, loss, optimiser, data order) is identical.
"""
from __future__ import annotations

import torch

RULES = ['error_trace', 'fixed_random', 'random_per_batch', 'most_active', 'task_assigned']


class GateRule:
    def __init__(self, name: str, seed: int):
        assert name in RULES, name
        self.name, self.g = name, torch.Generator().manual_seed(seed + 777)

    def _random(self, model):
        return [torch.randint(layer.B, (layer.n_units,), generator=self.g) for layer in model.branch_layers()]

    def start(self, model):
        if self.name == 'fixed_random':
            for layer, r in zip(model.branch_layers(), self._random(model)): layer.override = r
        if self.name == 'task_assigned': self.per_task = [self._random(model) for _ in range(5)]

    def task_start(self, model, task):
        if self.name == 'task_assigned':
            for layer, r in zip(model.branch_layers(), self.per_task[task]): layer.override = r

    @torch.no_grad()
    def before_step(self, model, x):
        if self.name == 'random_per_batch':
            for layer, r in zip(model.branch_layers(), self._random(model)): layer.override = r
        elif self.name == 'most_active':
            h = x
            for layer in model.branch_layers():  # layer by layer, so deeper layers see the chosen branches
                raw = layer.responses(h); layer.override = raw.tanh().abs().mean(0).argmax(-1); h, _ = layer.mix(raw, 'trace')

    @staticmethod
    def clear(model):
        for layer in model.branch_layers(): layer.override = None
