"""Short-lived branch state over an unlabelled test stream (depth-1 BranchMLP).

Images arrive one at a time. Every method is causal: predict, then update its state.

  Stateless            soft-magnitude mixing, no state
  ClassPriorState      EMA of predicted class probabilities added to the logits (non-branch control)
  BranchState          EMA of per-neuron branch preferences; the gate becomes
                       (1 - s) * current + s * state. Preferences come from the substitution
                       error of each branch against the model's own prediction ('pseudo')
                       or against the true label ('supervised', upper-bound control)
Optional persistence gate (c0): the state's strength is scaled by an EMA of whether
consecutive predictions fall in the same task, so it switches off in mixed streams.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.nn import functional as F

from ..data import N_TASKS

STREAMS = ['blocked', 'geometric', 'mixed']


# ------------------------------------------------------------------ stream orders
def stream_order(y: np.ndarray, seed: int, kind: str) -> np.ndarray:
    """blocked: task blocks of 200 in round-robin; geometric: random block lengths (mean 50); mixed: iid."""
    if kind == 'mixed': return np.random.default_rng(seed).permutation(len(y))
    if kind == 'blocked':
        g = np.random.default_rng(seed); queues = [list(g.permutation(np.flatnonzero(y // 2 == t))) for t in range(N_TASKS)]; out = []
        while any(queues):
            for q in queues:
                if q: out.extend(q[:200]); del q[:200]
        return np.array(out)
    g = np.random.default_rng(seed + 7); queues = [list(g.permutation(np.flatnonzero(y // 2 == t))) for t in range(N_TASKS)]; out = []
    while any(queues):
        t = g.choice([i for i in range(N_TASKS) if queues[i]]); n = int(g.geometric(1 / 50)); out += queues[t][:n]; del queues[t][:n]
    return np.array(out)


# ------------------------------------------------------------------ per-image cache
@torch.no_grad()
def stream_cache(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Branch outputs z, current soft gate, and branch preferences, per image (numpy)."""
    layer = model.layers[0]; W = model.head.weight; out = {k: [] for k in ['z', 'current', 'pseudo', 'supervised', 'logits']}
    for bx, by in zip(x.split(256), y.split(256)):
        raw = layer.responses(bx); z = raw.tanh(); cur = raw.abs().softmax(-1); h = (z * cur).sum(-1); logits = model.head(h)
        alt = logits[:, None, None, :] + (z - h[:, :, None])[..., None] * W.T[None, :, None, :]  # logits if a branch replaced the neuron
        for name, target in [('pseudo', logits.argmax(-1)), ('supervised', by)]:
            err = F.cross_entropy(alt.reshape(-1, 10), target[:, None, None].expand(alt.shape[:-1]).reshape(-1), reduction='none').reshape(alt.shape[:-1])
            c = err - err.mean(-1, keepdim=True); out[name].append((-c / c.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)).softmax(-1).numpy())
        out['z'].append(z.numpy()); out['current'].append(cur.numpy()); out['logits'].append(logits.numpy())
    c = {k: np.concatenate(v) for k, v in out.items()}; c['W'] = W.numpy(); c['b'] = model.head.bias.numpy(); c['y'] = y.numpy(); return c


def _softmax(v): e = np.exp(v - v.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class PersistenceGate:
    """Strength multiplier in [0, 1]: high when consecutive predictions stay in one task."""

    def __init__(self, c0: float | None, decay: float = .9):
        self.c0, self.decay, self.per, self.prev = c0, decay, None, None

    def __call__(self, logits) -> float:
        if self.c0 is None: return 1.
        task = int(logits.argmax()) // 2
        if self.prev is not None:
            same = float(task == self.prev); self.per = same if self.per is None else self.decay * self.per + (1 - self.decay) * same
        self.prev = task
        return 0. if self.per is None else float(np.clip((self.per - self.c0) / (1 - self.c0), 0, 1))


class Stateless:
    name = 'none'

    def run(self, c, order): return c['logits'][order].argmax(-1)

    @staticmethod
    def grid(): yield {}


class ClassPriorState:
    def __init__(self, decay, strength, c0=None): self.decay, self.strength, self.c0 = decay, strength, c0

    def run(self, c, order):
        st, gate, pred = None, PersistenceGate(self.c0), np.empty(len(order), int)
        for i, ix in enumerate(order):
            lg = c['logits'][ix]; s = self.strength * gate(lg)
            adj = lg if st is None else lg + s * np.log(st + 1e-4); pred[i] = adj.argmax()
            p = _softmax(adj); st = p if st is None else self.decay * st + (1 - self.decay) * p
        return pred

    @staticmethod
    def grid(gated):
        if gated:
            for d, s, c0 in itertools.product([.8, .9, .95], [.3, .5, .7, 1., 1.5], [.3, .5, .7]): yield dict(decay=d, strength=s, c0=c0)
        else:
            for d, s in itertools.product([.8, .9, .95], [.3, .5, .7]): yield dict(decay=d, strength=s)


class BranchState:
    def __init__(self, decay, strength, c0=None, feedback='pseudo'):
        self.decay, self.strength, self.c0, self.feedback = decay, strength, c0, feedback

    def run(self, c, order):
        st, gate, pred = None, PersistenceGate(self.c0), np.empty(len(order), int)
        for i, ix in enumerate(order):
            cur = c['current'][ix]; s = self.strength * gate(c['logits'][ix])
            g = cur if st is None else (1 - s) * cur + s * st
            pred[i] = int((c['W'] @ (c['z'][ix] * g).sum(-1) + c['b']).argmax())
            o = c[self.feedback][ix]; st = o.copy() if st is None else self.decay * st + (1 - self.decay) * o  # update after predicting
        return pred

    @staticmethod
    def grid(gated):
        for d, s in itertools.product([.9, .99], [.5, .9]):
            if gated:
                for c0 in [.3, .5, .7]: yield dict(decay=d, strength=s, c0=c0)
            else: yield dict(decay=d, strength=s)


METHODS = {
    'none': (lambda **k: Stateless(), lambda: Stateless.grid()),
    'class_prior': (ClassPriorState, lambda: ClassPriorState.grid(False)),
    'class_prior_gated': (ClassPriorState, lambda: ClassPriorState.grid(True)),
    'branch_state': (BranchState, lambda: BranchState.grid(False)),
    'branch_state_gated': (BranchState, lambda: BranchState.grid(True)),
    'branch_state_supervised': (lambda **k: BranchState(feedback='supervised', **k), lambda: BranchState.grid(False)),
}
