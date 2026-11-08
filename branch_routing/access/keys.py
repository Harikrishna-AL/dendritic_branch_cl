"""Per-task keys: pick a task's stored route from one image.

At the end of each task a Gaussian key is fitted on that task's training images only,
with the weights current at that time (no stored data). At test time the image is
scored under every task's key; the most likely task's route then predicts within that
task's two classes.
"""
from __future__ import annotations

import torch

from ..data import N_TASKS, normalize_cifar, task_of


class GaussianKey:
    """'diag': diagonal Gaussian. 'pca': top-k principal directions + isotropic residual."""

    def __init__(self, X: torch.Tensor, kind: str = 'pca', k: int = 50):
        X = X.double(); self.kind = kind; self.mu = X.mean(0); C = X - self.mu
        if kind == 'diag':
            v = C.var(0); self.var = v + 1e-3 * v.mean()
        else:
            ev, U = torch.linalg.eigh(C.T @ C / (len(X) - 1)); ev, U = ev.flip(0), U.flip(1)
            k = min(k, X.shape[1] - 1); self.U = U[:, :k]; self.D = X.shape[1]
            self.s2 = ev[k:].clamp_min(0).mean().clamp_min(1e-6) + 1e-3 * ev.mean()
            self.lam = ev[:k].clamp_min(1e-6) + 1e-3 * ev.mean()

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        C = X.double() - self.mu
        if self.kind == 'diag': return -.5 * ((C ** 2 / self.var).sum(1) + self.var.log().sum())
        P = C @ self.U; r = (C ** 2).sum(1) - (P ** 2).sum(1)
        return -.5 * ((P ** 2 / self.lam).sum(1) + self.lam.log().sum() + r / self.s2 + (self.D - len(self.lam)) * self.s2.log())


# name: (feature source, Gaussian kind)
KEY_KINDS = {'pixel_pca': ('pixel', 'pca'), 'pixel_diag': ('pixel', 'diag'), 'randproj_diag': ('randproj', 'diag'),
             'branch_diag': ('branch', 'diag'), 'branch_pca': ('branch', 'pca')}


class KeyFeatures:
    """Features a key can be built on, for a branch network and one task's route."""

    def __init__(self, model, bench: str, seed: int):
        self.model, self.bench = model, bench
        D = 784 if bench == 'mnist' else 3072
        n_branch = sum(layer.n_units * layer.B for layer in model.branch_layers())
        self.R = torch.randn(D, n_branch, generator=torch.Generator().manual_seed(seed + 99)) / D ** .5

    def pixels(self, x):
        return x if self.bench == 'mnist' else normalize_cifar(x).flatten(1)

    @torch.no_grad()
    def __call__(self, x, routes, task, chunk=500) -> dict[str, torch.Tensor]:
        m = self.model; was = m.training; m.eval(); out = {'pixel': [], 'randproj': [], 'branch': []}
        for b in x.split(chunk):
            r = [q[task].expand(len(b), -1) for q in routes]
            _, _, _, raws = m(b, 'route', r, internals=True)
            out['branch'].append(torch.cat([(q.mean((3, 4)) if q.dim() == 5 else q).flatten(1) for q in raws], 1))
            p = self.pixels(b); out['pixel'].append(p); out['randproj'].append(p @ self.R)
        m.train(was); return {k: torch.cat(v) for k, v in out.items()}


class TaskKeyRouter:
    """Stores one set of keys per task and routes test images by maximum likelihood."""

    def __init__(self, features: KeyFeatures, kinds=KEY_KINDS):
        self.features, self.kinds, self.keys = features, kinds, {}

    def fit_task(self, task, x_task, routes):
        f = self.features(x_task, routes, task)
        self.keys[task] = {name: GaussianKey(f[src], kind) for name, (src, kind) in self.kinds.items()}

    @torch.no_grad()
    def evaluate(self, x, y, routes, refit_data=None) -> dict:
        """refit_data=(x_train, y_train): also report branch_diag keys refit with the FINAL
        weights (not continual-learning legal; measures how much keys drift)."""
        m = self.features.model; feats = [self.features(x, routes, t) for t in range(N_TASKS)]
        route_logits = []
        for t in range(N_TASKS):
            m.eval(); route_logits.append(torch.cat([m(b, 'route', [q[t].expand(len(b), -1) for q in routes]) for b in x.split(500)])); m.train()
        within = torch.stack([route_logits[t][:, 2 * t:2 * t + 2].argmax(1) + 2 * t for t in range(N_TASKS)], 1)
        scores = {name: torch.stack([self.keys[t][name].log_likelihood(feats[t][src]) for t in range(N_TASKS)], 1)
                  for name, (src, _) in self.kinds.items()}
        if refit_data is not None:
            xt, yt = refit_data
            refit = [GaussianKey(self.features(xt[task_of(yt) == t], routes, t)['branch'], 'diag') for t in range(N_TASKS)]
            scores['branch_diag_refit'] = torch.stack([refit[t].log_likelihood(feats[t]['branch']) for t in range(N_TASKS)], 1)
        res = {}
        for name, ll in scores.items():
            tk = ll.argmax(1); pred = within.gather(1, tk[:, None]).squeeze(1)
            res[name] = dict(taskid=float((tk == task_of(y)).float().mean()), cil=float((pred == y).float().mean()))
        true = within.gather(1, task_of(y)[:, None]).squeeze(1)
        res['true_task'] = dict(taskid=1., cil=float((true == y).float().mean()))
        return res


@torch.no_grad()
def pixel_key_tasks(data, bench: str) -> dict[str, torch.Tensor]:
    """Task chosen for every validation/test image by per-task PCA-50 pixel keys.

    Each key uses only its own task's training pixels. Pixels never change, so fitting the
    keys at each task's end or after training gives the same keys (no stored data needed)."""
    pix = (lambda x: x) if bench == 'mnist' else (lambda x: normalize_cifar(x).flatten(1))
    keys = [GaussianKey(pix(data.x[task_of(data.y) == t]), 'pca') for t in range(N_TASKS)]
    return {split: torch.stack([k.log_likelihood(pix(x)) for k in keys], 1).argmax(1)
            for split, x in [('val', data.val_x), ('test', data.test_x)]}


def routed_accuracy(logits: torch.Tensor, tasks: torch.Tensor, y: torch.Tensor) -> float:
    """Predict within the two classes of the chosen task; logits were computed for that task."""
    idx = torch.arange(len(y)); pair = torch.stack([logits[idx, 2 * tasks], logits[idx, 2 * tasks + 1]], 1)
    return float((2 * tasks + pair.argmax(1) == y).float().mean())
