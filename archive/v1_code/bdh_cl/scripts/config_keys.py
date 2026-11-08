"""Cheap test: can a stored 'key' per configuration pick the right branch route from one image?

At each task boundary (current task's training images only, weights current at that time;
no replay) store a Gaussian key for that task's configuration. At test time score the image
under every stored key, take the most likely task, and predict within that task's classes
using that task's archived route.

Keys (all per task):
  route_diag    diagonal Gaussian over the responses of the branches in that task's route
                (one per neuron/channel; MNIST 168 dims, CIFAR 176)
  branch_diag   diagonal Gaussian over ALL branch responses (MNIST 672, CIFAR 704)
  branch_pca    PCA-50 Gaussian (+ isotropic residual) over all branch responses
  randproj_diag diagonal Gaussian over the same number of random projections of pixels
                (equal-size control for branch_diag: does the learned branch code matter?)
  pixel_diag    diagonal Gaussian over pixels
  pixel_pca     PCA-50 Gaussian over pixels (SQDA-like global reference, per task not class)
Also: branch_diag_final (NOT CL-legal: keys recomputed with final weights; measures drift).
Responses are raw branch pre-activations (CIFAR conv: spatial mean); for CIFAR each candidate
route is applied at every layer, as when its key was recorded.
"""
import json, statistics, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F

K_PCA = 50


class Gauss:
    def __init__(self, X, kind):
        X = X.double(); self.kind = kind; self.mu = X.mean(0); C = X - self.mu
        if kind == 'diag':
            v = C.var(0); self.var = v + 1e-3 * v.mean()
        else:
            ev, U = torch.linalg.eigh(C.T @ C / (len(X) - 1)); ev, U = ev.flip(0), U.flip(1)
            k = min(K_PCA, X.shape[1] - 1); self.U = U[:, :k]; self.lam = ev[:k].clamp_min(1e-6)
            self.s2 = ev[k:].clamp_min(0).mean().clamp_min(1e-6) + 1e-3 * ev.mean(); self.lam = self.lam + 1e-3 * ev.mean(); self.D = X.shape[1]
    def loglik(self, X):
        C = X.double() - self.mu
        if self.kind == 'diag': return -.5 * ((C ** 2 / self.var).sum(1) + self.var.log().sum())
        P = C @ self.U; r = (C ** 2).sum(1) - (P ** 2).sum(1)
        return -.5 * ((P ** 2 / self.lam).sum(1) + self.lam.log().sum() + r / self.s2 + (self.D - len(self.lam)) * self.s2.log())


def keys_for(feats, kinds):
    return {k: Gauss(feats[src], kind) for k, (src, kind) in kinds.items()}


KINDS = {'route_diag': ('route', 'diag'), 'branch_diag': ('branch', 'diag'), 'branch_pca': ('branch', 'pca'),
         'randproj_diag': ('randproj', 'diag'), 'pixel_diag': ('pixel', 'diag'), 'pixel_pca': ('pixel', 'pca')}


def evaluate(keys, feats_of, route_logits_of, y, final_keys=None):
    """keys[t][name]; feats_of(t) -> dict of features under route t; route_logits_of(t) -> N x 10."""
    t_true = y // 2; res = {}
    L = [route_logits_of(t) for t in range(5)]; F_ = [feats_of(t) for t in range(5)]
    names = list(keys[0]) + (['branch_diag_final'] if final_keys else [])
    for name in names:
        if name == 'branch_diag_final': ll = torch.stack([final_keys[t].loglik(F_[t]['branch']) for t in range(5)], 1)
        else: ll = torch.stack([keys[t][name].loglik(F_[t][KINDS[name][0]]) for t in range(5)], 1)
        tk = ll.argmax(1); within = torch.stack([L[t][:, 2 * t:2 * t + 2].argmax(1) for t in range(5)], 1).gather(1, tk[:, None]).squeeze(1)
        res[name] = dict(taskid=float((tk == t_true).float().mean()), acc=float((2 * tk + within == y).float().mean()))
    return res


# ---------------- MNIST ----------------
def mnist(seed, first, x, y, tx, ty):
    from bdh_cl.scripts.mnist_combined_selector import Model
    from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
    from bdh_cl.scripts.three_routes_self_scoring import route_logits
    torch.manual_seed(seed); m = Model('baseline', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5))); routes = torch.zeros(5, 168, dtype=torch.uint8); step = 0
    Rp = torch.randn(784, 672, generator=torch.Generator().manual_seed(seed + 99)) / 28

    @torch.no_grad()
    def feats(xx, route):
        raw = torch.cat([m.responses(b, False) for b in xx.split(2048)])
        return dict(route=raw.gather(-1, route.long()[None, :, None].expand(len(xx), -1, 1)).squeeze(-1), branch=raw.flatten(1), randproj=xx @ Rp, pixel=xx)
    keys = {}
    for stage, pair in enumerate(order):
        ix = torch.where(y // 2 == pair)[0]
        for epoch in range(2):
            ordered = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch))]
            for ids in ordered.split(128):
                opt.zero_grad(set_to_none=True); loss, feedback, _ = objective(m, x[ids], y[ids], 'hybrid5', step)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); m.observe(feedback); step += 1
        routes[pair] = m.error.argmin(-1).to(torch.uint8); keys[pair] = keys_for(feats(x[ix], routes[pair]), KINDS)
    final = {t: Gauss(feats(x[y // 2 == t], routes[t])['branch'], 'diag') for t in range(5)}
    RL = route_logits(m, tx, routes)
    with torch.no_grad():
        raw = m.responses(tx, False); soft = m.head((raw.tanh() * raw.abs().softmax(-1)).sum(-1)).argmax(1)
    res = evaluate(keys, lambda t: feats(tx, routes[t]), lambda t: RL[t], ty, final)
    res['soft_magnitude'] = dict(acc=float((soft == ty).float().mean())); res['oracle'] = dict(acc=float((torch.stack([RL[t] for t in range(5)])[ty // 2, torch.arange(len(ty))].argmax(1) == ty).float().mean()), taskid=1.)
    return res


# ---------------- CIFAR ----------------
def cifar(seed, first, data):
    from bdh_cl.scripts import branched_conv_cifar as bc
    from bdh_cl.scripts.branched_conv_access import forward
    x, y, _, _, tx, ty = data; m = bc.Net('conv_b4_sparse', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5)))
    routes = [torch.zeros(5, L.error.shape[0], dtype=torch.long) for L in m.layers()]
    Rp = torch.randn(3072, 704, generator=torch.Generator().manual_seed(seed + 99)) / 3072 ** .5

    @torch.no_grad()
    def feats(xx, t):
        m.eval(); out = {k: [] for k in ['route', 'branch', 'randproj', 'pixel']}
        for b in xx.split(500):
            ws = [F.one_hot(r[t], 4).float()[None].expand(len(b), -1, -1) for r in routes]
            _, _, _, raws, _ = forward(m, b, ws, keep=True); R = [r.mean((3, 4)) if r.dim() == 5 else r for r in raws]
            out['route'].append(torch.cat([q.gather(-1, routes[l][t][None, :, None].expand(len(b), -1, 1)).squeeze(-1) for l, q in enumerate(R)], 1))
            out['branch'].append(torch.cat([q.flatten(1) for q in R], 1)); p = bc.norm(b).flatten(1); out['pixel'].append(p); out['randproj'].append(p @ Rp)
        m.train(); return {k: torch.cat(v) for k, v in out.items()}

    @torch.no_grad()
    def rl(xx, t):
        m.eval(); o = torch.cat([forward(m, b, [F.one_hot(r[t], 4).float()[None].expand(len(b), -1, -1) for r in routes]) for b in xx.split(500)]); m.train(); return o
    keys = {}
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 10000 + stage * 100 + epoch))]
            for ids in perm.split(64):
                opt.zero_grad(set_to_none=True); loss, fb = bc.train_step(m, x[ids], y[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                with torch.no_grad():
                    for L, f in zip(m.layers(), fb): L.error.mul_(.95).add_(f, alpha=.05)
        for i, L in enumerate(m.layers()): routes[i][task] = L.error.argmin(-1)
        keys[task] = keys_for(feats(x[ix], task), KINDS)
    final = {t: Gauss(feats(x[y // 2 == t], t)['branch'], 'diag') for t in range(5)}
    RL = [rl(tx, t) for t in range(5)]
    with torch.no_grad(): m.eval(); soft = torch.cat([forward(m, b).argmax(1) for b in tx.split(500)])
    res = evaluate(keys, lambda t: feats(tx, t), lambda t: RL[t], ty, final)
    res['soft_magnitude'] = dict(acc=float((soft == ty).float().mean())); res['oracle'] = dict(acc=float((torch.stack(RL)[ty // 2, torch.arange(len(ty))].argmax(1) == ty).float().mean()), taskid=1.)
    return res


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bench', required=True); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('results/config_keys_20261107')); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    runs = []
    if a.bench == 'mnist':
        from bdh_cl.data.mnist import load_raw
        from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
        raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
        def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
        x, y = ds(raw['train_images'][split['train']], raw['train_labels'][split['train']]); tx, ty = ds(raw['test_images'], raw['test_labels'])
        seeds, job = [101, 102, 103], lambda s, f: mnist(s, f, x, y, tx, ty)
    else:
        from bdh_cl.data import cifar10
        d = cifar10.load('data_cache/cifar10')
        data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
        data = [v.long() if i % 2 else v for i, v in enumerate(data)]; seeds, job = [201, 202, 203], lambda s, f: cifar(s, f, data)
    for seed in seeds:
        for first in [0, 4]:
            t0 = time.perf_counter(); r = job(seed, first); r.update(seed=seed, first=first); runs.append(r)
            (a.out / f'{a.bench}.json').write_text(json.dumps(runs, indent=2))
            print(a.bench, seed, first, f'{time.perf_counter() - t0:.0f}s', {k: (round(100 * v['acc'], 1), round(100 * v.get('taskid', float('nan')), 1)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    names = [k for k, v in runs[0].items() if isinstance(v, dict)]
    print(f'\n{a.bench}: test accuracy / task-ID accuracy, mean ± sd over seed means')
    for n in names:
        acc = [statistics.mean(r[n]['acc'] for r in runs if r['seed'] == s) * 100 for s in seeds]
        tid = [statistics.mean(r[n].get('taskid', float('nan')) for r in runs if r['seed'] == s) * 100 for s in seeds]
        print(f'{n:18s} {statistics.mean(acc):6.2f} ± {statistics.stdev(acc):4.2f}   task-ID {statistics.mean(tid):6.2f}')


if __name__ == '__main__':
    main()
