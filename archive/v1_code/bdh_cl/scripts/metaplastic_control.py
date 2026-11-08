"""Control from Cho et al. 2025 (Neural Networks 107762): random intermixed per-synapse stability.

Each weight gets a fixed random flexibility f ~ U(0,1). Its update is scaled by
S = 1 - tanh^2(alpha * (1-f)/f * |dw|), dw = (weight after previous task) - (initial weight),
refreshed at each task boundary (their Eq. 1-2). Implemented on the Adam step:
w <- w_old + S * (w_adam - w_old). No replay, no task ID in training.

Question: does branch-structured protection beat random weight-level protection in a
parameter-matched ordinary network?
  MNIST: branch = depth-1 multilayer_branch_recipe (168x4 branches, 196 px); mlp = tanh MLP
         width-matched (~134k params); mlp_meta(alpha).
  CIFAR: branch = conv_b4_sparse (branched_conv_cifar); cnn = ordinary tanh CNN 32/64/128 +
         dense 32 + head (~159k params); cnn_meta(alpha).
Metrics (all arms): CIL = image-only, all 10 classes. TIL = argmax restricted to the true
pair (uses task ID at test). Branch arms also report oracle (true-task route) CIL/TIL.
Retention loss = mean over the first 4 learned tasks of (TIL right after learning it) -
(final TIL), validation. For branch the final TIL uses the oracle route.
alpha is selected on validation CIL (also reported: alpha selected on validation TIL loss).
"""
import json, math, statistics, argparse, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F

ALPHAS = [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]


class Meta:
    def __init__(self, model, alpha, seed):
        g = torch.Generator().manual_seed(seed + 555); self.ps = [p for p in model.parameters()]
        self.f = [torch.rand(p.shape, generator=g).clamp(1e-3, 1) for p in self.ps]; self.init = [p.detach().clone() for p in self.ps]
        self.alpha = alpha; self.boundary()
    def boundary(self): self.dw = [(p.detach() - w0).abs() for p, w0 in zip(self.ps, self.init)]
    def before(self): self.old = [p.detach().clone() for p in self.ps]
    @torch.no_grad()
    def after(self):
        for p, o, f, d in zip(self.ps, self.old, self.f, self.dw):
            S = 1 - torch.tanh(self.alpha * (1 - f) / f * d) ** 2; p.copy_(o + S * (p - o))
    def frozen_frac(self, thr=.05):
        n = sum(p.numel() for p in self.ps)
        return sum(float((1 - torch.tanh(self.alpha * (1 - f) / f * d) ** 2 < thr).sum()) for f, d in zip(self.f, self.dw)) / n


class CNN(nn.Module):
    def __init__(self, seed):
        super().__init__(); torch.manual_seed(seed)
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1) for a, b in zip([3, 32, 64], [32, 64, 128])]); self.fc = nn.Linear(2048, 32); self.head = nn.Linear(32, 10)
    def forward(self, x, mode=None, routes=None):
        from bdh_cl.scripts.branched_conv_cifar import norm
        h = norm(x)
        for c in self.convs: h = c(h).tanh()
        return self.head(self.fc(F.layer_norm(h.flatten(1), (2048,))).tanh())


def til_pred(lg, y):
    t = y // 2; pair = torch.stack([lg[torch.arange(len(y)), 2 * t], lg[torch.arange(len(y)), 2 * t + 1]], 1); return 2 * t + pair.argmax(1)


@torch.no_grad()
def predict(m, x, mode, routes_for=None, y=None):
    out = []
    for j in range(0, len(x), 500):
        r = None if routes_for is None else [q[y[j:j + 500] // 2] for q in routes_for]
        out.append(m(x[j:j + 500], mode, r))
    return torch.cat(out)


def metrics(m, x, y, branch, routes):
    m.eval(); res = {}
    modes = [('soft', None), ('oracle', routes)] if branch else [('plain', None)]
    for mode, R in modes:
        lg = predict(m, x, mode, R, y); res[mode] = dict(cil=float((lg.argmax(1) == y).float().mean()), til=float((til_pred(lg, y) == y).float().mean()),
                                                         til_tasks=[float((til_pred(lg, y) == y)[y // 2 == t].float().mean()) for t in range(5)])
    m.train(); return res


def run(bench, arm, alpha, seed, first, data):
    x, y, vx, vy, tx, ty = data; order = list(range(5)) if first == 0 else list(range(4, -1, -1))
    branch = arm == 'branch'
    if bench == 'mnist':
        from bdh_cl.scripts import multilayer_branch_recipe as mr
        m = mr.BranchNet(1, seed) if branch else mr.MLP(1, mr.mlp_width(1, sum(p.numel() for p in mr.BranchNet(1, seed).parameters())), seed); bs = 128
        layers = m.layers if branch else None
    else:
        from bdh_cl.scripts import branched_conv_cifar as bc
        m = bc.Net('conv_b4_sparse', seed) if branch else CNN(seed); bs = 64; layers = m.layers() if branch else None
    opt = torch.optim.Adam(m.parameters(), lr=.0005); meta = Meta(m, alpha, seed) if alpha else None
    routes = [torch.zeros(5, L.error.shape[0], dtype=torch.long) for L in layers] if branch else None; acq = {}
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        if meta: meta.boundary()
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * (100000 if bench == 'mnist' else 10000) + stage * 100 + epoch))]
            for ids in perm.split(bs):
                opt.zero_grad(set_to_none=True)
                if branch: loss, fb = (m.train_step(x[ids], y[ids]) if bench == 'mnist' else bc.train_step(m, x[ids], y[ids]))
                else: loss = F.cross_entropy(m(x[ids]), y[ids])
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5)
                if meta: meta.before()
                opt.step()
                if meta: meta.after()
                if branch:
                    with torch.no_grad():
                        for L, f in zip(layers, fb): L.error.mul_(.95).add_(f, alpha=.05)
        if branch:
            for i, L in enumerate(layers): routes[i][task] = L.error.argmin(-1)
        v = vy // 2 == task
        with torch.no_grad(): m.eval(); lg = predict(m, vx[v], 'trace' if branch else None); m.train()
        acq[task] = float((til_pred(lg, vy[v]) == vy[v]).float().mean())
    val, test = metrics(m, vx, vy, branch, routes), metrics(m, tx, ty, branch, routes)
    key = 'oracle' if branch else 'plain'
    loss = statistics.mean(acq[t] - val[key]['til_tasks'][t] for t in order[:4])
    return dict(bench=bench, arm=arm, alpha=alpha, seed=seed, first=first, params=sum(p.numel() for p in m.parameters()), acq_til=[acq[t] for t in range(5)],
                val=val, test=test, til_retention_loss_val=loss, frozen_frac_end=meta.frozen_frac() if meta else None)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bench', required=True); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('results/metaplastic_control_20261107')); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads); torch.set_flush_denormal(True)
    if a.bench == 'mnist':
        from bdh_cl.data.mnist import load_raw
        from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
        raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
        def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
        data = [*ds(raw['train_images'][split['train']], raw['train_labels'][split['train']]), *ds(raw['train_images'][split['validation']], raw['train_labels'][split['validation']]), *ds(raw['test_images'], raw['test_labels'])]
        seeds, plain = [101, 102, 103], 'mlp'
    else:
        from bdh_cl.data import cifar10
        d = cifar10.load('data_cache/cifar10')
        data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
        data = [v.long() if i % 2 else v for i, v in enumerate(data)]; seeds, plain = [201, 202, 203], 'cnn'
    path = a.out / f'{a.bench}.json'; runs = json.loads(path.read_text()) if path.exists() else []
    done = {(r['arm'], r['alpha'], r['seed'], r['first']) for r in runs}
    for arm, alpha in [('branch', 0), (plain, 0)] + [(plain, al) for al in ALPHAS]:
        for seed in seeds:
            for first in [0, 4]:
                if (arm, alpha, seed, first) in done: continue
                t0 = time.perf_counter(); r = run(a.bench, arm, alpha, seed, first, data); runs.append(r); path.write_text(json.dumps(runs, indent=2))
                k = 'soft' if arm == 'branch' else 'plain'
                print(a.bench, arm, alpha, seed, first, r['params'], 'CIL', round(100 * r['test'][k]['cil'], 2), 'TIL', round(100 * r['test'][k]['til'], 2),
                      'oracleTIL', round(100 * r['test'].get('oracle', {}).get('til', float('nan')), 2), 'loss', round(100 * r['til_retention_loss_val'], 1),
                      'frozen', r['frozen_frac_end'], f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
