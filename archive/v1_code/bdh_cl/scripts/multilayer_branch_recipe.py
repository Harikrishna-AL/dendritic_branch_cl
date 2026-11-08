"""Does the current winning MNIST recipe survive stacking branch layers?

Split MNIST, same split/normalisation as hybrid5; 2 epochs/task, Adam .0005, batch 128,
clip 5; 3 seeds x 2 orders. Branch depth L in {1,2,3}:
  layer 1: 168 neurons x 4 tanh branches, 196 random pixels per branch
  layer l>1: 168 x 4 branches, 42 of the 168 previous outputs per branch (25%)
Training (every layer): hard branch per neuron = argmin of EMA(.95) estimated
substitution loss; estimate = CE + g.(z_branch - h) with g the per-example autograd
gradient of CE w.r.t. that layer's output (first-order, as hybrid5's linear arm);
loss = 1.5 CE + .5 mean_layers mean(g_detached * delta)  (hybrid5 linear surrogate).
Inference: soft magnitude mix at every layer. Oracle: archived routes at all layers.
Controls: ordinary tanh MLP of the same depth, width chosen to match parameters, CE only.
"""
import json, math, statistics, argparse, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE


class BranchLayer(nn.Module):
    def __init__(self, d_in, k, gen, n=168, b=4):
        super().__init__(); self.n, self.b = n, b
        self.weight = nn.Parameter(torch.empty(n * b, k)); nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(k); self.bias = nn.Parameter(torch.empty(n * b).uniform_(-bound, bound))
        self.register_buffer('indices', torch.stack([torch.randperm(d_in, generator=gen)[:k] for _ in range(n * b)]))
        self.register_buffer('error', torch.full((n, b), math.log(10)))
    def responses(self, x): return ((x[:, self.indices] * self.weight).sum(-1) + self.bias).reshape(-1, self.n, self.b)


class BranchNet(nn.Module):
    def __init__(self, depth, seed):
        super().__init__(); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList([BranchLayer(784, 196, g)] + [BranchLayer(168, 42, g) for _ in range(depth - 1)]); self.head = nn.Linear(168, 10)
    def forward(self, x, mode, routes=None):
        h = x
        for i, L in enumerate(self.layers):
            raw = L.responses(h); z = raw.tanh()
            w = raw.abs().softmax(-1) if mode == 'soft' else F.one_hot(routes[i] if mode == 'oracle' else L.error.argmin(-1), 4).float()
            h = (z * w).sum(-1)
        return self.head(h)
    def train_step(self, x, y):
        hs, zs, h = [], [], x
        for L in self.layers:
            z = L.responses(h).tanh(); h = (z * F.one_hot(L.error.argmin(-1), 4).float()).sum(-1); hs.append(h); zs.append(z)
        lg = self.head(h); ce_ex = F.cross_entropy(lg, y, reduction='none'); ce = ce_ex.mean()
        gs = torch.autograd.grad(ce, hs, retain_graph=True); B = len(x); aux = 0; feedback = []
        for z, h, g in zip(zs, hs, gs):
            g = g.detach() * B; delta = z - h[:, :, None]
            aux = aux + (g[:, :, None] * delta).mean()
            feedback.append((ce_ex.detach()[:, None, None] + g[:, :, None] * delta.detach()).mean(0))
        return 1.5 * ce + .5 * aux / len(self.layers), feedback


class MLP(nn.Module):
    def __init__(self, depth, width, seed):
        super().__init__(); torch.manual_seed(seed); dims = [784] + [width] * depth
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:])]); self.head = nn.Linear(width, 10)
    def forward(self, x, mode=None, routes=None):
        for L in self.layers: x = L(x).tanh()
        return self.head(x)


def mlp_width(depth, target):
    return min(range(16, 400), key=lambda w: abs(785 * w + (depth - 1) * (w * w + w) + 11 * w - target))


@torch.no_grad()
def evaluate(m, x, y, routes=None):
    out = {}
    for mode in (['soft', 'oracle'] if routes is not None else ['plain']):
        p = torch.cat([m(bx, mode, [r[by // 2] for r in routes] if mode == 'oracle' else None).argmax(-1) for bx, by in zip(x.split(1000), y.split(1000))])
        out[mode] = dict(acc=float((p == y).float().mean()), tasks=[float((p == y)[y // 2 == t].float().mean()) for t in range(5)])
    return out


def run(kind, depth, seed, first, data):
    x, y, vx, vy, tx, ty = data; order = list(range(5)) if first == 0 else list(range(4, -1, -1))
    if kind == 'branch': m = BranchNet(depth, seed)
    else: m = MLP(depth, mlp_width(depth, sum(p.numel() for p in BranchNet(depth, seed).parameters())), seed)
    opt = torch.optim.Adam(m.parameters(), lr=.0005); routes = [torch.zeros(5, 168, dtype=torch.long) for _ in range(depth)] if kind == 'branch' else None
    acq = {}; t0 = time.perf_counter()
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch))]
            for ids in perm.split(128):
                opt.zero_grad(set_to_none=True)
                if kind == 'branch': loss, fb = m.train_step(x[ids], y[ids])
                else: loss = F.cross_entropy(m(x[ids]), y[ids])
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                if kind == 'branch':
                    with torch.no_grad():
                        for L, f in zip(m.layers, fb): L.error.mul_(.95).add_(f, alpha=.05)
        if kind == 'branch':
            for i, L in enumerate(m.layers): routes[i][task] = L.error.argmin(-1)
        v = vy // 2 == task
        with torch.no_grad(): acq[task] = float((m(vx[v], 'trace' if kind == 'branch' else None).argmax(-1) == vy[v]).float().mean())
    return dict(kind=kind, depth=depth, seed=seed, first=first, params=sum(p.numel() for p in m.parameters()), acquisition=[acq[t] for t in range(5)],
                val=evaluate(m, vx, vy, routes), test=evaluate(m, tx, ty, routes), seconds=time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/multilayer_branch_recipe_20261106')); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(3)
    raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
    def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
    data = [*ds(raw['train_images'][split['train']], raw['train_labels'][split['train']]), *ds(raw['train_images'][split['validation']], raw['train_labels'][split['validation']]), *ds(raw['test_images'], raw['test_labels'])]
    runs = []
    for depth in [1, 2, 3]:
        for kind in ['branch', 'mlp']:
            for seed in [101, 102, 103]:
                for first in [0, 4]:
                    r = run(kind, depth, seed, first, data); runs.append(r); (a.out / 'runs.json').write_text(json.dumps(runs, indent=2))
                    key = 'soft' if kind == 'branch' else 'plain'
                    print(kind, depth, seed, first, r['params'], 'acq', [round(v * 100) for v in r['acquisition']], 'test', round(r['test'][key]['acc'] * 100, 2), 'oracle', round(r['test'].get('oracle', {}).get('acc', float('nan')) * 100, 2), f"{r['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
