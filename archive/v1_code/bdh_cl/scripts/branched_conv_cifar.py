"""Branches INSIDE the conv layers: does restoring routes at every layer stop CNN forgetting?

Split CIFAR-10, 5 two-class tasks, from scratch, 2 epochs/task, Adam .0005, batch 64,
clip 5, seeds 201-203 x forward/reverse. Every conv layer: each output channel has B
filter sets (branches) -> tanh; training uses a hard per-channel branch chosen by an
EMA(.95) of estimated substitution loss; image-only inference mixes branches by
softmax(spatial-mean |response|); oracle restores the archived route at every layer.
Top: LayerNorm(flatten) -> sparse 64x4 branch layer (256 features/branch) -> head.
Substitution estimate (all layers): CE + sum_hw g.(z_branch - h), g = per-example
autograd gradient of CE w.r.t. the layer output. Loss = 1.5 CE + .5 mean_layers aux.

Arms
  conv_b4         branched convs 16/32/64 channels x 4 branches
  conv_b4_sparse  as conv_b4, layer 2-3 branches each see a random 50% of input channels
  conv_b1_wide    ordinary (1-branch) tanh convs 32/64/128 channels: same conv parameter
                  count (~93k), same top branch layer -> capacity control
"""
import json, math, statistics, itertools, argparse, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data import cifar10

MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]; STD = torch.tensor([.247, .2435, .2616])[None, :, None, None]
def norm(x): return (x.float() / 255 - MEAN) / STD


class BranchConv(nn.Module):
    def __init__(self, cin, cout, B, frac, gen):
        super().__init__(); self.c, self.B = cout, B
        self.conv = nn.Conv2d(cin, cout * B, 3, 2, 1)
        if frac < 1:
            k = math.ceil(frac * cin); m = torch.zeros(cout * B, cin, 1, 1)
            for o in range(cout * B): m[o, torch.randperm(cin, generator=gen)[:k]] = 1
            self.register_buffer('mask', m)
        self.register_buffer('error', torch.full((cout, B), math.log(10)))
    def raw(self, x):
        w = self.conv.weight * self.mask if hasattr(self, 'mask') else self.conv.weight
        r = F.conv2d(x, w, self.conv.bias, 2, 1); return r.reshape(len(x), self.c, self.B, *r.shape[-2:])
    def mix(self, r, mode, route=None):
        z = r.tanh()
        if self.B == 1: return z[:, :, 0], z
        if mode == 'soft': w = r.abs().mean((3, 4)).softmax(-1)[..., None, None]
        elif mode == 'oracle': w = F.one_hot(route, self.B).float()[..., None, None]
        else: w = F.one_hot(self.error.argmin(-1), self.B).float()[None, :, :, None, None]
        return (z * w).sum(2), z


class BranchDense(nn.Module):
    def __init__(self, d, gen, n=64, B=4, k=256):
        super().__init__(); self.n, self.B = n, B
        self.weight = nn.Parameter(torch.empty(n * B, k)); nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)); self.bias = nn.Parameter(torch.zeros(n * B))
        self.register_buffer('indices', torch.stack([torch.randperm(d, generator=gen)[:k] for _ in range(n * B)]))
        self.register_buffer('error', torch.full((n, B), math.log(10)))
    def raw(self, x): return ((x[:, self.indices] * self.weight).sum(-1) + self.bias).reshape(-1, self.n, self.B)
    def mix(self, r, mode, route=None):
        z = r.tanh()
        if mode == 'soft': w = r.abs().softmax(-1)
        elif mode == 'oracle': w = F.one_hot(route, self.B).float()
        else: w = F.one_hot(self.error.argmin(-1), self.B).float()[None]
        return (z * w).sum(-1), z


class Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__(); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        if arm == 'conv_b1_wide': ch, B, fr = [32, 64, 128], 1, [1, 1, 1]
        elif arm == 'conv_b4_sparse': ch, B, fr = [16, 32, 64], 4, [1, .5, .5]
        else: ch, B, fr = [16, 32, 64], 4, [1, 1, 1]
        self.convs = nn.ModuleList([BranchConv(a, b, B, f, g) for a, b, f in zip([3] + ch[:-1], ch, fr)])
        self.d = ch[-1] * 16; self.top = BranchDense(self.d, g); self.head = nn.Linear(64, 10)
    def layers(self): return [L for L in self.convs if L.B > 1] + [self.top]
    def forward(self, x, mode, routes=None, keep=False):
        h = norm(x); hs, zs = [], []; i = 0
        for L in self.convs:
            route = routes[i] if (routes is not None and L.B > 1) else None
            h, z = L.mix(L.raw(h), mode, route)
            if L.B > 1: hs.append(h); zs.append(z); i += 1
        f = F.layer_norm(h.flatten(1), (self.d,)); h, z = self.top.mix(self.top.raw(f), mode, routes[i] if routes is not None else None)
        hs.append(h); zs.append(z); lg = self.head(h)
        return (lg, hs, zs) if keep else lg


def train_step(m, x, y):
    lg, hs, zs = m(x, 'trace', keep=True); ce_ex = F.cross_entropy(lg, y, reduction='none'); ce = ce_ex.mean()
    gs = torch.autograd.grad(ce, hs, retain_graph=True); aux = 0; fb = []
    for h, z, g in zip(hs, zs, gs):
        g = g.detach() * len(x); delta = z - h.unsqueeze(2); prod = g.unsqueeze(2) * delta
        if prod.dim() == 5: prod = prod.sum((3, 4))
        aux = aux + prod.mean()
        with torch.no_grad(): fb.append((ce_ex.detach()[:, None, None] + prod.detach()).mean(0))
    return 1.5 * ce + .5 * aux / len(hs), fb


@torch.no_grad()
def evaluate(m, x, y, routes):
    m.eval(); out = {'soft': [], 'oracle': []}
    for ids in torch.arange(len(y)).split(250):
        out['soft'].append(m(x[ids], 'soft').argmax(1))
        out['oracle'].append(m(x[ids], 'oracle', [r[y[ids] // 2] for r in routes]).argmax(1))
    m.train(); res = {}
    for k, v in out.items():
        p = torch.cat(v); res[k] = dict(acc=float((p == y).float().mean()), tasks=[float((p == y)[y // 2 == t].float().mean()) for t in range(5)])
    return res


def run(arm, seed, order, data):
    x, y, vx, vy, tx, ty = data; m = Net(arm, seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    routes = [torch.zeros(5, L.error.shape[0], dtype=torch.long) for L in m.layers()]; acq = {}; t0 = time.perf_counter()
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 10000 + stage * 100 + epoch))]
            for ids in perm.split(64):
                opt.zero_grad(set_to_none=True); loss, fb = train_step(m, x[ids], y[ids]); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                with torch.no_grad():
                    for L, f in zip(m.layers(), fb): L.error.mul_(.95).add_(f, alpha=.05)
        for i, L in enumerate(m.layers()): routes[i][task] = L.error.argmin(-1)
        v = vy // 2 == task
        with torch.no_grad(): m.eval(); acq[task] = float(torch.cat([(m(vx[v][j:j + 250], 'trace').argmax(1)) for j in range(0, int(v.sum()), 250)]).eq(vy[v]).float().mean()); m.train()
    disagree = [statistics.mean(float((r[i] != r[j]).float().mean()) for i, j in itertools.combinations(range(5), 2)) for r in routes]
    return dict(arm=arm, seed=seed, order=order, params=sum(p.numel() for p in m.parameters()), acquisition=[acq[t] for t in range(5)],
                route_disagreement_per_layer=disagree, val=evaluate(m, vx, vy, routes), test=evaluate(m, tx, ty, routes), seconds=time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', required=True); p.add_argument('--threads', type=int, default=3)
    p.add_argument('--out', type=Path, default=Path('results/branched_conv_cifar_20261106')); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    d = cifar10.load('data_cache/cifar10')
    data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
    data = [v.long() if i % 2 else v for i, v in enumerate(data)]; runs = []
    for seed in [201, 202, 203]:
        for order in [list(range(5)), list(range(4, -1, -1))]:
            r = run(a.arm, seed, order, data); runs.append(r); (a.out / f'{a.arm}.json').write_text(json.dumps(runs, indent=2))
            print(a.arm, seed, order[0], r['params'], 'acq', [round(v * 100) for v in r['acquisition']], 'soft', round(r['test']['soft']['acc'] * 100, 2), 'oracle', round(r['test']['oracle']['acc'] * 100, 2),
                  'disagree', [round(v * 100) for v in r['route_disagreement_per_layer']], f"{r['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
