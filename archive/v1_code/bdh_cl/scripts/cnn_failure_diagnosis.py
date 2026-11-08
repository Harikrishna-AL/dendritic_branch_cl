"""Why does the branch idea fail on CIFAR-10 CNNs? Encoder-drift decomposition.

Split CIFAR-10 (5 two-class tasks), 2 epochs/task, Adam .0005, batch 64, clip 5,
3 seeds x 2 orders. All branch arms use the MNIST winning recipe: hard error-trace
(EMA .95) branch selection in training, CE + .5 single-neuron substitution CE,
soft-magnitude image-only inference; oracle = archived routes + true task (diagnostic).

Arms
  cnn_plain         ordinary CNN + tanh layer, CE only (no branches)
  cnn_learned       CNN trained on every task + sparse branch layer (previous design)
  cnn_freeze_after1 CNN trained on task 1 only, then frozen; branches keep learning
  cnn_random        random CNN never trained; only branches + head learn
  pixel_sparse      no CNN: MNIST recipe directly on pixels, 168x4 branches,
                    768 of 3072 pixels per branch (25%, as on MNIST)
"""
import json, math, statistics, argparse, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data import cifar10

MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]; STD = torch.tensor([.247, .2435, .2616])[None, :, None, None]
def norm(x): return (x.float() / 255 - MEAN) / STD


class Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__(); torch.manual_seed(seed); self.arm = arm
        if arm == 'pixel_sparse': n, b, d, k = 168, 4, 3072, 768
        elif arm == 'cnn_plain': n, b, d, k = 256, 1, 1024, 1024
        else: n, b, d, k = 64, 4, 1024, 256
        self.n, self.b, self.d = n, b, d
        if arm != 'pixel_sparse':
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())
            if arm == 'cnn_random': self.encoder.requires_grad_(False)
        self.weight = nn.Parameter(torch.empty(n * b, k)); nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)); self.bias = nn.Parameter(torch.zeros(n * b))
        g = torch.Generator().manual_seed(seed)
        if k < d: self.register_buffer('indices', torch.stack([torch.randperm(d, generator=g)[:k] for _ in range(n * b)]))
        self.head = nn.Linear(n, 10); self.register_buffer('error', torch.full((n, b), math.log(10)))

    def feats(self, x):
        if self.arm == 'pixel_sparse': return norm(x).reshape(len(x), -1)
        return F.layer_norm(self.encoder(norm(x)), (1024,))

    def responses(self, x):
        h = self.feats(x)
        z = (h[:, self.indices] * self.weight).sum(-1) + self.bias if hasattr(self, 'indices') else F.linear(h, self.weight, self.bias)
        return z.reshape(-1, self.n, self.b)

    def logits(self, x, mode, routes=None):
        raw = self.responses(x); z = raw.tanh()
        if mode == 'soft': w = raw.abs().softmax(-1)
        elif mode == 'hard': w = F.one_hot(raw.abs().argmax(-1), self.b).float()
        elif mode == 'oracle': w = F.one_hot(routes, self.b).float()
        else: w = F.one_hot(self.error.argmin(-1), self.b).float()[None].expand(len(x), -1, -1)
        return self.head((z * w).sum(-1))


@torch.no_grad()
def evaluate(m, x, y, routes=None):
    m.eval(); out = {k: [] for k in ['soft', 'hard', 'oracle']}
    for ids in torch.arange(len(y)).split(256):
        bx = x[ids]; out['soft'].append(m.logits(bx, 'soft').argmax(1)); out['hard'].append(m.logits(bx, 'hard').argmax(1))
        if routes is not None: out['oracle'].append(m.logits(bx, 'oracle', routes[y[ids] // 2]).argmax(1))
    res = {}
    for k, v in out.items():
        if not v: continue
        p = torch.cat(v); res[k] = dict(acc=float((p == y).float().mean()), tasks=[float((p == y)[y // 2 == t].float().mean()) for t in range(5)])
    m.train(); return res


def run(arm, seed, order, data):
    x, y, vx, vy, tx, ty = data; m = Net(arm, seed); opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.0005)
    routes = torch.zeros(5, m.n, dtype=torch.long); acquisition = {}; t0 = time.perf_counter()
    for stage, task in enumerate(order):
        if arm == 'cnn_freeze_after1' and stage == 1:
            m.encoder.requires_grad_(False); opt = torch.optim.Adam([p for n_, p in m.named_parameters() if p.requires_grad and not n_.startswith('encoder')], lr=.0005)
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 10000 + stage * 100 + epoch))]
            for ids in perm.split(64):
                bx, by = x[ids], y[ids]; opt.zero_grad(set_to_none=True)
                if arm == 'cnn_plain':
                    loss = F.cross_entropy(m.logits(bx, 'trace'), by)
                else:
                    raw = m.responses(bx); z = raw.tanh(); gate = F.one_hot(m.error.argmin(-1), m.b).float()
                    h = (z * gate).sum(-1); lg = m.head(h)
                    alt = lg[:, None, None, :] + (z - h[:, :, None])[:, :, :, None] * m.head.weight.T[None, :, None, :]
                    err = F.cross_entropy(alt.reshape(-1, 10), by[:, None, None].expand(alt.shape[:-1]).reshape(-1), reduction='none').reshape(alt.shape[:-1])
                    loss = F.cross_entropy(lg, by) + .5 * err.mean()
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                if arm != 'cnn_plain':
                    with torch.no_grad(): m.error.mul_(.95).add_(err.detach().mean(0), alpha=.05)
        routes[task] = m.error.argmin(-1)
        # acquisition: accuracy on the just-learned task under the training route, all 10 classes eligible
        vix = vy // 2 == task
        with torch.no_grad(): acquisition[task] = float((m.logits(vx[vix], 'trace').argmax(1) == vy[vix]).float().mean())
    secs = time.perf_counter() - t0
    return m, routes, dict(arm=arm, seed=seed, order=order, acquisition=[acquisition[t] for t in range(5)], val=evaluate(m, vx, vy, routes if arm != 'cnn_plain' else None),
                           test=evaluate(m, tx, ty, routes if arm != 'cnn_plain' else None), seconds=secs, trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/cnn_failure_diagnosis_20261106')); p.add_argument('--arms', default='cnn_plain,cnn_learned,cnn_freeze_after1,cnn_random,pixel_sparse'); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    d = cifar10.load('data_cache/cifar10')
    data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
    data = [v.long() if i % 2 else v for i, v in enumerate(data)]
    path = a.out / 'runs.json'; runs = json.loads(path.read_text()) if path.exists() else []
    done = {(r['arm'], r['seed'], tuple(r['order'])) for r in runs}
    for arm in a.arms.split(','):
        for seed in [201, 202, 203]:
            for order in [list(range(5)), list(range(4, -1, -1))]:
                if (arm, seed, tuple(order)) in done: continue
                m, routes, r = run(arm, seed, order, data); runs.append(r); path.write_text(json.dumps(runs, indent=2))
                print(arm, seed, order[0], 'acq', [round(v * 100) for v in r['acquisition']], 'soft', round(r['test']['soft']['acc'] * 100, 2), 'oracle', round(r['test'].get('oracle', {}).get('acc', float('nan')) * 100, 2), f"{r['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
