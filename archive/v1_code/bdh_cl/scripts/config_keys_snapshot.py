"""Drift-free branch keys: freeze a copy of the network at each task end and compute that task's key
AND its test-time features through the frozen copy. Separates 'is the learned branch code a good key?'
from 'do keys go stale?'. Memory cost: one weight snapshot per task (diagnostic, not a proposed method).
MNIST: snapshot of the first (branch) layer. CIFAR: snapshot of the whole branched CNN."""
import copy, json, statistics, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from bdh_cl.scripts.config_keys import Gauss

NAMES = {'route_snap_diag': ('route', 'diag'), 'branch_snap_diag': ('branch', 'diag'), 'branch_snap_pca': ('branch', 'pca')}


def score(keys, test_feats, RL, y):
    t_true = y // 2; res = {}
    within = torch.stack([RL[t][:, 2 * t:2 * t + 2].argmax(1) for t in range(5)], 1)
    for n, (src, _) in NAMES.items():
        ll = torch.stack([keys[t][n].loglik(test_feats[t][src]) for t in range(5)], 1); tk = ll.argmax(1)
        res[n] = dict(taskid=float((tk == t_true).float().mean()), acc=float((2 * tk + within.gather(1, tk[:, None]).squeeze(1) == y).float().mean()))
    return res


def mnist(seed, first, x, y, tx, ty):
    from bdh_cl.scripts.mnist_combined_selector import Model
    from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
    from bdh_cl.scripts.three_routes_self_scoring import route_logits
    torch.manual_seed(seed); m = Model('baseline', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5))); routes = torch.zeros(5, 168, dtype=torch.uint8); step = 0
    @torch.no_grad()
    def feats(net, xx, route):
        raw = torch.cat([net.responses(b, False) for b in xx.split(2048)])
        return dict(route=raw.gather(-1, route.long()[None, :, None].expand(len(xx), -1, 1)).squeeze(-1), branch=raw.flatten(1))
    keys, tf = {}, {}
    for stage, pair in enumerate(order):
        ix = torch.where(y // 2 == pair)[0]
        for epoch in range(2):
            ordered = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch))]
            for ids in ordered.split(128):
                opt.zero_grad(set_to_none=True); loss, feedback, _ = objective(m, x[ids], y[ids], 'hybrid5', step)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); m.observe(feedback); step += 1
        routes[pair] = m.error.argmin(-1).to(torch.uint8); snap = copy.deepcopy(m)
        f = feats(snap, x[ix], routes[pair]); keys[pair] = {n: Gauss(f[s], k) for n, (s, k) in NAMES.items()}; tf[pair] = feats(snap, tx, routes[pair])
    RL = route_logits(m, tx, routes); return score(keys, tf, RL, ty)


def cifar(seed, first, data):
    from bdh_cl.scripts import branched_conv_cifar as bc
    from bdh_cl.scripts.branched_conv_access import forward
    x, y, _, _, tx, ty = data; m = bc.Net('conv_b4_sparse', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5)))
    routes = [torch.zeros(5, L.error.shape[0], dtype=torch.long) for L in m.layers()]
    @torch.no_grad()
    def feats(net, xx, t):
        net.eval(); out = {'route': [], 'branch': []}
        for b in xx.split(500):
            ws = [F.one_hot(r[t], 4).float()[None].expand(len(b), -1, -1) for r in routes]
            _, _, _, raws, _ = forward(net, b, ws, keep=True); R = [r.mean((3, 4)) if r.dim() == 5 else r for r in raws]
            out['route'].append(torch.cat([q.gather(-1, routes[l][t][None, :, None].expand(len(b), -1, 1)).squeeze(-1) for l, q in enumerate(R)], 1)); out['branch'].append(torch.cat([q.flatten(1) for q in R], 1))
        return {k: torch.cat(v) for k, v in out.items()}
    keys, tf = {}, {}
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 10000 + stage * 100 + epoch))]
            for ids in perm.split(64):
                opt.zero_grad(set_to_none=True); loss, fb = bc.train_step(m, x[ids], y[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                with torch.no_grad():
                    for L, f in zip(m.layers(), fb): L.error.mul_(.95).add_(f, alpha=.05)
        for i, L in enumerate(m.layers()): routes[i][task] = L.error.argmin(-1)
        snap = copy.deepcopy(m); f = feats(snap, x[ix], task); keys[task] = {n: Gauss(f[s], k) for n, (s, k) in NAMES.items()}; tf[task] = feats(snap, tx, task)
    with torch.no_grad():
        m.eval(); RL = [torch.cat([forward(m, b, [F.one_hot(r[t], 4).float()[None].expand(len(b), -1, -1) for r in routes]) for b in tx.split(500)]) for t in range(5)]
    return score(keys, tf, RL, ty)


def main():
    from bdh_cl.scripts.config_keys import main as _  # noqa: ensure module importable
    p = argparse.ArgumentParser(); p.add_argument('--bench', required=True); p.add_argument('--threads', type=int, default=4); a = p.parse_args()
    out = Path('results/config_keys_20261107'); torch.set_num_threads(a.threads); runs = []
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
            t0 = time.perf_counter(); r = job(seed, first); r.update(seed=seed, first=first); runs.append(r); (out / f'{a.bench}_snapshot.json').write_text(json.dumps(runs, indent=2))
            print(a.bench, 'snap', seed, first, f'{time.perf_counter() - t0:.0f}s', {k: (round(100 * v['acc'], 1), round(100 * v['taskid'], 1)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    print(f'\n{a.bench} snapshot keys: test accuracy / task-ID')
    for n in NAMES:
        acc = [statistics.mean(r[n]['acc'] for r in runs if r['seed'] == s) * 100 for s in seeds]; tid = [statistics.mean(r[n]['taskid'] for r in runs if r['seed'] == s) * 100 for s in seeds]
        print(f'{n:18s} {statistics.mean(acc):6.2f} ± {statistics.stdev(acc):4.2f}   task-ID {statistics.mean(tid):6.2f}', flush=True)


if __name__ == '__main__':
    main()
