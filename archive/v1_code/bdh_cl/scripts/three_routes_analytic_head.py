"""Route 1: replay-free online analytic (ridge) head on hybrid5 branch features.

Retrains hybrid5 exactly as mnist_gradient_branch_scoring.fit, plus a no-grad pass
over the CURRENT task's training images at each task boundary that accumulates
ridge sufficient statistics G += Phi^T Phi, C += Phi^T Y with the weights current
at that time. No old images are revisited; statistics of old tasks are therefore
computed on stale (earlier) features. At the end the head is solved once.
"""
import json, statistics, argparse
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective

LAMBDAS = [1e-3, 1e-2, 1e-1, 1., 10., 100.]


@torch.no_grad()
def feats(m, x):
    out = {'mix168': [], 'all672': []}
    for bx in x.split(1024):
        raw = m.responses(bx, False); z = raw.tanh()
        out['mix168'].append((z * raw.abs().softmax(-1)).sum(-1).double())
        out['all672'].append(z.reshape(len(bx), -1).double())
    return {k: torch.cat([torch.cat(v), torch.ones(len(x), 1, dtype=torch.float64)], 1) for k, v in out.items()}


def solve(G, C, lam):
    reg = lam * torch.eye(len(G), dtype=G.dtype); reg[-1, -1] = 0  # bias unregularised
    return torch.linalg.solve(G + reg, C)


def fit(seed, first, x, y):
    torch.manual_seed(seed); m = Model('baseline', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5)))
    routes = torch.zeros(5, 168, dtype=torch.uint8); step = 0
    stats = {k: [torch.zeros(d, d, dtype=torch.float64), torch.zeros(d, 10, dtype=torch.float64)] for k, d in [('mix168', 169), ('all672', 673)]}
    for stage, pair in enumerate(order):
        ix = torch.where(y // 2 == pair)[0]
        for epoch in range(2):
            ordered = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch))]
            for ids in ordered.split(128):
                opt.zero_grad(set_to_none=True); loss, feedback, _ = objective(m, x[ids], y[ids], 'hybrid5', step)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); m.observe(feedback); step += 1
        routes[pair] = m.error.argmin(-1).to(torch.uint8)
        f = feats(m, x[ix]); Y = F.one_hot(y[ix], 10).double()
        for k in stats: stats[k][0] += f[k].T @ f[k]; stats[k][1] += f[k].T @ Y
    return m, routes, stats


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/three_routes_20261106/analytic_head')); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
    def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
    x, y = ds(raw['train_images'][split['train']], raw['train_labels'][split['train']])
    vx, vy = ds(raw['train_images'][split['validation']], raw['train_labels'][split['validation']]); tx, ty = ds(raw['test_images'], raw['test_labels'])
    runs = []
    for seed in [101, 102, 103]:
        for first in [0, 4]:
            m, routes, stats = fit(seed, first, x, y)
            saved = torch.load(Path('results/mnist_gradient_branch_scoring_20261106') / f'hybrid5_{seed}_{first}.pt', weights_only=True)
            same = all(torch.equal(saved['state'][k], v) for k, v in m.state_dict().items())
            fv, ft = feats(m, vx), feats(m, tx)
            with torch.no_grad():
                base_v = float((m.head((fv['mix168'][:, :-1]).float()).argmax(-1) == vy).float().mean())
                base_t = float((m.head((ft['mix168'][:, :-1]).float()).argmax(-1) == ty).float().mean())
            r = dict(seed=seed, first=first, reproduces_checkpoint=same, learned_head=dict(val=base_v, test=base_t), ridge={})
            for k in stats:
                for lam in LAMBDAS:
                    W = solve(*stats[k], lam)
                    r['ridge'][f'{k}|{lam}'] = dict(val=float(((fv[k] @ W).argmax(-1) == vy).double().mean()), test=float(((ft[k] @ W).argmax(-1) == ty).double().mean()),
                                                   test_pairs=[float(((ft[k] @ W).argmax(-1) == ty)[ty // 2 == g].double().mean()) for g in range(5)])
            runs.append(r); print(seed, first, 'reproduces', same, 'learned', round(base_t * 100, 2), flush=True)
            torch.save(dict(stats={k: [t.float() for t in v] for k, v in stats.items()}), a.out / f'stats_{seed}_{first}.pt')
    (a.out / 'runs.json').write_text(json.dumps(runs, indent=2))
    def seedmean(f): return [statistics.mean(f(r) for r in runs if r['seed'] == s) * 100 for s in [101, 102, 103]]
    summary = {'learned_head': {s: seedmean(lambda r: r['learned_head'][s]) for s in ['val', 'test']}}
    for k in ['mix168', 'all672']:
        lam = max(LAMBDAS, key=lambda l: statistics.mean(r['ridge'][f'{k}|{l}']['val'] for r in runs))
        summary[k] = dict(lambda_=lam, val=seedmean(lambda r: r['ridge'][f'{k}|{lam}']['val']), test=seedmean(lambda r: r['ridge'][f'{k}|{lam}']['test']),
                          test_pairs=[statistics.mean(r['ridge'][f'{k}|{lam}']['test_pairs'][g] for r in runs) * 100 for g in range(5)],
                          paired_gain_test=[t - b for t, b in zip(seedmean(lambda r: r['ridge'][f'{k}|{lam}']['test']), summary['learned_head']['test'])],
                          stat_bytes_float32=4 * (len(stats[k][0]) ** 2 + len(stats[k][0]) * 10))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if k == 'learned_head': print('learned head test', round(statistics.mean(v['test']), 2), '±', round(statistics.stdev(v['test']), 2))
        else: print(k, 'lambda', v['lambda_'], 'val', round(statistics.mean(v['val']), 2), 'test', round(statistics.mean(v['test']), 2), '±', round(statistics.stdev(v['test']), 2), 'gain', [round(g, 2) for g in v['paired_gain_test']], 'pairs', [round(q, 1) for q in v['test_pairs']], 'bytes', v['stat_bytes_float32'])


if __name__ == '__main__':
    main()
