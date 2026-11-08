"""Route 2: task inference by letting each archived branch configuration score itself.

Each task t archived a hard route (one branch per neuron) at its boundary, and the
learner knows which classes it trained in task t (training-time knowledge, not a
test-time task ID). For a test image, run all 5 routes, score how strongly route t
claims the image for ITS OWN classes, pick the best task, predict within it.
Optional per-route normalisation uses own-task TRAINING scores recorded at that
task's boundary (current data only; weights current at that time).
"""
import json, statistics, argparse
from pathlib import Path
import numpy as np, torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective

SCORES = ['maxlogit', 'energy', 'taskprob', 'margin_all']


@torch.no_grad()
def route_logits(m, x, routes, upto=None):
    out = []
    for bx in x.split(1024):
        z = m.responses(bx, False).tanh()
        out.append(torch.stack([m.head(z.gather(-1, routes[t].long()[None, :, None].expand(len(bx), -1, 1)).squeeze(-1)) for t in range(5)]))
    return torch.cat(out, 1)  # 5 x N x 10


def score(L, kind):
    """L: 5 x N x 10 logits. Returns 5 x N score of route t for its own classes 2t,2t+1."""
    idx = torch.arange(5)
    own = torch.stack([L[t, :, 2 * t:2 * t + 2] for t in range(5)])  # 5 x N x 2
    if kind == 'maxlogit': return own.max(-1).values
    if kind == 'energy': return own.logsumexp(-1)
    p = L.softmax(-1); ownp = torch.stack([p[t, :, 2 * t:2 * t + 2] for t in range(5)])
    if kind == 'taskprob': return ownp.sum(-1).clamp_min(1e-9).log()
    if kind == 'margin_all':  # own-class best logit minus best other-class logit under the same route
        other = L.clone()
        for t in range(5): other[t, :, 2 * t:2 * t + 2] = -1e9
        return own.max(-1).values - other.max(-1).values


def predict(L, s):
    task = s.argmax(0); n = torch.arange(L.shape[1])
    own = L[task, n].gather(-1, torch.stack([2 * task, 2 * task + 1], -1))
    return 2 * task + own.argmax(-1), task


def fit(seed, first, x, y):
    torch.manual_seed(seed); m = Model('baseline', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    order = list(range(5)) if first == 0 else list(reversed(range(5)))
    routes = torch.zeros(5, 168, dtype=torch.uint8); step = 0; norm = {k: torch.zeros(5, 2) for k in SCORES}
    for stage, pair in enumerate(order):
        ix = torch.where(y // 2 == pair)[0]
        for epoch in range(2):
            ordered = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch))]
            for ids in ordered.split(128):
                opt.zero_grad(set_to_none=True); loss, feedback, _ = objective(m, x[ids], y[ids], 'hybrid5', step)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); m.observe(feedback); step += 1
        routes[pair] = m.error.argmin(-1).to(torch.uint8)
        # Own-task score statistics for the just-archived route, current-task data only.
        L = route_logits(m, x[ix], routes)
        for k in SCORES:
            s = score(L, k)[pair]; norm[k][pair] = torch.tensor([s.mean(), s.std()])
    return m, routes, norm


def evaluate(m, routes, norm, x, y):
    L = route_logits(m, x, routes); out = {}
    for k in SCORES:
        s = score(L, k)
        for normed in [False, True]:
            ss = (s - norm[k][:, :1]) / norm[k][:, 1:].clamp_min(1e-6) if normed else s
            pred, task = predict(L, ss)
            out[f'{k}|{"norm" if normed else "raw"}'] = dict(acc=float((pred == y).float().mean()), task_acc=float((task == y // 2).float().mean()))
    oracle_pred = 2 * (y // 2) + L[y // 2, torch.arange(len(y))].gather(-1, torch.stack([2 * (y // 2), 2 * (y // 2) + 1], -1)).argmax(-1)
    out['oracle_task_within'] = dict(acc=float((oracle_pred == y).float().mean()), task_acc=1.0)
    return out, L


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/three_routes_20261106/self_scoring')); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
    def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
    x, y = ds(raw['train_images'][split['train']], raw['train_labels'][split['train']])
    vx, vy = ds(raw['train_images'][split['validation']], raw['train_labels'][split['validation']]); tx, ty = ds(raw['test_images'], raw['test_labels'])
    runs = []
    for seed in [101, 102, 103]:
        for first in [0, 4]:
            m, routes, norm = fit(seed, first, x, y)
            saved = torch.load(Path('results/mnist_gradient_branch_scoring_20261106') / f'hybrid5_{seed}_{first}.pt', weights_only=True)
            assert all(torch.equal(saved['state'][k], v) for k, v in m.state_dict().items()) and torch.equal(saved['routes'], routes)
            v, _ = evaluate(m, routes, norm, vx, vy); t, Lt = evaluate(m, routes, norm, tx, ty)
            torch.save(dict(norm=norm, routes=routes, test_logits=Lt.half(), val_logits=route_logits(m, vx, routes).half()), a.out / f'route_logits_{seed}_{first}.pt')
            runs.append(dict(seed=seed, first=first, val=v, test=t)); print(seed, first, {k: round(t[k]['acc'] * 100, 1) for k in t}, flush=True)
    (a.out / 'runs.json').write_text(json.dumps(runs, indent=2))
    keys = list(runs[0]['val'])
    def sm(k, split, f='acc'): return [statistics.mean(r[split][k][f] for r in runs if r['seed'] == s) * 100 for s in [101, 102, 103]]
    print('\nvalidation / test (seed-mean of order-means)')
    for k in keys: print(f'{k:22s} val {statistics.mean(sm(k, "val")):6.2f}  test {statistics.mean(sm(k, "test")):6.2f} ± {statistics.stdev(sm(k, "test")):.2f}  task-acc test {statistics.mean(sm(k, "test", "task_acc")):6.2f}')
    best = max([k for k in keys if k != 'oracle_task_within'], key=lambda k: statistics.mean(sm(k, 'val')))
    (a.out / 'summary.json').write_text(json.dumps(dict(selected=best, test=sm(best, 'test'), task_acc=sm(best, 'test', 'task_acc'), all={k: dict(val=sm(k, 'val'), test=sm(k, 'test')) for k in keys}), indent=2))
    print('validation-selected:', best)


if __name__ == '__main__':
    main()
