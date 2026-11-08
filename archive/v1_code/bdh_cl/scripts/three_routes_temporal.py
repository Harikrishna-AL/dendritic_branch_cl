"""Route 3: temporal context, with switch handling, on frozen hybrid5 checkpoints.

Streams (test images each appear once): blocked (fixed 200-image blocks, as before),
geometric (task blocks of random length, mean 50), mixed (iid shuffle).
Methods, all causal (predict, then update state) and label-free:
  none            stateless soft-magnitude model
  prior           EMA class-prior on logits (non-dendritic control)
  pseudo          pseudo-error branch state (original rule)
  pseudo_gated    same, strength scaled by a persistence signal: EMA of whether
                  consecutive stateless predictions fall in the same digit pair
  task_hmm        task-level forward filter over the 5 tasks (switch prob h),
                  emission = stateless pair probability; predict within inferred
                  task using that task's archived branch route
  task_hmm_mix    Bayesian mixture of task_hmm filters over several h (switch-rate
                  is inferred online, no h is selected)
Every method's hyper-parameters are selected on VALIDATION by the mean accuracy over
all three stream types, so the mixed-stream failure is penalised, not hidden.
"""
import json, statistics, itertools, argparse
from pathlib import Path
import numpy as np, torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_short_term_context_test import features, stream_order
from bdh_cl.scripts.three_routes_self_scoring import route_logits

STREAMS = ['blocked', 'geometric', 'mixed']


def geometric_order(y, seed, mean=50):
    g = np.random.default_rng(seed + 7); q = [list(g.permutation(np.flatnonzero(y // 2 == t))) for t in range(5)]; out = []
    while any(q):
        t = g.choice([i for i in range(5) if q[i]]); n = int(g.geometric(1 / mean)); out += q[t][:n]; del q[t][:n]
    return np.array(out)


def order_of(y, seed, kind): return geometric_order(y, seed) if kind == 'geometric' else stream_order(y, seed, kind)


def sm(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def run_prior(c, order, decay, strength):
    st = None; pred = np.empty(len(order), int)
    for i, ix in enumerate(order):
        lg = c['logits'][ix]; adj = lg if st is None else lg + strength * np.log(st + 1e-4); pred[i] = adj.argmax()
        p = sm(adj); st = p if st is None else decay * st + (1 - decay) * p
    return pred


def run_pseudo(c, order, decay, strength, c0=None, gdecay=.9):
    st = None; pred = np.empty(len(order), int); per = None; prev = None
    for i, ix in enumerate(order):
        cur = c['current'][ix]; s = strength
        if c0 is not None:
            pair = int(c['logits'][ix].argmax()) // 2
            if prev is not None: per = (1. if pair == prev else 0.) if per is None else gdecay * per + (1 - gdecay) * (pair == prev)
            prev = pair; s = strength * (0. if per is None else float(np.clip((per - c0) / (1 - c0), 0, 1)))
        gate = cur if st is None else (1 - s) * cur + s * st
        pred[i] = int((c['W'] @ (c['z'][ix] * gate).sum(-1) + c['b']).argmax())
        o = c['pseudo'][ix]; st = o.copy() if st is None else decay * st + (1 - decay) * o
    return pred


def run_hmm(c, order, hs, T):
    """hs: list of switch probabilities; >1 entries -> online Bayesian mixture over h."""
    K = len(hs); b = np.full((K, 5), .2); logw = np.zeros(K); pred = np.empty(len(order), int)
    H = np.array(hs)[:, None]
    for i, ix in enumerate(order):
        p = sm(c['logits'][ix]); e = (p.reshape(5, 2).sum(-1) + 1e-6) ** (1 / T)
        prior = (1 - H) * b + H / 5; joint = prior * e; ev = joint.sum(-1)
        logw = .99 * (logw + np.log(ev)); w = np.exp(logw - logw.max()); w /= w.sum()  # forgetting lets the switch-rate estimate track regime changes
        b = joint / ev[:, None]; post = (w[:, None] * b).sum(0); t = int(post.argmax())
        pred[i] = 2 * t + int(c['route'][t, ix, 2 * t:2 * t + 2].argmax())
    return pred


def configs():
    yield 'none', {}
    for d, s in itertools.product([.8, .9, .95], [.3, .5, .7]): yield 'prior', dict(decay=d, strength=s)
    for d, s in itertools.product([.9, .99], [.5, .9]): yield 'pseudo', dict(decay=d, strength=s)
    for d, s, c0 in itertools.product([.9, .99], [.5, .9], [.3, .5, .7]): yield 'pseudo_gated', dict(decay=d, strength=s, c0=c0)
    for h, T in itertools.product([.5, .1, .02, .005], [.5, 1., 2.]): yield 'task_hmm', dict(hs=[h], T=T)
    for T in [.5, 1., 2.]: yield 'task_hmm_mix', dict(hs=[.8, .3, .1, .03, .01, .003], T=T)


def run(method, cfg, c, order):
    if method == 'none': return c['logits'][order].argmax(-1)
    if method == 'prior': return run_prior(c, order, **cfg)
    if method in ['pseudo', 'pseudo_gated']: return run_pseudo(c, order, **cfg)
    return run_hmm(c, order, **cfg)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/three_routes_20261106/temporal')); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    raw = load_raw('data_cache/mnist'); split = np.load(SOURCE / 'split_indices.npz')
    def ds(im, lab): return (torch.tensor(im.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780, torch.tensor(lab.copy(), dtype=torch.long)
    vx, vy = ds(raw['train_images'][split['validation']], raw['train_labels'][split['validation']]); tx, ty = ds(raw['test_images'], raw['test_labels'])
    caches = {}
    for seed in [101, 102, 103]:
        for first in [0, 4]:
            m = Model('baseline', seed); saved = torch.load(Path('results/mnist_gradient_branch_scoring_20261106') / f'hybrid5_{seed}_{first}.pt', weights_only=True)
            m.load_state_dict(saved['state']); m.requires_grad_(False); W = m.head.weight.numpy(); b = m.head.bias.numpy()
            for name, x, yy in [('val', vx, vy), ('test', tx, ty)]:
                f = features(m, x, yy); f['logits'] = (f['z'] * f['current']).sum(-1) @ W.T + b
                f['route'] = route_logits(m, x, saved['routes']).numpy(); f['W'] = W; f['b'] = b; f['y'] = yy.numpy()
                caches[(seed, first, name)] = f
            print('cached', seed, first, flush=True)
    def evaluate(split_name, method, cfg):
        res = {k: [] for k in STREAMS}
        for seed in [101, 102, 103]:
            per = {k: [] for k in STREAMS}
            for first in [0, 4]:
                c = caches[(seed, first, split_name)]
                for k in STREAMS:
                    o = order_of(c['y'], seed + first, k); per[k].append(float((run(method, cfg, c, o) == c['y'][o]).mean()))
            for k in STREAMS: res[k].append(statistics.mean(per[k]) * 100)
        return res
    val = []
    for method, cfg in configs():
        r = evaluate('val', method, cfg); val.append(dict(method=method, cfg=cfg, val=r, objective=statistics.mean(statistics.mean(v) for v in r.values())))
        print('val', method, cfg, {k: round(statistics.mean(v), 2) for k, v in r.items()}, flush=True)
    (a.out / 'validation.json').write_text(json.dumps(val, indent=2))
    final = {}
    for method in dict.fromkeys(v['method'] for v in val):
        best = max((v for v in val if v['method'] == method), key=lambda v: v['objective'])
        final[method] = dict(cfg=best['cfg'], test=evaluate('test', method, best['cfg']))
    (a.out / 'test_results.json').write_text(json.dumps(final, indent=2))
    print('\nTEST (selected on validation, mean over all stream types); mean ± sd over 3 seeds')
    print(f'{"method":14s}' + ''.join(f'{k:>18s}' for k in STREAMS) + '   cfg')
    for method, r in final.items():
        print(f'{method:14s}' + ''.join(f'{statistics.mean(r["test"][k]):11.2f} ± {statistics.stdev(r["test"][k]):4.2f}' for k in STREAMS) + f'   {r["cfg"]}')


if __name__ == '__main__':
    main()
