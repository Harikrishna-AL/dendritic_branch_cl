"""Why does image-only branch selection fail in branched CNNs, and does branch memory fix it?

Model: conv_b4_sparse from branched_conv_cifar.py (retrained, identical recipe/seeds), saved.

Part A (static, per image, test set):
  A1 selection agreement: per layer, fraction of channels whose largest-|response| branch
     equals the archived route of the image's true task; task-ID read from branch choices
     (task whose route matches most channels). Same metrics on MNIST hybrid5 checkpoints.
  A2 partial oracle: true-task route at a subset of layers, soft-magnitude elsewhere.
  A3 alternative single-image selectors: hard magnitude; per-image pseudo-error (branch that
     most lowers CE against the model's own prediction, first-order).
  A4 is the task decodable at all? 5-way linear probe for task ID on (pixels / soft-mode
     top-layer features), fit on validation, scored on test. Diagnostic only (not CL-legal).
Part B (streams, causal, label-free): blocked(200) / geometric(mean 50) / mixed.
  none, prior (EMA class prior), pseudo (per-layer pseudo-error branch state),
  pseudo_gated (state strength scaled by persistence of predicted pair), supervised (state
  from true-label error: upper bound on feedback quality). Hyper-parameters selected on
  validation by mean accuracy over all three streams. All streams x configs run as one batch.
"""
import json, statistics, itertools, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from bdh_cl.data import cifar10
from bdh_cl.scripts import branched_conv_cifar as bc
from bdh_cl.scripts.three_routes_temporal import order_of

STREAMS = ['blocked', 'geometric', 'mixed']
OUT = Path('results/branched_conv_access_20261106')


def forward(m, x, ws=None, keep=False):
    """ws: per-layer list of weights (N,c,B) or None (soft magnitude). Returns logits, hs, zs, raws."""
    h = bc.norm(x); hs, zs, raws = [], [], []; i = 0
    for L in m.convs:
        r = L.raw(h); z = r.tanh(); w = ws[i] if ws is not None and ws[i] is not None else r.abs().mean((3, 4)).softmax(-1)
        h = (z * w[..., None, None]).sum(2); hs.append(h); zs.append(z); raws.append(r); i += 1
    f = F.layer_norm(h.flatten(1), (m.d,)); r = m.top.raw(f); z = r.tanh(); w = ws[i] if ws is not None and ws[i] is not None else r.abs().softmax(-1)
    h = (z * w).sum(-1); hs.append(h); zs.append(z); raws.append(r); lg = m.head(h)
    return (lg, hs, zs, raws, f) if keep else lg


def branch_pref(m, x, target, ws=None):
    """First-order substitution error per layer/channel/branch vs target -> within-channel softmax preference."""
    with torch.enable_grad():
        m.requires_grad_(True)
        lg, hs, zs, _, _ = forward(m, x, ws, keep=True); tgt = lg.argmax(1) if target is None else target
        ce = F.cross_entropy(lg, tgt, reduction='sum'); gs = torch.autograd.grad(ce, hs); m.requires_grad_(False)
    out = []
    for h, z, g in zip(hs, zs, gs):
        e = g.unsqueeze(2) * (z - h.unsqueeze(2)).detach()
        e = e.sum((3, 4)) if e.dim() == 5 else e
        c = e - e.mean(-1, keepdim=True); out.append((-c / c.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)).softmax(-1))
    return lg.detach(), out


def train(seed, order, data):
    """Copy of bc.run training loop that returns model and routes."""
    x, y = data[0], data[1]; m = bc.Net('conv_b4_sparse', seed); opt = torch.optim.Adam(m.parameters(), lr=.0005)
    routes = [torch.zeros(5, L.error.shape[0], dtype=torch.long) for L in m.layers()]
    for stage, task in enumerate(order):
        ix = torch.where(y // 2 == task)[0]
        for epoch in range(2):
            perm = ix[torch.randperm(len(ix), generator=torch.Generator().manual_seed(seed * 10000 + stage * 100 + epoch))]
            for ids in perm.split(64):
                opt.zero_grad(set_to_none=True); loss, fb = bc.train_step(m, x[ids], y[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                with torch.no_grad():
                    for L, f in zip(m.layers(), fb): L.error.mul_(.95).add_(f, alpha=.05)
        for i, L in enumerate(m.layers()): routes[i][task] = L.error.argmin(-1)
    return m, routes


def onehot_routes(routes, t, B=4): return [F.one_hot(r[t], B).float() for r in routes]


@torch.no_grad()
def static(m, routes, x, y, vx, vy):
    m.eval(); t = y // 2; nl = len(routes); res = {}
    sel, task_votes, lg_soft, hard_pred, pse_pred, feats = [[] for _ in range(nl)], [[] for _ in range(nl)], [], [], [], []
    for ids in torch.arange(len(y)).split(500):
        lg, hs, zs, raws, f = forward(m, x[ids], keep=True); lg_soft.append(lg.argmax(1)); feats.append(f)
        for l, r in enumerate(raws):
            a = (r.abs().mean((3, 4)) if r.dim() == 5 else r.abs()).argmax(-1)  # (N,c)
            sel[l].append((a == routes[l][t[ids]]).float().mean(1))
            task_votes[l].append(torch.stack([(a == routes[l][k]).float().mean(1) for k in range(5)], 1))
        # hard magnitude at every layer (sequential: hard choice changes downstream input)
        h = bc.norm(x[ids]); ws = []
        for L in m.convs:
            r = L.raw(h); w = F.one_hot(r.abs().mean((3, 4)).argmax(-1), 4).float(); ws.append(w); h = (r.tanh() * w[..., None, None]).sum(2)
        f2 = F.layer_norm(h.flatten(1), (m.d,)); r = m.top.raw(f2); ws.append(F.one_hot(r.abs().argmax(-1), 4).float())
        hard_pred.append(forward(m, x[ids], ws).argmax(1))
        # per-image pseudo-error selection: hard branch minimising estimated CE vs own soft prediction
        _, prefs = branch_pref(m, x[ids], None); pse_pred.append(forward(m, x[ids], [F.one_hot(p.argmax(-1), 4).float() for p in prefs]).argmax(1))
    acc = lambda p: float((torch.cat(p) == y).float().mean())
    res['soft'] = acc(lg_soft); res['hard_magnitude'] = acc(hard_pred); res['pseudo_per_image'] = acc(pse_pred)
    res['selection_agreement'] = [float(torch.cat(s).mean()) for s in sel]
    votes = [torch.cat(v) for v in task_votes]
    res['taskid_from_choices'] = [float((v.argmax(1) == t).float().mean()) for v in votes]
    res['taskid_from_choices_all_layers'] = float((sum(votes).argmax(1) == t).float().mean())
    # chance agreement: how often a different task's route matches the true route (routes overlap)
    res['route_agreement_other_task'] = [float(statistics.mean(float((r[i] == r[j]).float().mean()) for i, j in itertools.permutations(range(5), 2))) for r in routes]
    # partial oracle: true route at subset S of layers
    names = ['conv1', 'conv2', 'conv3', 'top']; po = {}
    subsets = {'all': [0, 1, 2, 3], 'convs_only': [0, 1, 2], 'top_only': [3], 'conv1_only': [0], 'conv2_only': [1], 'conv3_only': [2], 'all_but_conv1': [1, 2, 3], 'all_but_top': [0, 1, 2]}
    for k, S in subsets.items():
        p = []
        for ids in torch.arange(len(y)).split(500):
            ws = [F.one_hot(routes[l][t[ids]], 4).float() if l in S else None for l in range(nl)]; p.append(forward(m, x[ids], ws).argmax(1))
        po[k] = acc(p)
    res['partial_oracle'] = po
    # task-ID linear probes (diagnostic): fit on validation, score on test
    vf = torch.cat([forward(m, vx[j:j + 500], keep=True)[4] for j in range(0, len(vy), 500)])
    res['probe_taskid_features'] = probe(vf, vy // 2, torch.cat(feats), t)
    res['probe_taskid_pixels'] = probe(bc.norm(vx).flatten(1), vy // 2, bc.norm(x).flatten(1), t)
    m.train(); return res


def probe(a, ya, b, yb, epochs=300):
    mu, sd = a.mean(0), a.std(0) + 1e-5; a, b = (a - mu) / sd, (b - mu) / sd
    lin = torch.nn.Linear(a.shape[1], 5); opt = torch.optim.Adam(lin.parameters(), lr=.01, weight_decay=1e-3)
    with torch.enable_grad():
        for _ in range(epochs): opt.zero_grad(); F.cross_entropy(lin(a), ya).backward(); opt.step()
    with torch.no_grad(): return float((lin(b).argmax(1) == yb).float().mean())


def mnist_static():
    from bdh_cl.data.mnist import load_raw
    from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
    from bdh_cl.scripts.mnist_combined_selector import Model
    raw = load_raw('data_cache/mnist'); tx = (torch.tensor(raw['test_images'].copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780; ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long); t = ty // 2
    out = []
    for seed in [101, 102, 103]:
        for first in [0, 4]:
            m = Model('baseline', seed); s = torch.load(Path('results/mnist_gradient_branch_scoring_20261106') / f'hybrid5_{seed}_{first}.pt', weights_only=True); m.load_state_dict(s['state']); R = s['routes']
            with torch.no_grad(): a = m.responses(tx, False).abs().argmax(-1)
            votes = torch.stack([(a == R[k]).float().mean(1) for k in range(5)], 1)
            out.append(dict(selection_agreement=float((a == R[t]).float().mean()), taskid_from_choices=float((votes.argmax(1) == t).float().mean()),
                            route_agreement_other_task=statistics.mean(float((R[i] == R[j]).float().mean()) for i, j in itertools.permutations(range(5), 2))))
    return {k: statistics.mean(o[k] for o in out) for k in out[0]}


# ---------------- Part B: streams ----------------
def configs():
    yield 'none', {}
    for d, s in itertools.product([.8, .9, .95], [.3, .5, .7]): yield 'prior', dict(decay=d, strength=s)
    for d, s in itertools.product([.9, .99], [.5, .9]): yield 'pseudo', dict(decay=d, strength=s)
    for d, s, c0 in itertools.product([.9, .99], [.5, .9], [.3, .5, .7]): yield 'pseudo_gated', dict(decay=d, strength=s, c0=c0)
    for d, s in itertools.product([.9, .99], [.5, .9]): yield 'supervised', dict(decay=d, strength=s)


@torch.no_grad()
def stateless_cache(m, x, y):
    """Per image: stateless soft logits and per-layer pseudo/supervised branch preferences."""
    lg, pp, ps = [], [], []
    for ids in torch.arange(len(y)).split(500):
        l, a = branch_pref(m, x[ids], None); _, b = branch_pref(m, x[ids], y[ids]); lg.append(l); pp.append(a); ps.append(b)
    return dict(lg=torch.cat(lg), pp=[torch.cat([q[l] for q in pp]) for l in range(len(pp[0]))], ps=[torch.cat([q[l] for q in ps]) for l in range(len(ps[0]))])


@torch.no_grad()
def run_streams(m, x, y, orders, cfgs, cache):
    """orders: list of K index arrays (same length); cfgs: list of K (method, cfg). Returns K accuracies.
    Causal: each image is predicted with the state built from previous images, then the state is updated."""
    m.eval(); K = len(orders); O = torch.tensor(np.stack(orders)); n = O.shape[1]; correct = torch.zeros(K)
    nl = len(m.layers()); state = [None] * nl; prior = None; per = torch.full((K,), float('nan')); prev = None
    meth = [c[0] for c in cfgs]
    dec = torch.tensor([c[1].get('decay', 0.) for c in cfgs]); stren = torch.tensor([c[1].get('strength', 0.) for c in cfgs])
    c0 = torch.tensor([c[1].get('c0', -1.) for c in cfgs]); is_prior = torch.tensor([k == 'prior' for k in meth]); is_br = torch.tensor([k in ['pseudo', 'pseudo_gated', 'supervised'] for k in meth])
    is_sup = torch.tensor([k == 'supervised' for k in meth]); gated = c0 >= 0; br = torch.where(is_br)[0]
    for i in range(n):
        ids = O[:, i]; yb = y[ids]; lg0 = cache['lg'][ids]
        obs = [torch.where(is_sup[:, None, None], s[ids], p[ids]) for p, s in zip(cache['pp'], cache['ps'])]
        pair = lg0.argmax(1) // 2
        if prev is not None:
            same = (pair == prev).float(); per = torch.where(per.isnan(), same, .9 * per + .1 * same)
        prev = pair
        s = stren * is_br.float()
        s = torch.where(gated, s * ((per.nan_to_num(0.) - c0) / (1 - c0)).clamp(0, 1) * (~per.isnan()).float(), s)
        if state[0] is None: pred = lg0.argmax(1)
        else:
            lg = lg0 + is_prior[:, None].float() * stren[:, None] * (prior + 1e-4).log()
            if len(br):
                sb = s[br][:, None, None]; h = bc.norm(x[ids[br]])
                for l, L in enumerate(m.convs):
                    r = L.raw(h); w = (1 - sb) * r.abs().mean((3, 4)).softmax(-1) + sb * state[l][br]; h = (r.tanh() * w[..., None, None]).sum(2)
                r = m.top.raw(F.layer_norm(h.flatten(1), (m.d,))); w = (1 - sb) * r.abs().softmax(-1) + sb * state[-1][br]
                lg[br] = m.head((r.tanh() * w).sum(-1))
            pred = lg.argmax(1)
        correct += (pred == yb).float()
        sp = lg0.softmax(1) if prior is None else (lg0 + stren[:, None] * (prior + 1e-4).log()).softmax(1)
        prior = sp if prior is None else dec[:, None] * prior + (1 - dec[:, None]) * sp
        state = [o.clone() if st is None else dec[:, None, None] * st + (1 - dec[:, None, None]) * o for st, o in zip(state, obs)]
    m.train(); return (correct / n).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--threads', type=int, default=8); p.add_argument('--stage', default='all'); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    d = cifar10.load('data_cache/cifar10')
    data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
    data = [v.long() if i % 2 else v for i, v in enumerate(data)]; x, y, vx, vy, tx, ty = data
    models = {}
    for seed in [201, 202, 203]:
        for order in [list(range(5)), list(range(4, -1, -1))]:
            path = OUT / f'model_{seed}_{order[0]}.pt'
            if path.exists(): s = torch.load(path, weights_only=True); m = bc.Net('conv_b4_sparse', seed); m.load_state_dict(s['state']); routes = s['routes']
            else:
                t0 = time.perf_counter(); m, routes = train(seed, order, data); torch.save(dict(state=m.state_dict(), routes=routes), path); print('trained', seed, order[0], f'{time.perf_counter() - t0:.0f}s', flush=True)
            m.requires_grad_(False); models[(seed, order[0])] = (m, routes)
    if a.stage in ['all', 'static']:
        st = {}
        for k, (m, routes) in models.items():
            st[f'{k[0]}_{k[1]}'] = static(m, routes, tx, ty, vx, vy); print('static', k, json.dumps(st[f'{k[0]}_{k[1]}']), flush=True)
        st['mnist_hybrid5'] = mnist_static(); print('mnist', st['mnist_hybrid5'], flush=True)
        (OUT / 'static.json').write_text(json.dumps(st, indent=2))
    if a.stage in ['all', 'streams']:
        cfgs = list(configs()); results = {}
        for split, (sx, sy) in [('val', (vx, vy)), ('test', (tx, ty))]:
            for (seed, first), (m, routes) in models.items():
                orders, keys = [], []
                for (meth, cfg), kind in itertools.product(cfgs, STREAMS):
                    orders.append(order_of(sy.numpy(), seed + first, kind)); keys.append((meth, cfg, kind))
                t0 = time.perf_counter(); accs = run_streams(m, sx, sy, orders, [(k[0], k[1]) for k in keys], stateless_cache(m, sx, sy))
                results[(split, seed, first)] = list(zip(keys, accs)); print('streams', split, seed, first, f'{time.perf_counter() - t0:.0f}s', flush=True)
        # select per method on validation by mean over streams and runs
        def table(split, meth, cfg):
            per = {k: [] for k in STREAMS}
            for seed in [201, 202, 203]:
                for k in STREAMS: per[k].append(100 * statistics.mean(acc for first in [0, 4] for (mm, cc, kk), acc in results[(split, seed, first)] if mm == meth and cc == cfg and kk == k))
            return per
        final, val = {}, []
        for meth, cfg in cfgs:
            v = table('val', meth, cfg); val.append(dict(method=meth, cfg=cfg, val=v, objective=statistics.mean(statistics.mean(q) for q in v.values())))
        for meth in dict.fromkeys(c[0] for c in cfgs):
            best = max((v for v in val if v['method'] == meth), key=lambda v: v['objective']); final[meth] = dict(cfg=best['cfg'], test=table('test', meth, best['cfg']))
        (OUT / 'streams_validation.json').write_text(json.dumps(val, indent=2)); (OUT / 'streams_test.json').write_text(json.dumps(final, indent=2))
        for meth, r in final.items(): print(f'{meth:13s}' + ''.join(f'{statistics.mean(r["test"][k]):8.2f} ± {statistics.stdev(r["test"][k]):4.2f}' for k in STREAMS), r['cfg'], flush=True)


if __name__ == '__main__':
    main()
