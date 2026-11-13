"""Why does image-only accuracy fall with depth? Diagnostics and inference-only fixes.

Loads checkpoints written by
  python scripts/train.py --bench mnist --model branch --arch <d> --save-checkpoints results/depth/ckpt
and, for every run, measures (test split unless noted):

  layer swap     true route at one layer, soft mixing elsewhere (and the reverse)
  agreement      per layer: how often the strongest branch (argmax |response|) is the true
                 route's branch, with soft vs true-route layers upstream
  corruption     true routes everywhere except one layer, which gets another task's route
                 or a random route (does that layer's route carry information?)
  signal         per layer: mean |dCE/dh| and the gap between the two lowest error traces
  fixes          sharper soft mixing softmax(|r|/T) (T picked on validation), and choosing
                 one stored route for the whole network by branch agreement or by confidence

  python scripts/depth_diagnostic.py
"""
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
from torch.nn import functional as F

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.data import N_TASKS, task_of  # noqa: E402
from branch_routing.models import BranchMLP  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

CKPT = Path('results/depth/ckpt'); OUT = Path('results/depth')
TEMPS = [1., .5, .25, .1, .05, 0.]  # 0 = hard argmax


@torch.no_grad()
def forward(m, x, modes, routes=None, T=1.):
    """modes: per layer 'soft' | 'route'. routes: per layer (N, n) branch indices.
    Returns logits and per-layer raw responses."""
    h, raws = x, []
    for i, layer in enumerate(m.layers):
        raw = layer.responses(h); z = raw.tanh(); raws.append(raw)
        if modes[i] == 'route': w = F.one_hot(routes[i], layer.B).float()
        elif T == 0: w = F.one_hot(raw.abs().argmax(-1), layer.B).float()
        else: w = (raw.abs() / T).softmax(-1)
        h = (z * w).sum(-1)
    return m.head(h), raws


def per_sample(routes, tasks): return [r[tasks] for r in routes]


def cil(logits, y): return float((logits.argmax(1) == y).float().mean())


def til(logits, y):
    t = task_of(y); i = torch.arange(len(y)); return float((2 * t + logits[i[:, None], torch.stack([2 * t, 2 * t + 1], 1)].argmax(1) == y).float().mean())


def pair_pred(logits, t):
    i = torch.arange(len(t)); return 2 * t + torch.stack([logits[i, 2 * t], logits[i, 2 * t + 1]], 1).argmax(1)


@torch.no_grad()
def whole_route(m, x, routes):
    """Run every task's full stored route. Returns per-task (agreement score, layer-1 score,
    confidence, within-pair prediction), each (N, n_tasks)."""
    agree, agree1, conf, pred = [], [], [], []
    for t in range(N_TASKS):
        r = [q[t].expand(len(x), -1) for q in routes]; logits, raws = forward(m, x, ['route'] * len(routes), r)
        lp = [raw.abs().log_softmax(-1).gather(2, rr[..., None]).squeeze(2).mean(1) for raw, rr in zip(raws, r)]
        agree.append(torch.stack(lp).mean(0)); agree1.append(lp[0])
        pair = logits[:, 2 * t:2 * t + 2]; conf.append(pair.softmax(1).max(1).values); pred.append(2 * t + pair.argmax(1))
    return [torch.stack(v, 1) for v in (agree, agree1, conf, pred)]


def chunked(fn, x, *rest, n=1000):
    outs = [fn(x[j:j + n], *[r[j:j + n] for r in rest]) for j in range(0, len(x), n)]
    return [torch.cat([o[k] for o in outs]) for k in range(len(outs[0]))] if isinstance(outs[0], (list, tuple)) else torch.cat(outs)


def diagnose(ck, data):
    info = ck['info']; depth = int(info['arch']); m = BranchMLP(depth, info['seed']); m.load_state_dict(ck['state']); m.eval()
    routes = ck['routes']; L = len(routes); rec = dict(seed=info['seed'], order=ck['order'], depth=depth)
    x, y = data.test_x, data.test_y; t = task_of(y); true = per_sample(routes, t)

    def run(modes, rts=true, T=1., xx=x): return chunked(lambda a, *r: forward(m, a, modes, list(r), T)[0], xx, *rts)
    rec['soft'] = cil(run(['soft'] * L), y); rec['route_cil'] = cil(run(['route'] * L), y); rec['route_til'] = til(run(['route'] * L), y)

    # layer swap
    rec['fix_one'] = [cil(run(['route' if j == i else 'soft' for j in range(L)]), y) for i in range(L)]
    rec['soft_one'] = [cil(run(['soft' if j == i else 'route' for j in range(L)]), y) for i in range(L)]
    # agreement of the strongest branch with the true route, all-soft vs all-true upstream
    def agreement(modes):
        raws = chunked(lambda a, *r: forward(m, a, modes, list(r))[1], x, *true)
        return [float((raw.abs().argmax(-1) == tr).float().mean()) for raw, tr in zip(raws, true)]
    rec['agree_soft_upstream'] = agreement(['soft'] * L); rec['agree_true_upstream'] = agreement(['route'] * L)
    # corruption of one layer's route
    other = per_sample(routes, (t + 1) % N_TASKS); g = torch.Generator().manual_seed(info['seed'])
    rand = [torch.randint(0, m.layers[i].B, (routes[i].shape[1],), generator=g).expand(len(x), -1) for i in range(L)]
    for name, bad in [('other_task', other), ('random', rand)]:
        res = []
        for i in range(L):
            r = [bad[j] if j == i else true[j] for j in range(L)]; lg = run(['route'] * L, r); res.append(dict(cil=cil(lg, y), til=til(lg, y)))
        rec[f'corrupt_{name}'] = res
    # training signal at the end of training (last task's training images, training-mode routes)
    last = ck['order'][-1]; xi = data.x[task_of(data.y) == last][:2000]; yi = data.y[task_of(data.y) == last][:2000]
    with torch.enable_grad():
        logits, hs, _, _ = m(xi, 'trace', internals=True); grads = torch.autograd.grad(F.cross_entropy(logits, yi), hs)
    rec['grad_abs'] = [float((g_ * len(xi)).abs().mean()) for g_ in grads]
    rec['trace_gap'] = [float(e.sort(-1).values[:, 1].sub(e.sort(-1).values[:, 0]).median()) for e in (l.error for l in m.layers)]
    # fix 1: sharper soft mixing (val and test)
    rec['temp'] = {str(T): dict(val=cil(run(['soft'] * L, [r[:len(data.val_x)] for r in true], T, data.val_x), data.val_y), test=cil(run(['soft'] * L, T=T), y)) for T in TEMPS}
    # fix 2: one stored route for the whole network
    for split, (xs, ys) in [('val', (data.val_x, data.val_y)), ('test', (x, y))]:
        agree, agree1, conf, pred = chunked(lambda a: whole_route(m, a, routes), xs)
        for name, score in [('agree_all', agree), ('agree_layer1', agree1), ('confidence', conf)]:
            tk = score.argmax(1); p = pred.gather(1, tk[:, None]).squeeze(1)
            rec.setdefault(f'whole_{name}', {})[split] = dict(cil=float((p == ys).float().mean()), taskid=float((tk == task_of(ys)).float().mean()))
    return rec


def ms(runs, fn, d=1):
    m_, sd = mean_sd(seed_means(runs, fn).values()); return f'{100 * m_:.{d}f} ± {100 * sd:.{d}f}'


def summary(recs):
    L_ = ['# Depth diagnostic (Split MNIST, test)', '', 'Mean ± SD over 3 seed means (forward + reverse orders averaged). All-10 accuracy unless noted.', '']
    depths = sorted({r['depth'] for r in recs}); by = {d: [r for r in recs if r['depth'] == d] for d in depths}
    L_ += ['## Baseline', '', '| layers | image only (soft) | true route | within-pair, true route |', '|---:|---:|---:|---:|']
    for d, R in by.items(): L_.append(f"| {d} | {ms(R, lambda r: r['soft'])} | {ms(R, lambda r: r['route_cil'])} | {ms(R, lambda r: r['route_til'])} |")
    L_ += ['', '## Layer swap', '', 'fix layer i: true route at layer i only, soft elsewhere. soft layer i: soft at layer i only, true route elsewhere.', '',
           '| layers | layer | fix layer i | soft layer i |', '|---:|---:|---:|---:|']
    for d, R in by.items():
        for i in range(d): L_.append(f"| {d} | {i + 1} | {ms(R, lambda r: r['fix_one'][i])} | {ms(R, lambda r: r['soft_one'][i])} |")
    L_ += ['', '## Strongest branch = true route branch (chance 25%)', '', '| layers | layer | upstream soft | upstream true route |', '|---:|---:|---:|---:|']
    for d, R in by.items():
        for i in range(d): L_.append(f"| {d} | {i + 1} | {ms(R, lambda r: r['agree_soft_upstream'][i])} | {ms(R, lambda r: r['agree_true_upstream'][i])} |")
    L_ += ['', '## Corrupting one layer\'s route (true routes elsewhere)', '', '| layers | layer | other task: all-10 | other task: within-pair | random: all-10 | random: within-pair |', '|---:|---:|---:|---:|---:|---:|']
    for d, R in by.items():
        for i in range(d):
            L_.append(f"| {d} | {i + 1} | {ms(R, lambda r: r['corrupt_other_task'][i]['cil'])} | {ms(R, lambda r: r['corrupt_other_task'][i]['til'])} | {ms(R, lambda r: r['corrupt_random'][i]['cil'])} | {ms(R, lambda r: r['corrupt_random'][i]['til'])} |")
    L_ += ['', '## Training signal at the end of training', '', '| layers | layer | mean abs dCE/dh (x batch) | median gap between two lowest traces |', '|---:|---:|---:|---:|']
    for d, R in by.items():
        for i in range(d):
            L_.append(f"| {d} | {i + 1} | {statistics.mean(r['grad_abs'][i] for r in R):.4f} | {statistics.mean(r['trace_gap'][i] for r in R):.4f} |")
    L_ += ['', '## Fix 1: sharper soft mixing softmax(|r|/T); T = 0 is hard argmax', '', '| layers | ' + ' | '.join(f'T={T:g}' for T in TEMPS) + ' | T picked on val → test |', '|---:|' + '---:|' * (len(TEMPS) + 1)]
    for d, R in by.items():
        best = max(TEMPS, key=lambda T: statistics.mean(r['temp'][str(T)]['val'] for r in R))
        L_.append(f'| {d} | ' + ' | '.join(ms(R, lambda r: r['temp'][str(T)]['test']) for T in TEMPS) + f" | T={best:g}: {ms(R, lambda r: r['temp'][str(best)]['test'])} |")
    L_ += ['', '## Fix 2: one stored route for the whole network', '',
           'agree_all: route whose branches the network itself prefers (mean log soft weight of the route branch, all layers). agree_layer1: same, first layer only. confidence: route with the most confident within-pair prediction.', '',
           '| layers | method | all-10 | task picked correctly |', '|---:|---|---:|---:|']
    for d, R in by.items():
        for name in ['agree_all', 'agree_layer1', 'confidence']:
            L_.append(f"| {d} | {name} | {ms(R, lambda r: r[f'whole_{name}']['test']['cil'])} | {ms(R, lambda r: r[f'whole_{name}']['test']['taskid'])} |")
    return '\n'.join(L_) + '\n'


def main():
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist')
    recs = []
    for f in sorted(CKPT.glob('branch_*.pt')):
        recs.append(diagnose(torch.load(f, weights_only=False), data)); print(f.name, f"soft {100 * recs[-1]['soft']:.1f}", flush=True)
    (OUT / 'diagnostic.json').write_text(json.dumps(recs, indent=1)); s = summary(recs); (OUT / 'summary.md').write_text(s); print(s)


if __name__ == '__main__':
    main()
