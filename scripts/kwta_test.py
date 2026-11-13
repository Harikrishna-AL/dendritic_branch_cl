"""APL-style global inhibition: k-winners-take-all over all branches of a layer.

Inspired by the fly mushroom body: one inhibitory neuron (APL) lets only the ~5-10% most
strongly driven Kenyon cells fire, with no learned gate and no negative examples.
Here every branch layer keeps only its k_frac most strongly driven branches (largest
|response|, all neurons together) per input; a neuron outputs the sum of its active branches.

  inference : existing checkpoints (trained without competition), competition at test only
  trained   : checkpoints trained with the competition (train.py --kwta k); a branch then
              drives its neuron, and learns, only when it is among the active ones

Route hit = share of active branches that are the true task route's branch for that neuron
(chance 1/B = 25%). Enrichment = how much more often route branches are active than branches
in general (1 = no task information in which branches win).

  python scripts/kwta_test.py
"""
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.continual import predict_logits  # noqa: E402
from branch_routing.data import task_of  # noqa: E402
from branch_routing.models import BranchMLP  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

K_INFER = [.01, .02, .05, .1, .25, .5]


@torch.no_grad()
def route_stats(m, x, routes, chunk=1000):
    """Per layer: (route hit, enrichment, fraction of neurons with >= 1 active branch), image-only forward."""
    hits = [[0., 0., 0., 0.] for _ in m.layers]  # active route branches, active branches, route branches, neurons on
    for j in range(0, len(x), chunk):
        h = x[j:j + chunk]; r = [q[j:j + chunk] for q in routes]
        for i, layer in enumerate(m.layers):
            raw = layer.responses(h); mask = layer.active_mask(raw)
            on_route = mask.gather(2, r[i][..., None]).squeeze(2)
            hits[i][0] += float(on_route.sum()); hits[i][1] += float(mask.sum()); hits[i][2] += float(on_route.numel())
            hits[i][3] += float((mask.sum(-1) > 0).float().sum())
            h, _ = layer.mix(raw, 'soft')
    out = []
    for i, (a, act, n_route, on) in enumerate(hits):
        k = act / (n_route * m.layers[i].B)  # fraction of all branches active
        out.append(dict(hit=a / act, enrichment=(a / n_route) / k, neurons_on=on / n_route))
    return out


def load(f):
    ck = torch.load(f, weights_only=False); d = int(ck['info']['arch']); m = BranchMLP(d, ck['info']['seed']); m.load_state_dict(ck['state']); m.eval()
    return ck, m


def main():
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist'); x, y = data.test_x, data.test_y; t = task_of(y)
    acc = lambda m: float((predict_logits(m, x, 'soft').argmax(1) == y).float().mean())
    recs = []
    for f in sorted(Path('results/depth/ckpt').glob('branch_[123]_1*.pt')):  # trained without competition
        ck, m = load(f); routes = [q[t] for q in ck['routes']]; rec = dict(kind='inference', depth=int(ck['info']['arch']), seed=ck['info']['seed'], order=ck['order'], soft=acc(m), k={})
        for k in K_INFER:
            for layer in m.layers: layer.k_frac = k
            rec['k'][str(k)] = dict(cil=acc(m), stats=route_stats(m, x, routes))
        for layer in m.layers: layer.k_frac = None
        recs.append(rec); print(f.name, 'soft', round(rec['soft'], 3), {k: round(v['cil'], 3) for k, v in rec['k'].items()}, flush=True)
    for f in sorted(Path('results/kwta/ckpt').glob('branch_*_kwta*.pt')):  # trained with competition
        ck, m = load(f); k = float(f.name.split('kwta')[1].split('_')[0])
        for layer in m.layers: layer.k_frac = k
        routes = [q[t] for q in ck['routes']]
        recs.append(dict(kind='trained', depth=int(ck['info']['arch']), k_frac=k, seed=ck['info']['seed'], order=ck['order'], cil=acc(m), stats=route_stats(m, x, routes)))
    Path('results/kwta').mkdir(parents=True, exist_ok=True); Path('results/kwta/analysis.json').write_text(json.dumps(recs, indent=1))

    def ms(G, fn, s=100): mu, sd = mean_sd(seed_means(G, fn).values()); return f'{s * mu:.1f} ± {s * sd:.1f}' if s == 100 else f'{mu:.2f} ± {sd:.2f}'
    L = ['# APL-style global inhibition (k-winners-take-all over a layer\'s branches), Split MNIST test', '',
         'Route hit: share of active branches on the true route (chance 25%). Enrichment: route branches active / all branches active (1 = no task information). Mean ± SD over 3 seed means; layer stats averaged over layers.', '',
         '## Competition at test only (models trained without it)', '', '| layers | active fraction | image only | route hit | enrichment | neurons with an active branch |', '|---:|---:|---:|---:|---:|---:|']
    for d in [1, 2, 3]:
        G = [r for r in recs if r['kind'] == 'inference' and r['depth'] == d]
        if not G: continue
        L.append(f"| {d} | off (softmax) | {ms(G, lambda r: r['soft'])} | | | |")
        for k in K_INFER:
            s = lambda r, key: statistics.mean(l[key] for l in r['k'][str(k)]['stats'])
            L.append(f"| {d} | {k:g} | {ms(G, lambda r: r['k'][str(k)]['cil'])} | {ms(G, lambda r: s(r, 'hit'))} | {ms(G, lambda r: s(r, 'enrichment'), 1)} | {ms(G, lambda r: s(r, 'neurons_on'))} |")
    L += ['', '## Trained with the competition', '', '| layers | active fraction | image only | true route | forgetting | route hit | enrichment | neurons with an active branch |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, k in sorted({(r['depth'], r['k_frac']) for r in recs if r['kind'] == 'trained'}):
        G = [r for r in recs if r['kind'] == 'trained' and r['depth'] == d and r['k_frac'] == k]
        runs = json.loads(Path(f'results/kwta/branch_{d}_kwta{k:g}.json').read_text())
        s = lambda r, key: statistics.mean(l[key] for l in r['stats'])
        L.append(f"| {d} | {k:g} | {ms(G, lambda r: r['cil'])} | {ms(runs, lambda r: r['test']['route']['cil'])} | {ms(runs, lambda r: r['forgetting_til_val'])} | "
                 f"{ms(G, lambda r: s(r, 'hit'))} | {ms(G, lambda r: s(r, 'enrichment'), 1)} | {ms(G, lambda r: s(r, 'neurons_on'))} |")
    out = '\n'.join(L) + '\n'; Path('results/kwta/summary.md').write_text(out); print(out)


if __name__ == '__main__':
    main()
