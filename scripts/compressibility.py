"""How compressible are the hidden representations? A test of the 'fractured representation
causes forgetting' hypothesis (Lehman & Stanley 2026, arXiv 2611.11020) on Split MNIST.

The hypothesis predicts that the network that forgets (the plain MLP, which loses 18-34 points)
has less compressible hidden activations than the one that does not (the branch network).
Compressibility is measured the way UFOs regularise it (Sheppard et al. 2026, arXiv 2611.11862):
a small autoencoder with a 2- or 4-dimensional bottleneck is fitted to z-scored activations,
and we report its error on held-out data as the fraction of variance unexplained (FVU).
Lower FVU = more compressible = more regular.

  across neurons : one sample = one image's layer activation vector (z-scored over neurons);
                   AE fitted on 8,000 test images, evaluated on the other 2,000
  per neuron     : one sample = one neuron's activations over 1,000 fixed probe images
                   (z-scored over images); AE fitted on 80% of neurons, evaluated on 20%

PCA with the same number of dimensions is reported as a linear reference. Each architecture
is also measured at initialisation (untrained, same seed), as a control.
Branch networks are read under the true task's stored route (the representation they use).

Needs checkpoints from train.py --save-checkpoints results/depth/ckpt (branch and plain).
  python scripts/compressibility.py
"""
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
from torch import nn

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.data import task_of  # noqa: E402
from branch_routing.models import build_model  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

CKPT = Path('results/depth/ckpt'); OUT = Path('results/depth')


def zscore(a): return (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-5)


@torch.no_grad()
def hidden(model, x, routes=None, tasks=None):
    """Per-layer hidden activations (N, width)."""
    if getattr(model, 'is_branch', False):
        r = [q[tasks] for q in routes] if routes is not None else None
        _, hs, _, _ = model(x, 'route' if r is not None else 'trace', r, internals=True); return list(hs)
    out, h = [], x
    for layer in model.layers: h = layer(h).tanh(); out.append(h)
    return out


def fvu(pred, target): return float(((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())


def ae_fvu(train, test, k, hidden_units=64, steps=1500, seed=0):
    torch.manual_seed(seed); d = train.shape[1]
    ae = nn.Sequential(nn.Linear(d, hidden_units), nn.GELU(), nn.Linear(hidden_units, k), nn.Linear(k, hidden_units), nn.GELU(), nn.Linear(hidden_units, d))
    opt = torch.optim.Adam(ae.parameters(), 1e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = train[torch.randint(len(train), (min(512, len(train)),), generator=g)]
        opt.zero_grad(); ((ae(b) - b) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): return fvu(ae(test), test)


def pca_fvu(train, test, k):
    mu = train.mean(0); _, _, V = torch.linalg.svd(train - mu, full_matrices=False); P = V[:k].T
    return fvu((test - mu) @ P @ P.T + mu, test)


def measure(acts, seed):
    g = torch.Generator().manual_seed(seed); res = []
    for a in acts:
        inter = zscore(a); perm = torch.randperm(len(inter), generator=g); tr, te = inter[perm[:8000]], inter[perm[8000:]]
        intra = zscore(a[:1000].T.contiguous()); pn = torch.randperm(len(intra), generator=g); cut = int(.8 * len(intra))
        itr, ite = intra[pn[:cut]], intra[pn[cut:]]
        res.append(dict(width=a.shape[1],
                        across_ae2=ae_fvu(tr, te, 2, seed=seed), across_ae4=ae_fvu(tr, te, 4, seed=seed),
                        across_pca2=pca_fvu(tr, te, 2), across_pca4=pca_fvu(tr, te, 4),
                        neuron_ae2=ae_fvu(itr, ite, 2, hidden_units=128, steps=1500, seed=seed), neuron_pca2=pca_fvu(itr, ite, 2)))
    return res


METRICS = ['across_ae2', 'across_ae4', 'across_pca2', 'across_pca4', 'neuron_ae2', 'neuron_pca2']


def summary(recs, forgetting):
    L = ['# Compressibility of hidden representations (Split MNIST, test images)', '',
         'Held-out fraction of variance unexplained (FVU) by a 2- or 4-dim bottleneck; lower = more compressible.',
         'Mean over layers, then mean ± SD over 3 seed means. "init" = same architecture and seed, untrained.', '',
         '| model | layers | state | forgetting | across: AE-2 | across: AE-4 | across: PCA-2 | across: PCA-4 | per neuron: AE-2 | per neuron: PCA-2 |',
         '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in ['branch', 'plain']:
        for depth in [1, 2, 3]:
            for state in ['trained', 'init']:
                R = [r for r in recs if r['model'] == model and r['depth'] == depth and r['state'] == state]
                if not R: continue
                cells = []
                for m in METRICS:
                    mu, sd = mean_sd(seed_means(R, lambda r: statistics.mean(l[m] for l in r['layers'])).values()); cells.append(f'{mu:.3f} ± {sd:.3f}')
                f = f"{100 * forgetting[(model, depth)]:.1f}" if state == 'trained' else ''
                L.append(f'| {model} | {depth} | {state} | {f} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Per layer (trained, across neurons, AE-2)', '', '| model | layers | layer 1 | layer 2 | layer 3 |', '|---|---:|---:|---:|---:|']
    for model in ['branch', 'plain']:
        for depth in [1, 2, 3]:
            R = [r for r in recs if r['model'] == model and r['depth'] == depth and r['state'] == 'trained']
            L.append(f'| {model} | {depth} | ' + ' | '.join(f"{statistics.mean(r['layers'][i]['across_ae2'] for r in R):.3f}" for i in range(depth)) + ' |' * (3 - depth + 1))
    return '\n'.join(L) + '\n'


def main():
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist')
    x, y = data.test_x, data.test_y; recs = []; forgetting = {}
    for model_kind in ['branch', 'plain']:
        for depth in [1, 2, 3]:
            runs = json.loads(Path(f"results/depth/{'runs' if model_kind == 'branch' else 'runs_plain'}_d{depth}.json").read_text())
            forgetting[(model_kind, depth)] = statistics.mean(r['forgetting_til_val'] for r in runs)
            for f in sorted(CKPT.glob(f'{model_kind}_{depth}_*.pt')):
                ck = torch.load(f, weights_only=False); seed = ck['info']['seed']
                m = build_model('mnist', model_kind, seed, str(depth)); m.load_state_dict(ck['state']); m.eval()
                acts = hidden(m, x, ck['routes'], task_of(y)) if model_kind == 'branch' else hidden(m, x)
                recs.append(dict(model=model_kind, depth=depth, state='trained', seed=seed, order=ck['order'], layers=measure(acts, seed)))
                if ck['order'][0] == 0:  # init control once per seed
                    m0 = build_model('mnist', model_kind, seed, str(depth)); m0.eval()
                    recs.append(dict(model=model_kind, depth=depth, state='init', seed=seed, order=ck['order'], layers=measure(hidden(m0, x), seed)))
                print(f.name, f"across AE-2 {statistics.mean(l['across_ae2'] for l in recs[-1]['layers']):.3f}", flush=True)
    (OUT / 'compressibility.json').write_text(json.dumps(recs, indent=1)); s = summary(recs, forgetting); (OUT / 'compressibility.md').write_text(s); print(s)


if __name__ == '__main__':
    main()
