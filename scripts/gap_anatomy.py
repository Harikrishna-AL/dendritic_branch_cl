"""Where does the gap between image-only and true-route accuracy come from?

For branch networks (default training, and the not-legal upper bound that trains the gate
with replayed old images), on Split MNIST test images:

  1. error types        image-only accuracy = P(task right) x P(class right | task right);
                        wrong-task errors split into "predicted the last task" and others
  2. digit confusions   most frequent cross-task confusions (true digit -> predicted digit)
  3. information probes linear softmax probes trained on ALL tasks' training images
                        (NOT legal: a measurement of what the representation contains)
                          h_last   : last layer's soft-mixed output (what the head sees)
                          branches : every branch response of every layer (what gating sees)
                          pixels   : raw pixels (reference)
                        A probe near the true-route accuracy means the information is present
                        and only the continual readout/gating fails to use it.

  python scripts/gap_anatomy.py
"""
import collections
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
from branch_routing.data import task_of  # noqa: E402
from branch_routing.models import BranchMLP  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

GROUPS = {'default, 1 layer': 'results/depth/ckpt/branch_1_1*.pt', 'default, 3 layers': 'results/depth/ckpt/branch_3_1*.pt',
          'upper bound, 1 layer (1/10)': 'results/gate_loss/upper_bound/ckpt/d1_cur1_old10_*.pt',
          'upper bound, 3 layers (1/1)': 'results/gate_loss/upper_bound/ckpt/d3_cur1_old1_*.pt'}
N_PROBE = 20000


@torch.no_grad()
def features(m, x, chunk=1000):
    """logits (soft), last-layer soft output, all branch responses (soft upstream)."""
    lg, hl, br = [], [], []
    for j in range(0, len(x), chunk):
        logits, hs, _, raws = m(x[j:j + chunk], 'soft', internals=True)
        lg.append(logits); hl.append(hs[-1]); br.append(torch.cat([r.flatten(1) for r in raws], 1))
    return torch.cat(lg), torch.cat(hl), torch.cat(br)


def probe(ftr, ytr, fte, yte, n_out, steps=300):
    """Multinomial logistic regression (standardised features, Adam, light L2)."""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-5; ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    torch.manual_seed(0); W = torch.zeros(ftr.shape[1], n_out, requires_grad=True); b = torch.zeros(n_out, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=.01)
    for _ in range(steps):
        opt.zero_grad(); (F.cross_entropy(ftr @ W + b, ytr) + 1e-4 * (W ** 2).sum()).backward(); opt.step()
    with torch.no_grad(): return float(((fte @ W + b).argmax(1) == yte).float().mean())


def analyse(f, data, ix):
    ck = torch.load(f, weights_only=False); d = int(ck['info']['arch']); m = BranchMLP(d, ck['info']['seed']); m.load_state_dict(ck['state']); m.eval()
    x, y = data.test_x, data.test_y; t = task_of(y); last = ck['order'][-1]
    lg, hl, br = features(m, x); pred = lg.argmax(1); pt = task_of(pred)
    task_ok = pt == t; wrong_task = ~task_ok
    conf = collections.Counter((int(a), int(b)) for a, b in zip(y[wrong_task], pred[wrong_task]))
    _, hl_tr, br_tr = features(m, data.x[ix]); ytr = data.y[ix]
    return dict(seed=ck['info']['seed'], order=ck['order'], depth=d, acc=float((pred == y).float().mean()),
                p_task=float(task_ok.float().mean()), p_class_given_task=float((pred[task_ok] == y[task_ok]).float().mean()),
                wrong_task_last=float((wrong_task & (pt == last)).float().mean()), wrong_task_other=float((wrong_task & (pt != last)).float().mean()),
                within_task_err=float((task_ok & (pred != y)).float().mean()), confusions={f'{a}->{b}': c for (a, b), c in conf.most_common(8)},
                probe_h_class=probe(hl_tr, ytr, hl, y, 10), probe_h_task=probe(hl_tr, task_of(ytr), hl, t, 5),
                probe_branch_class=probe(br_tr, ytr, br, y, 10), probe_branch_task=probe(br_tr, task_of(ytr), br, t, 5))


def main():
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist')
    ix = torch.randperm(len(data.x), generator=torch.Generator().manual_seed(0))[:N_PROBE]
    pix = dict(cls=probe(data.x[ix], data.y[ix], data.test_x, data.test_y, 10), task=probe(data.x[ix], task_of(data.y[ix]), data.test_x, task_of(data.test_y), 5))
    out = {}
    for name, pat in GROUPS.items():
        files = sorted(Path('.').glob(pat))
        out[name] = [analyse(f, data, ix) for f in files]; print(name, len(files), flush=True)
    Path('results/gap_anatomy.json').write_text(json.dumps(dict(pixels=pix, groups=out), indent=1))

    def ms(G, fn): mu, sd = mean_sd(seed_means(G, fn).values()); return f'{100 * mu:.1f} ± {100 * sd:.1f}'
    L = ['# Anatomy of the image-only gap (Split MNIST test)', '', 'Mean ± SD over 3 seed means. Probes are linear, trained on 20,000 training images from ALL tasks (not legal; they measure information content).', '',
         '## 1. Error types', '', '| model | image only | task right | class right, given task right | errors: wrong task = last task | errors: other wrong task | errors: right task, wrong class |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, G in out.items():
        if G: L.append(f"| {name} | {ms(G, lambda r: r['acc'])} | {ms(G, lambda r: r['p_task'])} | {ms(G, lambda r: r['p_class_given_task'])} | {ms(G, lambda r: r['wrong_task_last'])} | {ms(G, lambda r: r['wrong_task_other'])} | {ms(G, lambda r: r['within_task_err'])} |")
    L += ['', '## 2. Most common cross-task confusions (true -> predicted, total over runs)', '']
    for name, G in out.items():
        if not G: continue
        c = collections.Counter()
        for r in G: c.update(r['confusions'])
        n_err = sum(c.values()); L.append(f"- {name}: " + ', '.join(f'{k} ({v})' for k, v in c.most_common(8)) + f'  [top-8 lists cover {n_err} errors]')
    L += ['', '## 3. Is the information there? Linear probes (all-task training data)', '',
          f"Pixels: 10-class {100 * pix['cls']:.1f}, task {100 * pix['task']:.1f}.", '',
          '| model | image only | probe on last-layer output: 10-class | ...: task | probe on all branch responses: 10-class | ...: task |', '|---|---:|---:|---:|---:|---:|']
    for name, G in out.items():
        if G: L.append(f"| {name} | {ms(G, lambda r: r['acc'])} | {ms(G, lambda r: r['probe_h_class'])} | {ms(G, lambda r: r['probe_h_task'])} | {ms(G, lambda r: r['probe_branch_class'])} | {ms(G, lambda r: r['probe_branch_task'])} |")
    s = '\n'.join(L) + '\n'; Path('results/gap_anatomy.md').write_text(s); print(s)


if __name__ == '__main__':
    main()
