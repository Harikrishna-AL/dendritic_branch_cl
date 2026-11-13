"""Continual learning on features from a frozen, ImageNet-pretrained ResNet-18.

Question: do cross-task look-alike confusions (classes never trained together) disappear on
good pretrained features, and how does the branch network compare with the standard
pretrained-backbone continual-learning baselines?

Backbone: torchvision ResNet-18, ImageNet weights (never saw MNIST or CIFAR-10 labels),
frozen; 512-d average-pooled features are extracted once. MNIST: 28 -> 112 px, 3 channels.
CIFAR-10: 32 -> 224 px. Both ImageNet-normalised.

Methods on the same features (all legal unless noted):
  branch      branch network, 1 layer of 168 x 4 branches (128 random features per branch);
              image-only soft mixing, and routed by a per-task Gaussian key on the features
  plain       MLP of matched size
  SimpleCIL   cosine nearest class mean (class means stored as each task ends; Zhou et al.)
  RanPAC      random ReLU projection to 5,000-d + ridge classifier from running sums of
              h h^T and h y^T (McDonnell et al. 2023), WITHOUT its first-session backbone
              adaptation (the backbone stays frozen here). Ridge lambda picked on validation.
  probe       linear classifier trained on all tasks jointly (NOT legal; ceiling)

Feature standardisation for the trained models uses the first task's statistics only.

  python scripts/backbone_test.py extract --bench cifar10
  python scripts/backbone_test.py run --bench cifar10 --model branch --seed 201 --first 0
  python scripts/backbone_test.py baselines --bench cifar10      # SimpleCIL, RanPAC, probe
  python scripts/backbone_test.py summarize --bench cifar10
"""
import argparse
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
from branch_routing.access import GaussianKey  # noqa: E402
from branch_routing.continual import BENCH_CONFIG, ContinualTrainer, predict_logits, run_record  # noqa: E402
from branch_routing.data import MNIST_MEAN, MNIST_STD, N_TASKS, SplitData, task_of  # noqa: E402
from branch_routing.models import MLP, BranchMLP  # noqa: E402
from branch_routing.stats import mean_sd, paired_ci, seed_means  # noqa: E402

OUT = Path('results/backbone')
SEEDS = {'mnist': [101, 102, 103], 'cifar10': [201, 202, 203]}
SIZE = {'mnist': 112, 'cifar10': 224}
RANPAC_M = 5000; RANPAC_SIZES = [100, 200, 300, 500, 1000]; RANPAC_LAMBDAS = [1e-1, 1, 10, 100, 1e3, 1e4, 1e5]


def feats_path(bench): return Path(f'data_cache/features/{bench}_resnet18.pt')


def to_images(x, bench):
    if bench == 'mnist': return (x * MNIST_STD + MNIST_MEAN).clamp(0, 1).reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1)
    return x.float() / 255


def extract(bench):
    from torchvision.models import ResNet18_Weights, resnet18
    torch.set_num_threads(4); data = D.load(bench)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = torch.nn.Identity(); net.eval()
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]; std = torch.tensor([.229, .224, .225])[None, :, None, None]
    out = {}
    for split in ['train', 'val', 'test']:
        x, y = data.split(split); fs = []
        with torch.no_grad():
            for j in range(0, len(x), 128):
                img = F.interpolate(to_images(x[j:j + 128], bench), size=SIZE[bench], mode='bilinear', align_corners=False)
                fs.append(net((img - mean) / std))
        out[split] = (torch.cat(fs), y); print(bench, split, out[split][0].shape, flush=True)
    feats_path(bench).parent.mkdir(parents=True, exist_ok=True); torch.save(out, feats_path(bench))


def load_features(bench, first_task):
    f = torch.load(feats_path(bench)); x = f['train'][0]
    ref = x[task_of(f['train'][1]) == first_task]; mu, sd = ref.mean(0), ref.std(0) + 1e-3
    z = lambda a: (a - mu) / sd
    return SplitData(f'{bench}_resnet18', z(x), f['train'][1], z(f['val'][0]), f['val'][1], z(f['test'][0]), f['test'][1]), f


def plain_width(n_params, d_in=512):
    return min(range(16, 600), key=lambda w: abs((d_in + 1) * w + (w + 1) * 10 - n_params))


def error_anatomy(pred, y):
    t = task_of(y); pt = task_of(pred); ok = pt == t; wrong = ~ok
    conf = collections.Counter((int(a), int(b)) for a, b in zip(y[wrong], pred[wrong]))
    return dict(acc=float((pred == y).float().mean()), p_task=float(ok.float().mean()),
                p_class_given_task=float((pred[ok] == y[ok]).float().mean()) if ok.any() else 0.,
                confusions={f'{a}->{b}': c for (a, b), c in conf.most_common(8)})


def run(a):
    torch.set_num_threads(2); torch.set_flush_denormal(True); order = D.task_order(a.first); data, raw = load_features(a.bench, order[0])
    branch = BranchMLP(1, a.seed, d_in=512); n_params = sum(p.numel() for p in branch.parameters())
    m = branch if a.model == 'branch' else MLP(1, plain_width(n_params), a.seed, d_in=512)
    keys = {}

    def on_task_end(task, tr):  # per-task Gaussian key on raw features, fitted as the task ends
        keys[task] = GaussianKey(raw['train'][0][task_of(raw['train'][1]) == task], 'pca', 50)

    run_ = ContinualTrainer(m, BENCH_CONFIG[a.bench], a.seed, on_task_end=on_task_end).fit(data, order)
    rec = run_record(run_, data, order, None, model=a.model, seed=a.seed)
    x, y = data.test_x, data.test_y; t = task_of(y)
    rec['anatomy'] = error_anatomy(predict_logits(m, x, 'soft' if a.model == 'branch' else None).argmax(1), y)
    kt = torch.stack([keys[k].log_likelihood(raw['test'][0]) for k in range(N_TASKS)], 1).argmax(1); rec['key_taskid'] = float((kt == t).float().mean())
    if a.model == 'branch':
        lg = predict_logits(m, x, 'route', run_.routes, kt); idx = torch.arange(len(y))
        rec['key_routed'] = error_anatomy(2 * kt + torch.stack([lg[idx, 2 * kt], lg[idx, 2 * kt + 1]], 1).argmax(1), y)
        rec['route_mass'] = m.route_mass(x, [q[t] for q in run_.routes])
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / f'{a.bench}_{a.model}.json'
    runs = json.loads(f.read_text()) if f.exists() else []; runs.append(rec); f.write_text(json.dumps(runs, indent=1))
    msg = f"{a.bench} {a.model} s{a.seed} o{a.first}: image-only {100 * rec['anatomy']['acc']:.1f} (task right {100 * rec['anatomy']['p_task']:.1f})"
    if a.model == 'branch': msg += f" key-routed {100 * rec['key_routed']['acc']:.1f} route mass {rec['route_mass'][0]:.3f}"
    print(msg + f" forgetting {100 * rec['forgetting_til_val']:.1f}", flush=True)


def ridge_from_sums(h, x, y, vx, vy, tx, ty):
    """Class-incremental ridge readout from running sums G = sum h h^T, C = sum h y^T,
    accumulated one task at a time (no data kept); lambda picked on validation.
    Returns (test anatomy, lambda, numbers stored in G + C)."""
    d = h(x[:1]).shape[1]; G = torch.zeros(d, d, dtype=torch.float64); C = torch.zeros(d, 10, dtype=torch.float64)
    for task in range(N_TASKS):
        ht = h(x[task_of(y) == task]); G += ht.T @ ht; C += ht.T @ F.one_hot(y[task_of(y) == task], 10).double()
    hv, ht_ = h(vx), h(tx); best = None
    for lam in RANPAC_LAMBDAS:
        W = torch.linalg.solve(G + lam * torch.eye(d, dtype=torch.float64), C); acc = float(((hv @ W).argmax(1) == vy).float().mean())
        if best is None or acc > best[0]: best = (acc, lam, W)
    return error_anatomy((ht_ @ best[2]).argmax(1), ty), best[1], G.numel() + C.numel()


def baselines(bench):
    """SimpleCIL, RanPAC (3 projection seeds; full size and memory-matched sizes), ridge on
    the raw features, joint linear probe. Accumulation is exact, so task order does not matter."""
    torch.set_num_threads(4); f = torch.load(feats_path(bench)); x, y = f['train']; vx, vy = f['val']; tx, ty = f['test']; res = {}
    nrm = lambda v: F.normalize(v, dim=1)
    M = torch.stack([nrm(x[y == c]).mean(0) for c in range(10)]); res['simplecil'] = error_anatomy((nrm(tx) @ M.T).argmax(1), ty); res['simplecil']['stored'] = M.numel()
    res['ranpac'] = []; res['ranpac_sweep'] = {}
    for dim in RANPAC_SIZES + [RANPAC_M]:
        rows = []
        for s in SEEDS[bench]:
            R = torch.randn(512, dim, generator=torch.Generator().manual_seed(s)).double()
            r, lam, stored = ridge_from_sums(lambda v: F.relu(v.double() @ R), x, y, vx, vy, tx, ty); r.update(seed=s, lam=lam, stored=stored); rows.append(r)
        if dim == RANPAC_M: res['ranpac'] = rows
        else: res['ranpac_sweep'][str(dim)] = rows
        print(bench, f'RanPAC M={dim}', f"stored {rows[0]['stored']:,}", f"test {100 * statistics.mean(r['acc'] for r in rows):.1f}", flush=True)
    r, lam, stored = ridge_from_sums(lambda v: v.double(), x, y, vx, vy, tx, ty); r.update(lam=lam, stored=stored); res['ridge_raw'] = r
    mu, sd = x.mean(0), x.std(0) + 1e-3; z = lambda v: (v - mu) / sd
    torch.manual_seed(0); W = torch.zeros(512, 10, requires_grad=True); b = torch.zeros(10, requires_grad=True); opt = torch.optim.Adam([W, b], lr=.01)
    for _ in range(500):
        opt.zero_grad(); (F.cross_entropy(z(x) @ W + b, y) + 1e-4 * (W ** 2).sum()).backward(); opt.step()
    with torch.no_grad(): res['probe'] = float(((z(tx) @ W + b).argmax(1) == ty).float().mean())
    (OUT / f'{bench}_baselines.json').write_text(json.dumps(res, indent=1)); print(bench, 'SimpleCIL', round(100 * res['simplecil']['acc'], 1), 'ridge raw', round(100 * r['acc'], 1), 'probe', round(100 * res['probe'], 1))


def summarize(bench):
    B = json.loads((OUT / f'{bench}_branch.json').read_text()); P = json.loads((OUT / f'{bench}_plain.json').read_text())
    base = json.loads((OUT / f'{bench}_baselines.json').read_text()); rp = base['ranpac']
    def ms(G, fn): mu, sd = mean_sd(seed_means(G, fn).values()); return f'{100 * mu:.1f} ± {100 * sd:.1f}'
    def ci(a, b, fa, fb): m, lo, hi = paired_ci(seed_means(a, fa), seed_means(b, fb)); return f'{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]'
    rp_by_seed = {r['seed']: r for r in rp}; rpm = lambda fn: f"{100 * statistics.mean(fn(r) for r in rp):.1f} ± {100 * statistics.stdev(fn(r) for r in rp):.1f}"
    if bench == 'mnist':
        pix = json.loads(Path('results/gap_anatomy.json').read_text())['groups']['default, 1 layer']
        pix_row = f"| branch network on raw pixels (earlier) | {ms(pix, lambda r: r['acc'])} | {ms(pix, lambda r: r['p_task'])} | {ms(pix, lambda r: r['p_class_given_task'])} | 98.5 | 0.2 |"
    else:
        pix = json.loads(Path('results/runs/cifar10/branch_conv_b4_sparse_a0.json').read_text())
        pix_row = f"| branch CNN on raw pixels (earlier) | {ms(pix, lambda r: r['test']['soft']['cil'])} | | | {ms(pix, lambda r: r['test']['route']['til'])} | {ms(pix, lambda r: r['forgetting_til_val'])} |"
    s0 = base['simplecil']; n_branch = sum(p.numel() for p in BranchMLP(1, 0, d_in=512).parameters()); n_routes = N_TASKS * 168; n_keys = N_TASKS * (512 + 50 * 512 + 50 + 1)
    L = [f'# Split {bench.upper()} on frozen ImageNet ResNet-18 features', '',
         'Mean ± SD over 3 seed means (forward + reverse orders). SimpleCIL is deterministic; RanPAC over 3 projection seeds (order-invariant). '
         'RanPAC here has no first-session backbone adaptation.', '',
         '| method | all-10, image only | task right | class right given task | within-pair, task given | forgetting |', '|---|---:|---:|---:|---:|---:|', pix_row,
         f"| branch network, soft mixing (no task info) | {ms(B, lambda r: r['anatomy']['acc'])} | {ms(B, lambda r: r['anatomy']['p_task'])} | {ms(B, lambda r: r['anatomy']['p_class_given_task'])} | {ms(B, lambda r: r['test']['route']['til'])} | {ms(B, lambda r: r['forgetting_til_val'])} |",
         f"| branch network + per-task feature key | {ms(B, lambda r: r['key_routed']['acc'])} | {ms(B, lambda r: r['key_taskid'])} | {ms(B, lambda r: r['key_routed']['p_class_given_task'])} | | |",
         f"| plain MLP, matched size | {ms(P, lambda r: r['anatomy']['acc'])} | {ms(P, lambda r: r['anatomy']['p_task'])} | {ms(P, lambda r: r['anatomy']['p_class_given_task'])} | {ms(P, lambda r: r['test']['plain']['til'])} | {ms(P, lambda r: r['forgetting_til_val'])} |",
         f"| SimpleCIL (cosine class means) | {100 * s0['acc']:.1f} | {100 * s0['p_task']:.1f} | {100 * s0['p_class_given_task']:.1f} | | 0 by construction |",
         f"| RanPAC (random ReLU 5,000-d + ridge) | {rpm(lambda r: r['acc'])} | {rpm(lambda r: r['p_task'])} | {rpm(lambda r: r['p_class_given_task'])} | | 0 by construction |",
         f"| linear probe, all tasks jointly (ceiling, not legal) | {100 * base['probe']:.1f} | | | | |", '',
         f"Branch route mass on these features: {ms(B, lambda r: r['route_mass'][0])} (chance 25.0; on raw MNIST pixels 31.6).", '',
         f"- branch soft − plain MLP: {ci(B, P, lambda r: r['anatomy']['acc'], lambda r: r['anatomy']['acc'])}",
         f"- branch soft − SimpleCIL: {ms(B, lambda r: r['anatomy']['acc'] - s0['acc'])}",
         f"- branch + key − RanPAC (difference of means; RanPAC seeds are projection seeds, so unpaired): {100 * (statistics.mean(seed_means(B, lambda r: r['key_routed']['acc']).values()) - statistics.mean(r['acc'] for r in rp)):+.1f}",
         f"- branch soft − RanPAC (difference of means): {100 * (statistics.mean(seed_means(B, lambda r: r['anatomy']['acc']).values()) - statistics.mean(r['acc'] for r in rp)):+.1f}",
         '', '## Accuracy vs numbers stored', '',
         'Branch network: trained parameters + route indices (+ keys: per-task mean and 50 principal directions of 512-d features). '
         'SimpleCIL: class means. RanPAC/ridge: running sums G and C (the projection is regenerated from a seed).', '',
         '| method | numbers stored | all-10, image only |', '|---|---:|---:|',
         f"| branch network, soft mixing | {n_branch:,} + {n_routes} route indices | {ms(B, lambda r: r['anatomy']['acc'])} |",
         f"| branch network + per-task keys | {n_branch + n_keys:,} | {ms(B, lambda r: r['key_routed']['acc'])} |",
         f"| SimpleCIL | {s0['stored']:,} | {100 * s0['acc']:.1f} |"]
    for dim, rows in sorted(base['ranpac_sweep'].items(), key=lambda kv: int(kv[0])) + [(str(RANPAC_M), rp)]:
        L.append(f"| RanPAC, {dim}-d projection | {rows[0]['stored']:,} | {100 * statistics.mean(r['acc'] for r in rows):.1f} ± {100 * statistics.stdev(r['acc'] for r in rows):.1f} |")
    L += [f"| ridge on raw 512-d features | {base['ridge_raw']['stored']:,} | {100 * base['ridge_raw']['acc']:.1f} |",
          '', '## Top cross-task confusions (true -> predicted, summed over runs)', '']
    for name, G, k in [('branch, soft', B, 'anatomy'), ('branch + key', B, 'key_routed'), ('SimpleCIL', [s0], None), ('RanPAC', rp, None)]:
        c = collections.Counter()
        for r in G: c.update(r[k]['confusions'] if k else r['confusions'])
        L.append(f"- {name}: " + ', '.join(f'{a} ({v})' for a, v in c.most_common(8)))
    out = '\n'.join(L) + '\n'; (OUT / f'{bench}_summary.md').write_text(out); print(out)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    for c in ['extract', 'baselines', 'summarize', 'run']:
        q = sub.add_parser(c); q.add_argument('--bench', choices=list(SEEDS), required=True)
        if c == 'run': q.add_argument('--model', choices=['branch', 'plain'], required=True); q.add_argument('--seed', type=int, required=True); q.add_argument('--first', type=int, required=True)
    a = p.parse_args()
    {'extract': lambda: extract(a.bench), 'baselines': lambda: baselines(a.bench), 'summarize': lambda: summarize(a.bench), 'run': lambda: run(a)}[a.cmd]()


if __name__ == '__main__':
    main()
