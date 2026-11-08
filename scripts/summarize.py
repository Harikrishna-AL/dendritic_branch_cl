"""Summarise training runs (results/runs/<bench>/*.json) into results/summary_<bench>.md.

  python scripts/summarize.py --bench mnist
  python scripts/summarize.py --bench cifar10
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing.stats import mean_sd, paired_ci, seed_means  # noqa: E402

MAIN_ARCH = {'mnist': '1', 'cifar10': 'conv_b4_sparse'}


def cell(runs, fn):
    m, sd = mean_sd(seed_means(runs, fn).values()); return f'{100 * m:.2f} ± {100 * sd:.2f}'


def key_of(r): return 'soft' if r['model'] == 'branch' else 'plain'


def acq_final(r):
    """(mean TIL right after learning, mean final TIL) over all but the last task, validation."""
    k = 'route' if r['model'] == 'branch' else 'plain'; ts = r['order'][:-1]
    return statistics.mean(r['acquisition_til'][t] for t in ts), statistics.mean(r['val'][k]['til_tasks'][t] for t in ts)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bench', choices=['mnist', 'cifar10'], required=True); a = p.parse_args()
    runs = [r for f in sorted(Path(f'results/runs/{a.bench}').glob('*.json')) for r in json.loads(f.read_text())]
    groups = {}
    for r in runs: groups.setdefault((r['model'], str(r['arch']), r['alpha']), []).append(r)
    plain_arch = '1' if a.bench == 'mnist' else 'cnn'; L = [f'# Training runs: {a.bench}', '', f'{len(runs)} runs. ± is the SD over seed means (orders averaged); intervals are paired 95% t over seeds.', '']

    # alpha sweep on validation
    metas = sorted([k for k in groups if k[0] == 'plain' and k[1] == plain_arch], key=lambda k: k[2])
    L += ['## Metaplasticity sweep (validation)', '', '| α | CIL | TIL | learned → final TIL | forgetting | frozen weights |', '|---:|---:|---:|---:|---:|---:|']
    for k in metas:
        g = groups[k]; af = [acq_final(r) for r in g]
        frozen = statistics.mean(r['frozen_fraction'] for r in g) if g[0]['frozen_fraction'] is not None else 0
        L.append(f"| {k[2]:g} | {cell(g, lambda r: r['val']['plain']['cil'])} | {cell(g, lambda r: r['val']['plain']['til'])} | "
                 f"{100 * statistics.mean(x for x, _ in af):.1f} → {100 * statistics.mean(y for _, y in af):.1f} | {cell(g, lambda r: r['forgetting_til_val'])} | {100 * frozen:.0f}% |")
    on = [k for k in metas if k[2] > 0]  # select the control's strength among runs with metaplasticity on
    best_cil = max(on, key=lambda k: statistics.mean(r['val']['plain']['cil'] for r in groups[k]))
    best_til = max(on, key=lambda k: statistics.mean(r['val']['plain']['til'] for r in groups[k]))

    # main comparison on test
    branch = groups[('branch', MAIN_ARCH[a.bench], 0)]
    rows = [('branch network, image only', branch, 'soft'), ('branch network, true route', branch, 'route'),
            ('plain', groups[('plain', plain_arch, 0)], 'plain'),
            (f'plain + metaplasticity, α={best_cil[2]:g} (best val CIL)', groups[best_cil], 'plain'),
            (f'plain + metaplasticity, α={best_til[2]:g} (best val TIL)', groups[best_til], 'plain')]
    L += ['', '## Test', '', '| Model | CIL | TIL | learned → final TIL (val) | forgetting (val) |', '|---|---:|---:|---:|---:|']
    for name, g, k in rows:
        af = [acq_final(r) for r in g]
        L.append(f"| {name} | {cell(g, lambda r: r['test'][k]['cil'])} | {cell(g, lambda r: r['test'][k]['til'])} | "
                 f"{100 * statistics.mean(x for x, _ in af):.1f} → {100 * statistics.mean(y for _, y in af):.1f} | {cell(g, lambda r: r['forgetting_til_val'])} |")
    L += ['', 'Paired differences, branch − metaplasticity control:', '']
    for label, ctrl in [('best val CIL', best_cil), ('best val TIL', best_til)]:
        c = groups[ctrl]
        for what, bk, metric in [('CIL, image only', 'soft', 'cil'), ('TIL, image only', 'soft', 'til'), ('TIL, true route', 'route', 'til')]:
            m, lo, hi = paired_ci(seed_means(branch, lambda r: r['test'][bk][metric]), seed_means(c, lambda r: r['test']['plain'][metric]))
            L.append(f'- vs α={ctrl[2]:g} ({label}), {what}: {100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]')

    # architectures
    L += ['', '## Architectures (test)', '', '| Model | arch | params | learned CIL (val) | CIL image only | CIL true route | forgetting | routes differ |', '|---|---|---:|---:|---:|---:|---:|---|']
    for k in sorted(k for k in groups if k[2] == 0):
        g = groups[k]; acq = statistics.mean(statistics.mean(r['acquisition_cil']) for r in g)
        route = cell(g, lambda r: r['test']['route']['cil']) if 'route' in g[0]['test'] else '–'
        dis = ', '.join(f'{100 * statistics.mean(r["route_disagreement"][i] for r in g):.0f}%' for i in range(len(g[0]['route_disagreement']))) if g[0]['route_disagreement'] else '–'
        L.append(f"| {k[0]} | {k[1]} | {g[0]['params']:,} | {100 * acq:.1f} | {cell(g, lambda r: r['test'][key_of(r)]['cil'])} | {route} | {cell(g, lambda r: r['forgetting_til_val'])} | {dis} |")
    out = Path(f'results/summary_{a.bench}.md'); out.write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
