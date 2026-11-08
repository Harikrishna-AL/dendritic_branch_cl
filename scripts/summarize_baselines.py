"""Summarise the HAT comparison and the gate-rule ablation -> results/summary_baselines.md

  python scripts/summarize_baselines.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing.stats import mean_sd, paired_ci, seed_means  # noqa: E402

MAIN = {'mnist': 'results/runs/mnist/branch_1_a0.json', 'cifar10': 'results/runs/cifar10/branch_conv_b4_sparse_a0.json'}


def load(path): return json.loads(Path(path).read_text())


def cell(runs, fn):
    m, sd = mean_sd(seed_means(runs, fn).values()); return f'{100 * m:.2f} ± {100 * sd:.2f}'


def learned_kept(runs):
    a = statistics.mean(statistics.mean(r['acquisition_til'][t] for t in r['order'][:-1]) for r in runs)
    k = statistics.mean(statistics.mean(r['val']['route']['til_tasks'][t] for t in r['order'][:-1]) for r in runs)
    return f'{100 * a:.1f} → {100 * k:.1f}'


def ci(a, b, fa, fb):
    m, lo, hi = paired_ci(seed_means(a, fa), seed_means(b, fb)); return f'{100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]'


def main():
    L = ['# Baselines at matched size', '', '± = SD over 3 seed means (forward + reverse order averaged); [a, b] = paired 95% t-interval.', '']
    for bench in ['mnist', 'cifar10']:
        files = sorted(Path(f'results/runs/{bench}_hat').glob('*.json'))
        if not files: continue
        groups = {f.stem.split('_a0_')[-1]: load(f) for f in files}
        L += [f'## HAT vs branch network: {bench}', '', 'HAT settings (validation, true task given):', '', '| setting | within-pair | all-10 |', '|---|---:|---:|']
        for v, g in groups.items(): L.append(f"| {v} | {cell(g, lambda r: r['val']['route']['til'])} | {cell(g, lambda r: r['val']['route']['cil'])} |")
        best = max(groups, key=lambda v: statistics.mean(r['val']['route']['til'] for r in groups[v])); hat = groups[best]
        branch = load(MAIN[bench]); keys = {(r['seed'], r['order'][0]): r['test']['pixel_pca']['cil'] for r in load(f'results/task_keys/{bench}.json')}
        bkey = lambda r: keys[(r['seed'], r['order'][0])]
        L += ['', f'Selected HAT setting: {best}. Test:', '', '| Model | params | all-10, true task | within-pair, true task | all-10, pixel-key router | learned → kept | forgetting |',
              '|---|---:|---:|---:|---:|---:|---:|',
              f"| branch network | {branch[0]['params']:,} | {cell(branch, lambda r: r['test']['route']['cil'])} | {cell(branch, lambda r: r['test']['route']['til'])} | {cell(branch, bkey)} | {learned_kept(branch)} | {cell(branch, lambda r: r['forgetting_til_val'])} |",
              f"| HAT | {hat[0]['params']:,} | {cell(hat, lambda r: r['test']['route']['cil'])} | {cell(hat, lambda r: r['test']['route']['til'])} | {cell(hat, lambda r: r['key_routed']['test'])} | {learned_kept(hat)} | {cell(hat, lambda r: r['forgetting_til_val'])} |",
              f"| branch network, image only (no task, no key) | | {cell(branch, lambda r: r['test']['soft']['cil'])} | | | | |",
              '', 'Branch − HAT:', '',
              f"- all-10, true task: {ci(branch, hat, lambda r: r['test']['route']['cil'], lambda r: r['test']['route']['cil'])}",
              f"- within-pair, true task: {ci(branch, hat, lambda r: r['test']['route']['til'], lambda r: r['test']['route']['til'])}",
              f"- all-10, pixel-key router: {ci(branch, hat, bkey, lambda r: r['key_routed']['test'])}", '']
    files = sorted(Path('results/runs/mnist_gates').glob('*.json'))
    if files:
        L += ['## Gate-rule ablation (MNIST, test)', '', '| Training rule | all-10, image only | all-10, pixel-key router | all-10, true route | forgetting | routes differ |', '|---|---:|---:|---:|---:|---:|']
        for f in files:
            g = load(f); rule = g[0]['variant']; dis = statistics.mean(r['route_disagreement'][0] for r in g)
            L.append(f"| {rule} | {cell(g, lambda r: r['test']['soft']['cil'])} | {cell(g, lambda r: r['key_routed']['test'])} | {cell(g, lambda r: r['test']['route']['cil'])} | {cell(g, lambda r: r['forgetting_til_val'])} | {100 * dis:.0f}% |")
    out = Path('results/summary_baselines.md'); out.write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
