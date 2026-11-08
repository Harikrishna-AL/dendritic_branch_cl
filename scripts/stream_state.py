"""Short-lived branch state over unlabelled test streams (Split MNIST, depth-1 BranchMLP).

Trains the 6 branch networks (seeds x orders, same recipe as train.py), caches per-image
branch outputs, then runs every stream method. Each method's hyper-parameters are chosen
on VALIDATION by the mean accuracy over all three stream types (blocked, geometric, mixed),
so a method cannot win by failing on mixed streams.

  python scripts/stream_state.py
"""
import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.access import METHODS, STREAMS, stream_cache, stream_order  # noqa: E402
from branch_routing.continual import BENCH_CONFIG, ContinualTrainer  # noqa: E402
from branch_routing.models import BranchMLP  # noqa: E402
from branch_routing.stats import fmt, mean_sd, paired_ci  # noqa: E402

SEEDS = [101, 102, 103]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('results/stream_state'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2); torch.set_flush_denormal(True)
    data = D.load_mnist(); caches = {}
    for seed in SEEDS:
        for first in [0, 4]:
            model = BranchMLP(1, seed); ContinualTrainer(model, BENCH_CONFIG['mnist'], seed).fit(data, D.task_order(first)); model.eval()
            for split in ['val', 'test']: caches[(seed, first, split)] = stream_cache(model, *data.split(split))
            print('trained + cached', seed, first, flush=True)

    def score(split, method, cfg):
        """{stream: {seed: accuracy averaged over orders}}"""
        make = METHODS[method][0]; res = {}
        for k in STREAMS:
            res[k] = {}
            for seed in SEEDS:
                accs = []
                for first in [0, 4]:
                    c = caches[(seed, first, split)]; o = stream_order(c['y'], seed + first, k)
                    accs.append(float((make(**cfg).run(c, o) == c['y'][o]).mean()))
                res[k][seed] = statistics.mean(accs)
        return res

    objective = lambda r: statistics.mean(statistics.mean(v.values()) for v in r.values())
    validation, final = [], {}
    for method, (_, grid) in METHODS.items():
        trials = [(cfg, score('val', method, cfg)) for cfg in grid()]
        validation += [dict(method=method, cfg=cfg, val=r, objective=objective(r)) for cfg, r in trials]
        cfg, _ = max(trials, key=lambda t: objective(t[1])); final[method] = dict(cfg=cfg, test=score('test', method, cfg))
        print(method, cfg, {k: round(100 * statistics.mean(v.values()), 2) for k, v in final[method]['test'].items()}, flush=True)
    (a.out / 'validation.json').write_text(json.dumps(validation, indent=1, default=str))
    (a.out / 'test.json').write_text(json.dumps(final, indent=1, default=str))

    lines = ['| Method | ' + ' | '.join(STREAMS) + ' | selected (validation) |', '|---|' + '---:|' * len(STREAMS) + '---|']
    for method, r in final.items():
        lines.append(f'| {method} | ' + ' | '.join(fmt(*mean_sd(r['test'][k].values())) for k in STREAMS) + f" | {r['cfg']} |")
    lines.append('')
    for a_, b_ in [('branch_state_gated', 'none'), ('branch_state_gated', 'class_prior_gated')]:
        cis = [paired_ci(final[a_]['test'][k], final[b_]['test'][k]) for k in STREAMS]
        lines.append(f'- {a_} − {b_}: ' + '; '.join(f'{k} {100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]' for k, (m, lo, hi) in zip(STREAMS, cis)))
    (a.out / 'summary.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
