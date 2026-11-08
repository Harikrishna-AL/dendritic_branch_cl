"""Per-task keys: can a key stored at each task boundary pick the right route from one image?

Trains the branch network (same recipe as train.py), fits keys at every task end on that
task's training images only, then routes test images by key likelihood.

  python scripts/task_keys.py --bench mnist
  python scripts/task_keys.py --bench cifar10 --seeds 201 --orders 0
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.access import KeyFeatures, TaskKeyRouter  # noqa: E402
from branch_routing.continual import BENCH_CONFIG, ContinualTrainer, evaluate  # noqa: E402
from branch_routing.models import build_model  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

SEEDS = {'mnist': [101, 102, 103], 'cifar10': [201, 202, 203]}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bench', choices=['mnist', 'cifar10'], required=True)
    p.add_argument('--seeds', type=int, nargs='+'); p.add_argument('--orders', type=int, nargs='+', default=[0, 4])
    p.add_argument('--out', type=Path); p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(); torch.set_num_threads(a.threads); torch.set_flush_denormal(True)
    out = a.out or Path(f'results/task_keys/{a.bench}.json'); out.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(out.read_text()) if out.exists() else []; done = {(r['seed'], r['order'][0]) for r in runs}
    data = D.load(a.bench)
    for seed in a.seeds or SEEDS[a.bench]:
        for first in a.orders:
            if (seed, first) in done: continue
            t0 = time.perf_counter(); order = D.task_order(first); model = build_model(a.bench, 'branch', seed)
            router = TaskKeyRouter(KeyFeatures(model, a.bench, seed))
            fit_key = lambda task, tr: router.fit_task(task, data.x[D.task_of(data.y) == task], tr.routes)
            trainer = ContinualTrainer(model, BENCH_CONFIG[a.bench], seed, on_task_end=fit_key); run = trainer.fit(data, order)
            res = router.evaluate(data.test_x, data.test_y, run.routes, refit_data=(data.x, data.y))
            res['soft_no_key'] = dict(cil=evaluate(model, data.test_x, data.test_y)['soft']['cil'])
            rec = dict(bench=a.bench, seed=seed, order=order, test=res, seconds=time.perf_counter() - t0)
            runs = json.loads(out.read_text()) if out.exists() else []; runs.append(rec); out.write_text(json.dumps(runs, indent=1))
            print(a.bench, seed, first, {k: (round(100 * v['cil'], 1), round(100 * v.get('taskid', float('nan')), 1)) for k, v in res.items()}, f"{rec['seconds']:.0f}s", flush=True)

    summarize(runs, out.with_suffix('.md'))


def summarize(runs, path):
    names = list(runs[0]['test']); lines = ['| Key | CIL | task identified |', '|---|---:|---:|']
    for n in names:
        cil = mean_sd(seed_means(runs, lambda r: r['test'][n]['cil']).values())
        tid = f"{100 * mean_sd(seed_means(runs, lambda r: r['test'][n]['taskid']).values())[0]:.1f}" if 'taskid' in runs[0]['test'][n] else '–'
        lines.append(f'| {n} | {100 * cil[0]:.2f} ± {100 * cil[1]:.2f} | {tid} |')
    path.write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
