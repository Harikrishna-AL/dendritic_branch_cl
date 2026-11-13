"""Option 1 against recency bias: rebalance route strengths at test time.

Route mass of task t's route on image x: mean over layers and neurons of the soft gate
weight softmax(|r|) on the branch that route t uses (all layers mixing softly, as in
image-only inference). The image is assigned to the task whose route has the highest score,
and is then classified with that task's stored route, within the task's two classes.

Scores compared:
  raw         s_t(x)
  calibrated  s_t(x) / m_t, where m_t is the mean of s_t over task t's training images,
              recorded at the END OF TASK t with the weights of that time (one number per
              task; continual-learning legal, no data stored)
  calib_now   same, but m_t recomputed after all training on task t's training images
              (NOT legal: uses old data; shows how much drift of m_t matters)

  python scripts/route_calibration.py run --depth 1 --gate-loss 1 --gate-mode full --seed 101 --first 0
  python scripts/route_calibration.py summarize
"""
import argparse
import dataclasses
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.continual import BENCH_CONFIG, ContinualTrainer, predict_logits  # noqa: E402
from branch_routing.data import N_TASKS, task_of  # noqa: E402
from branch_routing.models import build_model  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402

OUT = Path('results/gate_loss/calibration')  # one JSON per configuration (parallel-safe)


@torch.no_grad()
def route_scores(m, x, routes, chunk=500) -> torch.Tensor:
    """(N, n_tasks): route mass of every task's route, upstream layers mixing softly."""
    was = m.training; m.eval(); out = []
    for j in range(0, len(x), chunk):
        _, _, _, raws = m(x[j:j + chunk], 'soft', internals=True)
        w = [raw.abs().softmax(-1) for raw in raws]
        out.append(torch.stack([torch.stack([wl.gather(2, r[t].expand(len(wl), -1)[..., None]).squeeze(2).mean(1) for wl, r in zip(w, routes)]).mean(0)
                                for t in range(N_TASKS)], 1))
    m.train(was); return torch.cat(out)


@torch.no_grad()
def routed_predictions(m, x, routes) -> torch.Tensor:
    """(N, n_tasks): within-pair prediction under each task's stored route."""
    preds = []
    for t in range(N_TASKS):
        lg = predict_logits(m, x, 'route', routes, torch.full((len(x),), t)); preds.append(2 * t + lg[:, 2 * t:2 * t + 2].argmax(1))
    return torch.stack(preds, 1)


def run(a):
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist')
    m = build_model('mnist', 'branch', a.seed, str(a.depth))
    cfg = dataclasses.replace(BENCH_CONFIG['mnist'], gate_weight=a.gate_loss, gate_mode=a.gate_mode); own = {}

    def on_task_end(task, trainer):  # m_t with the weights at the end of task t
        xt = data.x[task_of(data.y) == task][:3000]; own[task] = float(route_scores(m, xt, trainer.routes)[:, task].mean())

    order = D.task_order(a.first); run_ = ContinualTrainer(m, cfg, a.seed, on_task_end=on_task_end).fit(data, order); routes = run_.routes
    own_now = {t: float(route_scores(m, data.x[task_of(data.y) == t][:3000], routes)[:, t].mean()) for t in range(N_TASKS)}
    rec = dict(depth=a.depth, gate_loss=a.gate_loss, gate_mode=a.gate_mode, seed=a.seed, order=order, own_at_end=own, own_now=own_now)
    for split in ['val', 'test']:
        x, y = data.split(split); s = route_scores(m, x, routes); pred = routed_predictions(m, x, routes); true_t = task_of(y)
        soft = float((predict_logits(m, x, 'soft').argmax(1) == y).float().mean())
        res = dict(soft=soft, true_route=float((pred.gather(1, true_t[:, None]).squeeze(1) == y).float().mean()))
        for name, sc in [('raw', s), ('calibrated', s / torch.tensor([own[t] for t in range(N_TASKS)])),
                         ('calib_now', s / torch.tensor([own_now[t] for t in range(N_TASKS)]))]:
            tk = sc.argmax(1); res[name] = dict(cil=float((pred.gather(1, tk[:, None]).squeeze(1) == y).float().mean()),
                                                 taskid=float((tk == true_t).float().mean()),
                                                 picked_share=[float((tk == t).float().mean()) for t in range(N_TASKS)])
        rec[split] = res
    f = OUT / f'd{a.depth}_gate{a.gate_loss:g}_{a.gate_mode}.json'; OUT.mkdir(parents=True, exist_ok=True)
    runs = json.loads(f.read_text()) if f.exists() else []; runs.append(rec); f.write_text(json.dumps(runs, indent=1))
    t = rec['test']; print(f"d{a.depth} gate{a.gate_loss:g}_{a.gate_mode} s{a.seed} o{a.first}: soft {100 * t['soft']:.1f} raw {100 * t['raw']['cil']:.1f} "
                           f"calibrated {100 * t['calibrated']['cil']:.1f} calib_now {100 * t['calib_now']['cil']:.1f} own {[round(own[k], 3) for k in range(N_TASKS)]}", flush=True)


def summarize():
    R = [r for f in sorted(OUT.glob('*.json')) for r in json.loads(f.read_text())]; keys = sorted({(r['depth'], r['gate_loss'], r['gate_mode']) for r in R})
    L = ['# Route-strength calibration (Split MNIST, test)', '',
         'Task chosen by route mass (raw), by route mass divided by its own-task value recorded at the end of that task (calibrated, legal), '
         'or recomputed after training (calib_now, not legal). Then classified with that task\'s route. Mean ± SD over 3 seed means.', '',
         '| layers | gate loss | image only (soft) | raw: all-10 | raw: task right | calibrated: all-10 | calibrated: task right | calib_now: all-10 | last task picked (calibrated) | true route |',
         '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def ms(G, fn): m, sd = mean_sd(seed_means(G, fn).values()); return f'{100 * m:.1f} ± {100 * sd:.1f}'
    for d, g, mode in keys:
        G = [r for r in R if (r['depth'], r['gate_loss'], r['gate_mode']) == (d, g, mode)]; t = lambda r: r['test']
        L.append(f"| {d} | {'off' if not g else f'{g:g} {mode}'} | {ms(G, lambda r: t(r)['soft'])} | {ms(G, lambda r: t(r)['raw']['cil'])} | {ms(G, lambda r: t(r)['raw']['taskid'])} | "
                 f"{ms(G, lambda r: t(r)['calibrated']['cil'])} | {ms(G, lambda r: t(r)['calibrated']['taskid'])} | {ms(G, lambda r: t(r)['calib_now']['cil'])} | "
                 f"{ms(G, lambda r: t(r)['calibrated']['picked_share'][r['order'][-1]])} | {ms(G, lambda r: t(r)['true_route'])} |")
    s = '\n'.join(L) + '\n'; Path('results/gate_loss/calibration.md').write_text(s); print(s)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--depth', type=int, default=1); r.add_argument('--gate-loss', type=float, default=0)
    r.add_argument('--gate-mode', default='full'); r.add_argument('--seed', type=int, required=True); r.add_argument('--first', type=int, required=True)
    sub.add_parser('summarize'); a = p.parse_args()
    run(a) if a.cmd == 'run' else summarize()


if __name__ == '__main__':
    main()
