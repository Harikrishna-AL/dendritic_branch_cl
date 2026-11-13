"""Upper bound for gating on branch response strength: the gate loss with REAL old images.

NOT continual-learning legal (it replays old images), and meant only as a ceiling.
During every task after the first, each batch adds n_old images sampled from the earlier
tasks' training data. On those images the gate loss pulls softmax(|response|) towards
their own task's stored route (upstream layers mixing softly, as at test time). That gives
the inhibition missing in the legal version: the new task's branches learn to be quiet on
old images, and old routes keep their share. Only the gate term sees old images; the
classification loss is unchanged, so this bounds gating, not replay in general.

  python scripts/gate_upper_bound.py run --depth 1 --cur 1 --old 1 --seed 101 --first 0
  python scripts/gate_upper_bound.py summarize
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
from branch_routing.continual import BENCH_CONFIG, ContinualTrainer, predict_logits, run_record  # noqa: E402
from branch_routing.data import task_of  # noqa: E402
from branch_routing.models import build_model  # noqa: E402
from branch_routing.stats import mean_sd, seed_means  # noqa: E402
from route_calibration import route_scores, routed_predictions  # noqa: E402

OUT = Path('results/gate_loss/upper_bound')


class OracleGateTrainer(ContinualTrainer):
    def __init__(self, model, cfg, seed, data, old_weight, n_old=128):
        super().__init__(model, cfg, seed, on_task_end=lambda task, tr: tr.seen.append(task))
        self.data, self.old_weight, self.n_old, self.seen = data, old_weight, n_old, []
        self.gen = torch.Generator().manual_seed(seed + 7)

    def old_gate_loss(self):
        pool = torch.where(torch.isin(task_of(self.data.y), torch.tensor(self.seen)))[0]
        ids = pool[torch.randint(len(pool), (self.n_old,), generator=self.gen)]
        x, t = self.data.x[ids], task_of(self.data.y[ids])
        _, _, _, raws = self.model(x, 'soft', internals=True); total = 0.
        for raw, r in zip(raws, self.routes):
            total = total - raw.abs().log_softmax(-1).gather(2, r[t][..., None]).mean()
        return total / len(raws)

    def train_step(self, x, y):
        m = self.model; self.opt.zero_grad(set_to_none=True)
        loss, feedback = m.substitution_loss(x, y, self.cfg.gate_weight, self.cfg.gate_mode)
        if self.seen and self.old_weight: loss = loss + self.old_weight * self.old_gate_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), self.cfg.clip); self.opt.step()
        m.update_traces(feedback, self.cfg.trace_decay); return loss.item()


def run(a):
    torch.set_num_threads(2); torch.set_flush_denormal(True); data = D.load('mnist')
    m = build_model('mnist', 'branch', a.seed, str(a.depth)); order = D.task_order(a.first)
    cfg = dataclasses.replace(BENCH_CONFIG['mnist'], gate_weight=a.cur, gate_mode='full')
    run_ = OracleGateTrainer(m, cfg, a.seed, data, a.old).fit(data, order)
    rec = run_record(run_, data, order, None, depth=a.depth, cur=a.cur, old=a.old, seed=a.seed)
    x, y = data.test_x, data.test_y; t = task_of(y)
    rec['route_mass'] = m.route_mass(x, [q[t] for q in run_.routes])
    s = route_scores(m, x, run_.routes); tk = s.argmax(1); pred = routed_predictions(m, x, run_.routes)
    rec['pick_route'] = dict(cil=float((pred.gather(1, tk[:, None]).squeeze(1) == y).float().mean()), taskid=float((tk == t).float().mean()))
    pt = task_of(predict_logits(m, x, 'soft').argmax(1)); rec['soft_last_task_share'] = float((pt == order[-1]).float().mean())
    if a.save:
        ck = OUT / 'ckpt'; ck.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=m.state_dict(), routes=run_.routes, info=dict(arch=str(a.depth), seed=a.seed, cur=a.cur, old=a.old), order=order), ck / f'd{a.depth}_cur{a.cur:g}_old{a.old:g}_{a.seed}_{a.first}.pt')
        return
    f = OUT / f'd{a.depth}_cur{a.cur:g}_old{a.old:g}.json'; OUT.mkdir(parents=True, exist_ok=True)
    runs = json.loads(f.read_text()) if f.exists() else []; runs.append(rec); f.write_text(json.dumps(runs, indent=1))
    print(f"d{a.depth} cur{a.cur:g} old{a.old:g} s{a.seed} o{a.first}: soft {100 * rec['test']['soft']['cil']:.1f} pick-route {100 * rec['pick_route']['cil']:.1f} "
          f"true-route {100 * rec['test']['route']['cil']:.1f} forgetting {100 * rec['forgetting_til_val']:.1f} route mass {[round(v, 3) for v in rec['route_mass']]} "
          f"last-task share {100 * rec['soft_last_task_share']:.0f}%", flush=True)


def summarize():
    R = [r for f in sorted(OUT.glob('*.json')) for r in json.loads(f.read_text())]
    def ms(G, fn): m, sd = mean_sd(seed_means(G, fn).values()); return f'{100 * m:.1f} ± {100 * sd:.1f}'
    L = ['# Upper bound: gate loss with real old images (NOT legal; ceiling only)', '',
         'cur = gate-loss weight on the current task; old = weight on replayed old images (target = their stored route). '
         'Classification loss never sees old images. Split MNIST test, mean ± SD over 3 seed means.', '',
         '| layers | cur | old | image only (soft) | pick one route by route mass | true route | forgetting | route mass (mean over layers) | soft picks last task |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in sorted({(r['depth'], r['cur'], r['old']) for r in R}):
        G = [r for r in R if (r['depth'], r['cur'], r['old']) == key]
        L.append(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {ms(G, lambda r: r['test']['soft']['cil'])} | {ms(G, lambda r: r['pick_route']['cil'])} | "
                 f"{ms(G, lambda r: r['test']['route']['cil'])} | {ms(G, lambda r: r['forgetting_til_val'])} | {ms(G, lambda r: statistics.mean(r['route_mass']))} | "
                 f"{ms(G, lambda r: r['soft_last_task_share'])} |")
    s = '\n'.join(L) + '\n'; (OUT.parent / 'upper_bound.md').write_text(s); print(s)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--depth', type=int, default=1); r.add_argument('--cur', type=float, default=1)
    r.add_argument('--old', type=float, default=1); r.add_argument('--seed', type=int, required=True); r.add_argument('--first', type=int, required=True)
    r.add_argument('--save', action='store_true', help='save a checkpoint instead of appending results (for gap_anatomy.py)')
    sub.add_parser('summarize'); a = p.parse_args()
    run(a) if a.cmd == 'run' else summarize()


if __name__ == '__main__':
    main()
