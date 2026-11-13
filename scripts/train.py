"""Train and evaluate continual-learning runs; append one JSON record per run.

Examples
  python scripts/train.py --bench mnist --model branch --arch 1
  python scripts/train.py --bench mnist --model plain --alpha 1000          # + metaplasticity control
  python scripts/train.py --bench cifar10 --model branch --arch conv_b4_sparse
  python scripts/train.py --bench cifar10 --model plain --seeds 201 --orders 0
  python scripts/train.py --bench mnist --model hat --lr 0.1 --dropout 1          # HAT baseline
  python scripts/train.py --bench mnist --model branch --gate-rule most_active    # gate ablation

Runs already in --out are skipped, so an interrupted sweep can be restarted.
"""
import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')  # side effect of flush-denormal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.access import pixel_key_tasks, routed_accuracy  # noqa: E402
from branch_routing.continual import BENCH_CONFIG, GATE_RULES, ContinualTrainer, GateRule, HATTrainer, predict_hat, predict_logits, run_record  # noqa: E402
from branch_routing.models import build_hat, build_model, patch_indices  # noqa: E402

DEFAULT_SEEDS = {'mnist': [101, 102, 103], 'cifar10': [201, 202, 203]}


def run_key(r): return (r['model'], str(r['arch']), r['alpha'], r.get('variant', ''), r['seed'], r['order'][0])


def key_routed(model, run, data, split, tasks):
    """Accuracy when the task is chosen by the pixel key (same router for every model)."""
    x, y = data.split(split)
    logits = predict_hat(model, x, tasks) if getattr(model, 'uses_task', False) else predict_logits(model, x, 'route', run.routes, tasks)
    return routed_accuracy(logits, tasks, y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bench', choices=['mnist', 'cifar10'], required=True)
    p.add_argument('--model', choices=['branch', 'plain', 'hat'], required=True)
    p.add_argument('--arch', default=None, help="MNIST: depth (1, 2, 3). CIFAR branch: conv_b4_sparse | conv_b4 | conv_b1_wide")
    p.add_argument('--alpha', type=float, default=0, help='metaplasticity strength (plain models only; 0 = off)')
    p.add_argument('--gate-rule', choices=GATE_RULES, help='branch models: ablation of how the training branch is chosen')
    p.add_argument('--gate-loss', type=float, default=0, help='branch models: weight of the gate loss (raises route mass; 0 = off)')
    p.add_argument('--kwta', type=float, default=0, help='branch MLP: global k-winners-take-all fraction of active branches per layer (0 = off)')
    p.add_argument('--input-layout', choices=['random', 'patch'], default='random', help='MNIST branch MLP, first layer: random 196-pixel subsets or 14x14 patches')
    p.add_argument('--gate-mode', choices=['full', 'up'], default='full', help="gate loss: 'up' only pushes the learning branch up")
    p.add_argument('--lr', type=float, default=.1, help='HAT only (SGD)'); p.add_argument('--dropout', type=int, default=1, help='HAT only')
    p.add_argument('--key-routing', action='store_true', help='also report accuracy with the pixel-key router')
    p.add_argument('--seeds', type=int, nargs='+'); p.add_argument('--orders', type=int, nargs='+', default=[0, 4], help='0 = forward, 4 = reverse')
    p.add_argument('--out', type=Path, help='JSON file to append to (default results/runs/<bench>/<model>_<arch>_a<alpha>.json)')
    p.add_argument('--save-checkpoints', type=Path, help='directory for model state + routes')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(); torch.set_num_threads(a.threads); torch.set_flush_denormal(True)
    if a.alpha and a.model == 'branch': p.error('--alpha applies to plain models only')
    arch = a.arch or ('1' if a.bench == 'mnist' else ('conv_b4_sparse' if a.model == 'branch' else 'cnn'))
    variant = f'lr{a.lr:g}_do{a.dropout}' if a.model == 'hat' else '_'.join(v for v in [a.gate_rule, f'gate{a.gate_loss:g}_{a.gate_mode}' if a.gate_loss else None, f'kwta{a.kwta:g}' if a.kwta else None, 'patch' if a.input_layout == 'patch' else None] if v)
    if a.gate_loss and a.model != 'branch': p.error('--gate-loss applies to branch models only')
    if a.model == 'hat': arch = 'hat_mlp' if a.bench == 'mnist' else 'hat_cnn'
    folder = {'hat': f'{a.bench}_hat'}.get(a.model, f'{a.bench}_gates' if a.gate_rule else a.bench)
    out = a.out or Path(f'results/runs/{folder}/{a.model}_{arch}_a{a.alpha:g}' + (f'_{variant}' if variant else '') + '.json'); out.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(out.read_text()) if out.exists() else []; done = {run_key(r) for r in runs}
    data = D.load(a.bench); key_tasks = pixel_key_tasks(data, a.bench) if a.key_routing else None
    for seed in a.seeds or DEFAULT_SEEDS[a.bench]:
        for first in a.orders:
            info = dict(bench=a.bench, model=a.model, arch=arch, alpha=a.alpha, variant=variant, seed=seed)
            if (a.model, arch, a.alpha, variant, seed, first) in done: continue
            order = D.task_order(first)
            if a.model == 'hat':
                model = build_hat(a.bench, seed, bool(a.dropout)); trainer = HATTrainer(model, BENCH_CONFIG[a.bench], seed, lr=a.lr); meta = None
            else:
                model = build_model(a.bench, a.model, seed, arch)
                if a.input_layout == 'patch':
                    first_layer = model.branch_layers()[0]; first_layer.indices = patch_indices(first_layer.n * first_layer.B, torch.Generator().manual_seed(seed + 1))
                if a.kwta:
                    for layer in model.branch_layers(): layer.k_frac = a.kwta
                cfg = dataclasses.replace(BENCH_CONFIG[a.bench], gate_weight=a.gate_loss, gate_mode=a.gate_mode)
                trainer = ContinualTrainer(model, cfg, seed, metaplasticity_alpha=a.alpha,
                                           gate_rule=GateRule(a.gate_rule, seed) if a.gate_rule else None); meta = trainer.meta
            run = trainer.fit(data, order)
            if a.gate_rule: GateRule.clear(model)
            rec = run_record(run, data, order, meta, **info)
            if getattr(model, 'is_branch', False):
                rec['route_mass'] = {split: model.route_mass(data.split(split)[0], [q[D.task_of(data.split(split)[1])] for q in run.routes]) for split in ['val', 'test']}
            if key_tasks is not None and a.model != 'plain':
                rec['key_routed'] = {split: key_routed(model, run, data, split, key_tasks[split]) for split in ['val', 'test']}
            runs = json.loads(out.read_text()) if out.exists() else []; runs.append(rec); out.write_text(json.dumps(runs, indent=1))
            if a.save_checkpoints:
                a.save_checkpoints.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state=model.state_dict(), routes=run.routes, info=info, order=order), a.save_checkpoints / (f'{a.model}_{arch}_' + (f'{variant}_' if variant else '') + f'{seed}_{first}.pt'))
            main_key = {'branch': 'soft', 'plain': 'plain', 'hat': 'route'}[a.model]
            print(a.bench, a.model, arch, a.alpha, variant, seed, first, f"CIL {100 * rec['test'][main_key]['cil']:.2f}",
                  f"TIL {100 * rec['test'][main_key]['til']:.2f}", f"key-routed {100 * rec['key_routed']['test']:.2f}" if 'key_routed' in rec else '', f"route-TIL {100 * rec['test']['route']['til']:.2f}" if 'route' in rec['test'] else '',
                  f"forgetting {100 * rec['forgetting_til_val']:.1f}", f"route mass {[round(v, 3) for v in rec['route_mass']['test']]}" if 'route_mass' in rec else '', f"{rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
