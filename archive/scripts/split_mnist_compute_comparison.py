"""Matched-parameter accuracy and local CPU compute audit of the existing method.

Run with python -m bdh_cl.scripts.split_mnist_compute_comparison --out DIR.
Uses the original model, objective, data order, optimizer and evaluation code.
"""
import argparse
import copy
import hashlib
import json
import platform
import statistics as stats
import sys
import time
from pathlib import Path

import torch

from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts import split_mnist_usefulness as original
from bdh_cl.scripts.split_mnist_route_audit import fixed_route_predictions


@torch.no_grad()
def inference_benchmark(models, x):
    """Interleave models; report seven warmed block means per batch size."""
    result = {}
    for size, repeats in ((1, 500), (128, 100), (512, 50)):
        bx = x[:size]
        for model in models.values():
            for _ in range(20):
                model(bx)
        timings = {arm: [] for arm in models}
        for block in range(7):
            order = list(models) if block % 2 == 0 else list(reversed(models))
            for arm in order:
                start = time.perf_counter()
                for _ in range(repeats):
                    models[arm](bx)
                timings[arm].append((time.perf_counter() - start) * 1e6 / repeats)
        result[str(size)] = {arm: {'block_us': values, 'median_us': stats.median(values)}
                             for arm, values in timings.items()}
    return result


def report(out, records, protocol):
    lines = ['# Parameter-matched Split MNIST: accuracy and compute', '',
             'Three repeat seeds of the existing experiment; these are reproducibility/timing runs, '
             'not new independent confirmation seeds. All five digit pairs, two epochs per pair, '
             '120,000 training presentations, raw pixels, no replay or pretraining. '
             'No task ID, output mask or label feedback at ordinary inference.', '',
             'Both models use Adam 0.001, batch 128, gradient clipping 5, and the same per-seed '
             'sample order. Model initialization differs with architecture. Branch training retains '
             'its original counterfactual auxiliary loss; the MLP uses ordinary cross-entropy. '
             'This compares complete methods, not architecture alone.', '',
             '| Model | Parameters | Final accuracy | Acquisition | Old-task forgetting | Training wall seconds | Training CPU seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('mlp', 'local'):
        group = [r for r in records if r['arm'] == arm]
        if not group:
            continue
        def cell(key, scale=1):
            vals = [r[key] * scale for r in group]
            return f'{stats.mean(vals):.2f} ± {stats.stdev(vals) if len(vals)>1 else 0:.2f}'
        lines.append(f'| {arm} | {group[0]["parameters"]:,} | {cell("final_accuracy",100)}% | '
                     f'{cell("acquisition",100)}% | {cell("forgetting",100)} pp | '
                     f'{cell("train_wall_s")} | {cell("train_cpu_s")} |')
    lines += ['', 'Values are mean ± sample standard deviation across seeds. Training timers include '
              'shuffling, batch indexing, forward/backward, clipping, Adam, and memory writes; exclude '
              'dataset loading, initial warmup, evaluation, checkpoint saving, and oracle diagnostics.', '',
              '| Inference batch | MLP μs/batch | Branch μs/batch | Branch/MLP |',
              '|---|---:|---:|---:|']
    for size in ('1', '128', '512'):
        vals = {}
        for arm in ('mlp', 'local'):
            vals[arm] = [r['inference'][size][arm]['median_us'] for r in records
                         if r['arm'] == arm and 'inference' in r]
        if all(vals.values()):
            m, b = (stats.mean(vals[a]) for a in ('mlp', 'local'))
            lines.append(f'| {size} | {m:.2f} | {b:.2f} | {b/m:.2f}× |')
    paired = []
    for seed in sorted({r['seed'] for r in records}):
        runs = {r['arm']: r for r in records if r['seed'] == seed}
        if len(runs) == 2:
            paired.append(runs['local']['train_wall_s'] / runs['mlp']['train_wall_s'])
    if paired:
        lines += ['', f'Mean paired branch/MLP training wall-time ratio: **{stats.mean(paired):.2f}×**.']
    oracle = [r['oracle_accuracy'] for r in records if 'oracle_accuracy' in r]
    if oracle:
        lines += ['', f'Historical-route oracle accuracy: **{100*stats.mean(oracle):.2f}%**. '
                  'This uses externally saved configurations and true test digit-pair identities with '
                  'the final weights/classifier. It is a separate diagnostic, not a fair ordinary '
                  'accuracy comparison with the MLP. Oracle diagnostic cost is excluded from ordinary '
                  'inference latency. Low MLP accuracy does not establish that its representations '
                  'contain no earlier information.']
    lines += ['', '## Compute and storage interpretation', '',
              '- Trainable weights: branch 75,610 versus MLP 75,535 (0.0993% difference). '
              'FP32 weight storage: 302,440 versus 302,140 bytes.',
              '- Branch running-error memory: 96 FP32 values = 384 bytes. Five saved configurations '
              'contain 120 branch indices (120 bytes if encoded as uint8); this external diagnostic '
              'archive is separate from the running memory. Optimizer state and activations are extra.',
              '- Dense forward matrix products per image: MLP 75,430 multiply-accumulates; '
              'branch 75,504. These counts exclude biases, tanh, routing and other elementwise work; '
              'they are not complete FLOP counts or training FLOP estimates.',
              '- Branch training additionally constructs 96 alternative ten-class predictions per '
              'image and differentiates their mean loss. At batch 128, that logits tensor alone is '
              '491,520 bytes, versus 5,120 bytes for the factual logits. These are tensor sizes, '
              'not measured peak memory. The selected candidate is included among the 96 alternatives.',
              '- Inference uses the existing unoptimized implementation: all 96 branch responses are '
              'computed, then 24 selected. No counterfactual predictions or memory writes at inference.',
              '', '## Timing scope', '',
              f'Environment: {protocol["environment"]}. CPU float32, two PyTorch threads. '
              'Seven warmed timing blocks per inference batch size, interleaved model order; '
              'table averages within-seed medians. Training model order alternates across seeds. '
              'These are local runtime measurements, not energy measurements or GPU/scaling claims. '
              'No hyperparameter tuning, equal-wall-time learning comparison, or peak-memory profiling '
              'was performed. Existing joint-training capacity results are in '
              '`results/split_mnist_usefulness_hard_20261011/summary.md`.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'seeds': [34, 35, 36], 'epochs': 2, 'batch_size': 128,
                'environment': f'{platform.platform()}, torch {torch.__version__}, Python {sys.version.split()[0]}',
                'command': sys.argv, 'device': 'cpu', 'threads': 2, 'dtype': 'float32'}
    for name, path in [('benchmark', Path(__file__)), ('model', Path(original.__file__)),
                       ('route_audit', Path(sys.modules[fixed_route_predictions.__module__].__file__))]:
        source = path.read_bytes()
        (out / f'{name}_source.py').write_bytes(source)
        protocol[f'{name}_sha256'] = hashlib.sha256(source).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1) / 255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1) / 255
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    old = json.loads(Path('results/split_mnist_usefulness_hard_20261011/results.json').read_text())
    records = []
    for seed_i, seed in enumerate(protocol['seeds']):
        models = {}
        for arm in (('mlp', 'local') if seed_i % 2 == 0 else ('local', 'mlp')):
            # Warm kernels on a disposable model; reset the random seed afterwards.
            warm = original.make(arm, 'hard')
            opt = torch.optim.Adam(warm.parameters(), lr=.001)
            original.train_epoch(warm, opt, x, y, torch.arange(640), 0, 128)
            del warm, opt
            torch.manual_seed(seed)
            model = original.make(arm, 'hard')
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            row = {'seed': seed, 'arm': arm, 'parameters': sum(p.numel() for p in model.parameters()),
                   'train_wall_s': 0., 'train_cpu_s': 0., 'stages': []}
            routes, seen = [], []
            for stage, pair in enumerate(SPLITS):
                indices = torch.nonzero((y == pair[0]) | (y == pair[1]), as_tuple=True)[0]
                wall, cpu = time.perf_counter(), time.process_time()
                for epoch in range(2):
                    original.train_epoch(model, opt, x, y, indices, seed*100000+stage*100+epoch, 128)
                row['train_cpu_s'] += time.process_time() - cpu
                row['train_wall_s'] += time.perf_counter() - wall
                seen.extend(pair)
                score = original.evaluate(model, tx, ty, seen)
                row['stages'].append(score)
                if arm == 'local':
                    routes.append(model.gates().argmax(-1).clone())
                print(seed, arm, 'stage', stage+1, 'accuracy', round(score['seen_accuracy']*100,2),
                      'cumulative train s', round(row['train_wall_s'],2), flush=True)
            final = row['stages'][-1]['pair_accuracy']
            row['final_accuracy'] = row['stages'][-1]['all_ten_accuracy']
            row['acquisition'] = stats.mean(row['stages'][i]['pair_accuracy'][i] for i in range(5))
            row['forgetting'] = stats.mean(max(row['stages'][s]['pair_accuracy'][i] for s in range(i,4))-final[i]
                                          for i in range(4))
            reference = next(r for r in old if r['seed']==seed and r['arm']==arm and r['mode']=='sequential')
            row['max_accuracy_difference_from_original'] = max(abs(a['all_ten_accuracy']-b['all_ten_accuracy'])
                                                               for a,b in zip(row['stages'],reference['stages']))
            if arm == 'local':
                correct = 0
                for pair, route in zip(SPLITS, routes):
                    pred = fixed_route_predictions(model, tx, route)
                    mask = (ty==pair[0]) | (ty==pair[1])
                    correct += int((pred[mask]==ty[mask]).sum())
                row['oracle_accuracy'] = correct/len(ty)
                row['routes'] = [r.tolist() for r in routes]
            torch.save(model.state_dict(), out / f'{seed}_{arm}.pt')
            records.append(row)
            models[arm] = model
            (out / 'results.json').write_text(json.dumps(records, indent=2))
        before = {a: copy.deepcopy(m.state_dict()) for a,m in models.items()}
        latency = inference_benchmark(models, tx)
        for arm, model in models.items():
            assert all(torch.equal(v, model.state_dict()[k]) for k,v in before[arm].items())
        for row in records:
            if row['seed'] == seed:
                row['inference'] = latency
        (out / 'results.json').write_text(json.dumps(records, indent=2))
        report(out, records, protocol)
    print((out / 'REPORT.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
