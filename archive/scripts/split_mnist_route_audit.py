"""Independent forward-pass audit of previously saved hard branch routes.

Each route is evaluated on the ENTIRE test set without label input. Ground-truth
pair identities select the diagonal only afterwards: explicitly an oracle audit.
"""
import argparse
import copy
import json
import statistics
from pathlib import Path

import torch
from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make


@torch.no_grad()
def fixed_route_predictions(model, x, winners=None):
    m = copy.deepcopy(model)
    if winners is not None and m.arm != 'uniform':
        m.error.fill_(1)
        m.error.scatter_(1, winners[:, None], 0)
    return torch.cat([m(b).argmax(-1) for b in x.split(512)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='results/split_mnist_usefulness_hard_20261011')
    p.add_argument('--diagnostic', default='results/split_mnist_retention_diagnostic_20261011')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = json.loads((source / 'results.json').read_text())
    prior = json.loads((Path(args.diagnostic) / 'results.json').read_text())
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1) / 255
    y = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    records = []
    for row in runs:
        if row['mode'] != 'sequential' or row['arm'] not in ('local', 'shared', 'uniform'):
            continue
        m = make(row['arm'], gate='hard')
        m.load_state_dict(torch.load(source / f'{row["seed"]}_{row["arm"]}_sequential.pt', weights_only=True))
        original = {k: v.clone() for k, v in m.state_dict().items()}
        routes = torch.tensor([r['preferred_branches'] for r in row['stages']])
        # No test targets or task labels are passed into any forward call here.
        predictions = [fixed_route_predictions(m, x, route) for route in routes]
        matrix, count, wrong_count = [], 0, 0
        for i, pred in enumerate(predictions):
            values = []
            for j, (a, b) in enumerate(SPLITS):
                ix = (y == a) | (y == b)
                values.append((pred[ix] == y[ix]).float().mean().item())
                if i == j:
                    count += int((pred[ix] == y[ix]).sum())
                    wrong_count += int((predictions[(i + 1) % 5][ix] == y[ix]).sum())
            matrix.append(values)
        assert all(torch.equal(v, m.state_dict()[k]) for k, v in original.items())
        if row['arm'] == 'local':
            expected = next(r for r in prior if r['arm'] == 'local' and r['seed'] == row['seed'])
            assert abs(count / len(y) - expected['saved_routes_original_readout_oracle']['all_accuracy']) < 1e-6
        record = {'seed': row['seed'], 'arm': row['arm'], 'original_accuracy': row['final_accuracy'],
                  'oracle_saved_route_accuracy': count / len(y), 'wrong_route_accuracy': wrong_count / len(y),
                  'route_by_test_pair_accuracy': matrix, 'routes': routes.tolist()}
        records.append(record)
        print(row['seed'], row['arm'], 'oracle', round(count / len(y) * 100, 2),
              'wrong route', round(wrong_count / len(y) * 100, 2), flush=True)
    (out / 'results.json').write_text(json.dumps(records, indent=2))
    (out / 'source.py').write_text(Path(__file__).read_text())
    (out / 'protocol.json').write_text(json.dumps({**vars(args),
        'oracle': 'True test pair used ONLY after complete ten-class forward passes to choose the corresponding saved route score. This is task-aware diagnosis, not a task-ID-free CL result.',
        'invariance': 'Original final network weights and classifier are loaded from saved checkpoints; only a copy of error-memory state is changed to select each old route. No classifier fitting, gradient update, training label or test label enters a forward pass.',
        'cross_check': 'Independent model.forward with explicit buffer restoration agrees exactly with the prior diagnostic feature.gather plus head calculation, for all three local seeds.',
        'uniform': 'Uniform averaging ignores route interventions and is evaluated unchanged.',
        'wrong': 'Cyclically assign the next task route to each pair, retaining unrestricted ten-digit outputs.'}, indent=2))
    lines = ['# Saved branch-route retrieval: independent oracle audit', '',
             '| Model | Ordinary no-ID accuracy | Correct old-route oracle | Wrong-route control |', '|---|---:|---:|---:|']
    for arm in ('uniform', 'shared', 'local'):
        rs = [r for r in records if r['arm'] == arm]
        vals = [statistics.mean(r[k] for r in rs) * 100 for k in
                ('original_accuracy', 'oracle_saved_route_accuracy', 'wrong_route_accuracy')]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{v:.2f}%' for v in vals) + ' |')
    lines += ['', 'Correct-route oracle has the true pair identity and a cache of earlier routes. It must never be presented as task-ID-free accuracy. The local result independently matches the earlier audit exactly. All predictions still compete across all ten digits; there is no pair mask and the final classifier is unchanged.', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
