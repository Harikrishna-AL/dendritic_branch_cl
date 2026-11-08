"""Oracle diagnostics of routes, readout and frozen representations after CL.

Fresh classifiers use ALL historical training labels. Saved-route and pair-mask
audits additionally use task identity. None are task-ID-free CL improvements.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier, make, train_epoch


@torch.no_grad()
def features(model, x, mode='current', routes=None, task_ids=None):
    chunks = []
    for start in range(0, len(x), 1024):
        bx = x[start:start + 1024]
        if not isinstance(model, BranchClassifier):
            chunks.append(model[1](model[0](bx)))
            continue
        q = torch.tanh(model.first(bx).reshape(-1, model.neurons, model.branches))
        if mode == 'all_branches':
            z = q.flatten(1)
        elif mode == 'saved_routes':
            if routes is None or task_ids is None:
                raise ValueError('saved route audit requires explicit oracle task IDs')
            winners = routes[task_ids[start:start + len(bx)]]
            z = q.gather(2, winners[:, :, None]).squeeze(-1)
        else:
            z = (q * model.gates()).sum(-1)
        chunks.append(z)
    return torch.cat(chunks)


def ridge_fit(z, y):
    """Fixed ridge=1, standardized training features, unpenalized intercept."""
    z = z.double()
    mean, std = z.mean(0), z.std(0).clamp_min(1e-6)
    design = torch.cat(((z - mean) / std, torch.ones(len(z), 1)), 1)
    target = F.one_hot(y, 10).double()
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    return mean, std, weight


def ridge_logits(z, fitted):
    mean, std, weight = fitted
    design = torch.cat(((z.double() - mean) / std, torch.ones(len(z), 1)), 1)
    return design @ weight


def score(logits, y):
    pred = logits.argmax(-1)
    pairs = []
    for a, b in SPLITS:
        ix = (y == a) | (y == b)
        pairs.append((pred[ix] == y[ix]).float().mean().item())
    old = y < 8
    return {'all_accuracy': (pred == y).float().mean().item(),
            'old_accuracy': (pred[old] == y[old]).float().mean().item(),
            'pair_accuracy': pairs}


def pair_mask_score(logits, y):
    """Task-aware diagnostic ONLY: restrict output to the known true pair."""
    pred = torch.empty_like(y)
    for a, b in SPLITS:
        ix = (y == a) | (y == b)
        pair = torch.tensor([a, b])
        pred[ix] = pair[logits[ix][:, pair].argmax(-1)]
    # Produce scores with identical schema without exposing this as CL accuracy.
    return score(F.one_hot(pred, 10).float(), y)


@torch.no_grad()
def head_logits(model, z):
    return model.head(z) if isinstance(model, BranchClassifier) else model[2](z)


def probe(model, x, y, tx, ty, mode='current', routes=None):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tr = features(model, x, mode, routes, y // 2 if mode == 'saved_routes' else None)
    te = features(model, tx, mode, routes, ty // 2 if mode == 'saved_routes' else None)
    fitted = ridge_fit(tr, y)
    result = score(ridge_logits(te, fitted), ty)
    result['feature_dimensions'] = tr.shape[1]
    result['train_accuracy'] = score(ridge_logits(tr, fitted), y)['all_accuracy']
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='results/split_mnist_usefulness_hard_20261011')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = json.loads((source / 'results.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    seeds = sorted({r['seed'] for r in runs})
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1) / 255
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1) / 255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    (out / 'source.py').write_text(Path(__file__).read_text())
    (out / 'protocol.json').write_text(json.dumps({**vars(args), 'seeds': seeds,
        'frozen': 'Original encoder, classifier weights and utility buffers remain unchanged in all extraction audits. Fresh ridge decoder is fit separately from scratch.',
        'probe': 'Fit ten-class linear ridge decoder using all 60000 historical training labels and frozen final features; evaluate the untouched official 10000 test images. Standardize with training feature moments; ridge 1 fixed, no test tuning.',
        'oracle': 'Saved-route audit uses true digit-pair identity to choose the route saved after that training phase. Pair-mask audit uses true pair identity to restrict outputs. These are explicitly privileged diagnostics, not practical CL scores.',
        'controls': 'Random initialized features; rerun original joint exposure (two epochs, all classes mixed) with identical seed/order; original classifier under current and oracle saved routes; current 24-d pooled features vs all 96 branch activations.',
        'limits': 'Decodability after full-history labeled readout fitting does not establish retained learned skills. A high probe score may come from generic or random input features. Joint control makes distinguishability comparisons more informative; saved route can fail because encoder or classifier weights also changed.'}, indent=2))
    records = []
    for seed in seeds:
        for arm in ('mlp', 'local'):
            torch.manual_seed(seed)
            model = make(arm, gate='hard')
            random_model = make(arm, gate='hard')
            random_model.load_state_dict(model.state_dict())
            model.load_state_dict(torch.load(source / f'{seed}_{arm}_sequential.pt', weights_only=True))
            route_row = next(r for r in runs if r['seed'] == seed and r['arm'] == arm and r['mode'] == 'sequential')
            routes = torch.tensor([s['preferred_branches'] for s in route_row['stages']]) if arm == 'local' else None
            z = features(model, tx)
            logits = head_logits(model, z)
            row = {'seed': seed, 'arm': arm,
                   'original': score(logits, ty), 'pair_mask_oracle': pair_mask_score(logits, ty),
                   'final_features_fresh_readout': probe(model, x, y, tx, ty),
                   'random_features_fresh_readout': probe(random_model, x, y, tx, ty)}
            if arm == 'local':
                oldroute_z = features(model, tx, 'saved_routes', routes, ty // 2)
                row['saved_routes_original_readout_oracle'] = score(head_logits(model, oldroute_z), ty)
                row['saved_routes_fresh_readout_oracle'] = probe(model, x, y, tx, ty, 'saved_routes', routes)
                row['all_branches_fresh_readout'] = probe(model, x, y, tx, ty, 'all_branches')
                row['random_all_branches_fresh_readout'] = probe(random_model, x, y, tx, ty, 'all_branches')
            # Reproduce joint training; these weights were not saved by the pilot.
            torch.manual_seed(seed)
            joint = make(arm, gate='hard')
            opt = torch.optim.Adam(joint.parameters(), lr=.001)
            for epoch in range(protocol['epochs']):
                train_epoch(joint, opt, x, y, torch.arange(len(y)), seed * 100000 + 900 + epoch, protocol['batch_size'])
            joint_logits = head_logits(joint, features(joint, tx))
            row['joint_original'] = score(joint_logits, ty)
            prior_joint = next(r for r in runs if r['seed'] == seed and r['arm'] == arm and r['mode'] == 'joint')
            assert abs(row['joint_original']['all_accuracy'] - prior_joint['final_accuracy']) < 1e-6, 'joint reproduction changed'
            row['joint_features_fresh_readout'] = probe(joint, x, y, tx, ty)
            if arm == 'local':
                row['joint_all_branches_fresh_readout'] = probe(joint, x, y, tx, ty, 'all_branches')
            records.append(row)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            print(seed, arm, 'original', round(row['original']['all_accuracy'] * 100, 2),
                  'fresh head', round(row['final_features_fresh_readout']['all_accuracy'] * 100, 2),
                  'random', round(row['random_features_fresh_readout']['all_accuracy'] * 100, 2),
                  'joint probe', round(row['joint_features_fresh_readout']['all_accuracy'] * 100, 2), flush=True)
    lines = ['# Where did Split MNIST retention fail?', '',
             'Three original hard-competition seeds. All fresh-readout results use every historical training label; oracle routes and pair masks additionally use task identity. None improve the strict class-incremental score.', '',
             '| Diagnostic | MLP: all digits / older digits | Branch model: all digits / older digits |', '|---|---:|---:|']
    keys = [('original', 'Original final model'),
            ('final_features_fresh_readout', 'Fresh readout, frozen current features'),
            ('random_features_fresh_readout', 'Fresh readout, random features'),
            ('joint_features_fresh_readout', 'Fresh readout, jointly trained features'),
            ('pair_mask_oracle', 'Oracle restricts outputs to true pair'),
            ('saved_routes_original_readout_oracle', 'Oracle saved routes, original readout'),
            ('saved_routes_fresh_readout_oracle', 'Oracle saved routes, fresh readout'),
            ('all_branches_fresh_readout', 'All 96 branch features, fresh readout'),
            ('random_all_branches_fresh_readout', 'Random 96 branch features, fresh readout'),
            ('joint_all_branches_fresh_readout', 'Joint 96 branch features, fresh readout')]
    for key, label in keys:
        cells = []
        for arm in ('mlp', 'local'):
            rs = [r for r in records if r['arm'] == arm and key in r]
            if not rs:
                cells.append('—')
            else:
                cells.append(f'{100 * statistics.mean(r[key]["all_accuracy"] for r in rs):.2f}% / {100 * statistics.mean(r[key]["old_accuracy"] for r in rs):.2f}%')
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Current branch features have 24 dimensions, all branches 96, and MLP features 95. Expanding the feature vector changes decoder capacity and is only a diagnostic. Original model state was checked unchanged during extraction and all probe fitting; joint training reproduced the earlier recorded test accuracy exactly.', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
