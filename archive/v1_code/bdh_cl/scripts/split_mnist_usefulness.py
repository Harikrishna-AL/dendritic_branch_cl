"""Strict class-incremental Split MNIST: branch utility without test-label feedback.

The electricity rule is adapted to classification: scalar squared forecast error
becomes cross-entropy of counterfactual branch outputs. Images are iid samples,
so there are no artificial per-minibatch-lane memories or temporal activation
correlation gates. Every model emits ten logits, including for unseen classes.
"""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw, SPLITS


ARMS = ('mlp', 'uniform', 'shared', 'local')


class BranchClassifier(nn.Module):
    def __init__(self, arm, inputs=784, neurons=24, branches=4, gate='soft'):
        super().__init__()
        self.arm, self.neurons, self.branches = arm, neurons, branches
        self.gate = gate
        self.first = nn.Linear(inputs, neurons * branches)
        self.head = nn.Linear(neurons, 10)
        self.register_buffer('error', torch.full((neurons, branches), math.log(10)))

    def gates(self):
        if self.arm == 'uniform':
            return torch.full_like(self.error, 1 / self.branches)
        if self.gate == 'hard':
            return F.one_hot(self.error.argmin(-1), self.branches).to(self.error.dtype)
        return torch.softmax(-self.error / .2, dim=-1)

    def forward(self, x, counterfactual=False):
        branches = torch.tanh(self.first(x).reshape(-1, self.neurons, self.branches))
        # Snapshot the gate: subsequent EMA writes must not mutate a saved graph.
        gate = self.gates().detach().clone()
        hidden = (branches * gate).sum(-1)
        logits = self.head(hidden)
        if not counterfactual:
            return logits
        # Replace exactly one neuron's pooled value with each candidate branch,
        # holding the other neuron contributions fixed. Labels never enter this.
        delta = branches - hidden[:, :, None]
        alternatives = logits[:, None, None, :] + delta[:, :, :, None] * self.head.weight.T[None, :, None, :]
        return logits, alternatives

    @torch.no_grad()
    def observe(self, error):
        if self.arm == 'shared':
            error = error.mean(0, keepdim=True).expand_as(self.error)
        # Shared vector is replicated only to equalize allocated buffer bytes;
        # it has four distinct values rather than neurons*branches values.
        self.error.mul_(.95).add_(error, alpha=.05)


def make(arm, gate='soft'):
    if arm == 'mlp':
        return nn.Sequential(nn.Linear(784, 95), nn.Tanh(), nn.Linear(95, 10))
    return BranchClassifier(arm, gate=gate)


def train_epoch(model, opt, x, y, indices, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    order = indices[torch.randperm(len(indices), generator=generator)]
    for ids in order.split(batch_size):
        bx, by = x[ids], y[ids]
        opt.zero_grad(set_to_none=True)
        if isinstance(model, BranchClassifier):
            logits, candidates = model(bx, counterfactual=True)
            targets = by[:, None, None].expand(candidates.shape[:-1]).reshape(-1)
            errors = F.cross_entropy(candidates.reshape(-1, 10), targets, reduction='none').reshape(candidates.shape[:-1])
            loss = F.cross_entropy(logits, by) + .5 * errors.mean()
            factual_errors = errors.detach().mean(0)
        else:
            loss = F.cross_entropy(model(bx), by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if isinstance(model, BranchClassifier):
            model.observe(factual_errors)


@torch.no_grad()
def evaluate(model, x, y, seen_classes):
    # A complete state snapshot checks that evaluation performs no learning or
    # feedback writes. Ground truth is used solely for scoring predictions.
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pred = torch.cat([model(b).argmax(-1) for b in x.split(512)])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    pair_acc = []
    for pair in SPLITS:
        mask = (y == pair[0]) | (y == pair[1])
        pair_acc.append((pred[mask] == y[mask]).float().mean().item())
    seen = torch.zeros_like(y, dtype=torch.bool)
    for c in seen_classes:
        seen |= y == c
    return {'seen_accuracy': (pred[seen] == y[seen]).float().mean().item(),
            'all_ten_accuracy': (pred == y).float().mean().item(),
            'pair_accuracy': pair_acc}


def summarize(records, out):
    lines = ['# Split MNIST usefulness-memory pilot', '',
             'Five sequential digit pairs; all ten logits at every stage; no task IDs, replay, output masking, pretraining, or test-label feedback. Full 28×28 images and official 10,000-image test set. Memory and weights are frozen during every test.', '',
             '| Model | Parameters | Final all-digit accuracy | Acquisition accuracy | Old-task forgetting | Final old-pair accuracy | Final new-pair accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rs = [r for r in records if r['arm'] == arm and r['mode'] == 'sequential']
        if not rs:
            continue
        vals = [statistics.mean(r[key] for r in rs) * 100 for key in
                ('final_accuracy', 'acquisition_accuracy', 'forgetting', 'final_old_accuracy', 'final_new_accuracy')]
        sd = statistics.stdev(r['final_accuracy'] for r in rs) * 100 if len(rs) > 1 else 0
        lines.append(f'| {arm} | {rs[0]["parameters"]} | {vals[0]:.2f}% ± {sd:.2f} | {vals[1]:.2f}% | {vals[2]:.2f} pp | {vals[3]:.2f}% | {vals[4]:.2f}% |')
    lines += ['', 'Acquisition is accuracy on each pair immediately after learning it, using unrestricted ten-class predictions. Forgetting averages the best pre-final accuracy minus final accuracy over the first four pairs. Accuracy after every phase and the full pair matrix are in results.json.', '',
              '| Joint-exposure capacity control | All-digit accuracy |', '|---|---:|']
    for arm in ('mlp', 'local'):
        rs = [r for r in records if r['arm'] == arm and r['mode'] == 'joint']
        if rs:
            lines.append(f'| {arm} | {100 * statistics.mean(r["final_accuracy"] for r in rs):.2f}% |')
    lines += ['', 'The branch models share identical architecture and the same cross-entropy plus counterfactual auxiliary objective. Local scores are per neuron and branch; shared scores pool across neurons. All three allocate the same 96-value error buffer. The MLP has 75 fewer parameters but does not receive the auxiliary objective. This tests usefulness memory adapted to classification, not an unchanged electricity model or local inter-branch communication.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='31,32,33')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--joint-arms', default='mlp,local')
    p.add_argument('--gate', choices=('soft', 'hard'), default='soft')
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1) / 255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1) / 255
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    (out / 'source.py').write_text(Path(__file__).read_text())
    (out / 'protocol.json').write_text(json.dumps({
        **vars(args), 'pairs': SPLITS, 'train_images': len(x), 'test_images': len(tx),
        'inputs': 'Raw 28x28 pixels scaled by 255; no feature pretraining or all-class feature fitting.',
        'branch_network': '24 hidden neurons, 4 tanh dendritic branches each, pooled with error-derived weights, one shared 10-logit classifier; 75610 slow parameters.',
        'dense_network': '784->95 tanh->10; 75535 slow parameters, <0.1% parameter difference.',
        'cue': f'EMA .95 of branch counterfactual cross-entropy; {args.gate} gate (soft: softmax(-error/.2); hard: argmin one-hot); scores updated after a training prediction using training labels. Counterfactual replaces one neuron value at a time. No previous-image correlation assumption.',
        'matching': 'Local/shared/uniform have identical weights, objective, initialization, sample order, and allocated 96-float state. Shared pools score estimates across neurons and replicates its four unique values. Uniform ignores scores.',
        'loss': 'Branch current cross-entropy plus .5 mean counterfactual cross-entropy; all candidates participate in training. MLP uses ordinary cross-entropy.',
        'learning': 'One Adam .001 per run, clipping 5, two passes per digit pair, no optimizer or memory reset at task boundaries. No boundary or pair ID supplied to model; pairs determine only the data order.',
        'evaluation': 'Full official test set after each phase; ten output logits always, including unseen classes. No target feedback, input-memory updates, task IDs, or pair masks. Full model state verified unchanged. Secondary erasure audit replaces error state with uniform values on a copy.',
        'joint_control': 'Train the same total number of labeled examples from all ten classes mixed together for two epochs; a capacity/learning control, not a continual-learning result.',
        'limitations': 'Three seeds and fixed natural pair order; mini-batch training, not single-example online learning. Gate is shared across images and cannot infer image-specific contexts at test time. Test labels only score; no test-based tuning. No result implies biological locality or dendrite communication.'
    }, indent=2))
    records = []
    for seed in map(int, args.seeds.split(',')):
        for arm in args.arms.split(','):
            torch.manual_seed(seed)
            m = make(arm, args.gate)
            opt = torch.optim.Adam(m.parameters(), lr=.001)
            row = {'seed': seed, 'arm': arm, 'mode': 'sequential',
                   'parameters': sum(p.numel() for p in m.parameters()), 'stages': []}
            seen = []
            for stage, pair in enumerate(SPLITS):
                indices = torch.nonzero((y == pair[0]) | (y == pair[1]), as_tuple=True)[0]
                for epoch in range(args.epochs):
                    train_epoch(m, opt, x, y, indices, seed * 100000 + stage * 100 + epoch, args.batch_size)
                seen.extend(pair)
                score = evaluate(m, tx, ty, seen)
                if isinstance(m, BranchClassifier):
                    score['preferred_branches'] = m.gates().argmax(-1).tolist()
                    score['gate_entropy'] = float(-(m.gates() * m.gates().clamp_min(1e-12).log()).sum(-1).mean())
                row['stages'].append(score)
                print(seed, arm, 'phase', stage + 1, 'seen accuracy', round(score['seen_accuracy'] * 100, 2), flush=True)
            last = row['stages'][-1]['pair_accuracy']
            row.update(final_accuracy=row['stages'][-1]['all_ten_accuracy'],
                       acquisition_accuracy=statistics.mean(row['stages'][i]['pair_accuracy'][i] for i in range(5)),
                       forgetting=statistics.mean(max(row['stages'][s]['pair_accuracy'][i] for s in range(i, 4)) - last[i] for i in range(4)),
                       final_old_accuracy=statistics.mean(last[:4]), final_new_accuracy=last[4])
            if isinstance(m, BranchClassifier):
                erased = copy.deepcopy(m)
                erased.error.fill_(math.log(10))
                row['erased_memory_audit'] = evaluate(erased, tx, ty, list(range(10)))
            torch.save(m.state_dict(), out / f'{seed}_{arm}_sequential.pt')
            records.append(row)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            summarize(records, out)
        for arm in filter(None, args.joint_arms.split(',')):
            torch.manual_seed(seed)
            m = make(arm, args.gate)
            opt = torch.optim.Adam(m.parameters(), lr=.001)
            indices = torch.arange(len(y))
            for epoch in range(args.epochs):
                train_epoch(m, opt, x, y, indices, seed * 100000 + 900 + epoch, args.batch_size)
            score = evaluate(m, tx, ty, list(range(10)))
            row = {'seed': seed, 'arm': arm, 'mode': 'joint',
                   'parameters': sum(p.numel() for p in m.parameters()), 'final_accuracy': score['all_ten_accuracy'], 'evaluation': score}
            records.append(row)
            print(seed, arm, 'joint accuracy', round(row['final_accuracy'] * 100, 2), flush=True)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            summarize(records, out)


if __name__ == '__main__':
    main()
