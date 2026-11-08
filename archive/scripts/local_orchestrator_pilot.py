"""Small paired factorial for layer-local dendritic orchestration."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.dendritic_cl_pilot import data_for_seed, ROOT
from bdh_cl.data.mnist import build_pools

ARMS = ('context', 'local', 'winner', 'fixed')
PRE = ('scratch', 'autoencoder', 'extra_supervised')

class Layer(nn.Module):
    def __init__(self, inputs, neurons=32, branches=4, context=8):
        super().__init__()
        self.neurons, self.branches = neurons, branches
        self.input = nn.Linear(inputs, neurons*branches)
        self.context = nn.Linear(inputs, context)
        self.keys = nn.Parameter(torch.randn(neurons*branches, context)*.2)

    def forward(self, x, arm, steps, gain=1.0):
        a = F.relu(self.input(x))
        keys = F.normalize(self.keys, dim=-1)
        c = torch.tanh(self.context(x))
        score = c @ keys.T
        message = a.unsqueeze(-1)*keys.unsqueeze(0)
        winner = a.argmax(-1)
        if arm in ('winner', 'fixed'):
            index = winner if arm == 'winner' else torch.zeros_like(winner)
            mask = F.one_hot(index, a.shape[-1]).unsqueeze(-1)
            message = message*mask
        if arm != 'context':
            # Each synchronous step accesses only self and ring neighbors.
            for _ in range(steps):
                message = (message + message.roll(1, 1) + message.roll(-1, 1))/3
            score = score + gain*(message*keys.unsqueeze(0)).sum(-1)
        gate = torch.sigmoid(score)
        h = (a*gate).reshape(len(x), self.neurons, self.branches).mean(-1)
        return h, gate, winner

class Network(nn.Module):
    def __init__(self, arm='context', steps=4, gain=1.0):
        super().__init__()
        self.arm, self.steps, self.gain = arm, steps, gain
        self.layers = nn.ModuleList([Layer(196), Layer(32)])
        self.head = nn.Linear(32, 10)
        self.decoder = nn.Linear(32, 196)

    def encode(self, x):
        diagnostics = []
        for layer in self.layers:
            x, g, w = layer(x, self.arm, self.steps, self.gain)
            diagnostics.append((g, w))
        return x, diagnostics

    def forward(self, x):
        return self.head(self.encode(x)[0])

def fit(model, x, y, epochs, seed, objective='classification'):
    parameters = list(model.layers.parameters()) + list(
        model.decoder.parameters() if objective == 'reconstruction' else model.head.parameters())
    opt = torch.optim.Adam(parameters, lr=.003)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ix in torch.randperm(len(x), generator=gen).split(64):
            opt.zero_grad(set_to_none=True)
            z, _ = model.encode(x[ix])
            loss = (F.mse_loss(torch.sigmoid(model.decoder(z)), x[ix])
                    if objective == 'reconstruction' else F.cross_entropy(model.head(z), y[ix]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5)
            opt.step()

@torch.no_grad()
def evaluate(model, data):
    accuracy, routing = [], []
    for _, _, x, y in data:
        z, diagnostics = model.encode(x)
        accuracy.append(float((model.head(z).argmax(-1)==y).float().mean()))
        routing.append([{'mean_gate': float(g.mean()),
                         'winner_max_share': float(torch.bincount(w, minlength=g.shape[1]).max()/len(w)),
                         'gate_profile': g.mean(0).tolist()} for g,w in diagnostics])
    return accuracy, routing

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--pretrain-epochs', type=int, default=4)
    p.add_argument('--ntrain', type=int, default=512)
    p.add_argument('--ntest', type=int, default=200)
    p.add_argument('--steps', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    pools = build_pools(ROOT/'data_cache/mnist', side=14, task_tags=False)
    protocol = {**vars(args), 'arms': ARMS, 'pretraining': PRE,
        'architecture': 'two layers, 32 neurons x 4 branches per layer, shared ten-class head',
        'context': 'learned projection of each layer input; no supplied task identity',
        'winner': 'largest raw branch activation per example per layer; changes with input',
        'locality': '128 branches in a fixed ring; four synchronous nearest-neighbor averaging steps; signed key-vector messages modulate sigmoid gates',
        'pretrain_data': 'disjoint first-task (digits 0/1) training examples only, same count as one downstream task; no future-task or test exposure',
        'budget_control': 'extra_supervised uses identical pretraining examples and optimizer steps with labels; downstream schedule identical',
        'evaluation': 'global ten-class prediction after every task; no replay, consolidation, or task identity; soft gates modulate gradients but do not protect shared parameters',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_version': torch.__version__}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    start = time.time()
    results = []
    for seed in map(int, args.seeds.split(',')):
        raw, indices = data_for_seed(pools, seed, args.ntrain*2, args.ntest)
        data = [(x[:args.ntrain], y[:args.ntrain], tx, ty) for x,y,tx,ty in raw]
        px, py = raw[0][0][args.ntrain:], raw[0][1][args.ntrain:]
        (out/f'indices_seed{seed}.json').write_text(json.dumps(indices))
        torch.manual_seed(seed)
        initial = Network(steps=args.steps)
        for pre in PRE:
            for arm in ARMS:
                model = copy.deepcopy(initial)
                model.arm = arm
                if pre != 'scratch':
                    fit(model, px, py, args.pretrain_epochs, seed+100,
                        'reconstruction' if pre == 'autoencoder' else 'classification')
                stages = []
                for task, (x,y,_,_) in enumerate(data):
                    fit(model, x, y, args.epochs, seed+200+task)
                    acc, routing = evaluate(model, data)
                    stages.append({'accuracy': acc, 'routing': routing})
                matrix = torch.tensor([s['accuracy'] for s in stages])
                forgetting = torch.stack([matrix[t:,t].max()-matrix[-1,t] for t in range(len(data)-1)]).mean().item()
                row = {'seed': seed, 'pretrain': pre, 'arm': arm, 'stages': stages,
                       'final_accuracy': matrix[-1].mean().item(),
                       'learning_accuracy': matrix.diag().mean().item(), 'forgetting': forgetting}
                results.append(row)
                (out/'results.json').write_text(json.dumps(results, indent=2))
                print(f'{seed} {pre:16} {arm:8} final={row["final_accuracy"]:.3f} learn={row["learning_accuracy"]:.3f} forgetting={forgetting:.3f}', flush=True)
    lines = ['# Local orchestrator pilot', '', 'Mean ± sample standard deviation across seeds. Accuracy uses the global ten-class head.', '',
             '| Pre-training | Mechanism | Final accuracy | Accuracy when learned | Forgetting |',
             '|---|---|---:|---:|---:|']
    for pre in PRE:
        for arm in ARMS:
            rows = [r for r in results if r['pretrain']==pre and r['arm']==arm]
            def cell(key):
                t = torch.tensor([r[key]*100 for r in rows])
                return f'{t.mean():.1f} ± {t.std().item() if len(t)>1 else 0:.1f}'
            lines.append(f'| {pre} | {arm} | {cell("final_accuracy")} | {cell("learning_accuracy")} | {cell("forgetting")} |')
    lines += ['', f'Elapsed: {time.time()-start:.1f} seconds.', '',
              'Pilot limitations: fixed ring topology; per-example rather than persistent task-level winner; soft gating; first-task-only reconstruction pre-training; three seeds are exploratory. Scratch uses fewer optimizer steps. Extra-supervised controls steps/data but also uses labels.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
