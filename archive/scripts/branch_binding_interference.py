"""Bounded synthetic feature-binding/interference experiment; no route memories.

Local regulation is an engineered activity-dependent Adam step multiplier, not
a biological learning-rule claim. The grouped-dense equivalence is intentional.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path('results/branch_binding_interference_20261025')
LR = [.001, .003, .01]
STRENGTH = [.1, 1.]
EPOCHS, BATCH, NTRAIN, NEVAL = 12, 128, 1024, 1024

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

class Model(nn.Module):
    def __init__(self, seed, architecture, tracking=False):
        super().__init__()
        torch.manual_seed(seed)
        self.architecture = architecture
        self.width = 59 if architecture == 'mlp' else 64
        self.input = nn.Linear(16, self.width)
        self.head = nn.Linear(59 if architecture == 'mlp' else 16, 2)
        self.register_buffer('exposure', torch.zeros(self.width if tracking else 0))
        nn.init.constant_(self.input.bias, -.1)

    def forward(self, x, return_activity=False):
        z = self.input(x)
        a = z if self.architecture == 'linear_branches' else F.relu(z)
        h = a if self.architecture == 'mlp' else a.reshape(-1, 16, 4).sum(-1) / 2
        out = self.head(h)
        return (out, a) if return_activity else out

    def flat_equivalent(self, x):
        # Ordinary hidden ReLUs with tied readout columns; no dendrite object.
        a = F.relu(F.linear(x, self.input.weight, self.input.bias))
        w = self.head.weight.repeat_interleave(4, dim=1) / 2
        return F.linear(a, w, self.head.bias)

def data(seed):
    """Four phases: disjoint pairs of colour-shape offsets, balanced marginals.

    Label is a single fixed function of the conjunction throughout the stream.
    Evaluation differs from training only in independently sampled observation
    noise; unseen conjunction generalization is NOT measured by this task.
    """
    rng = np.random.default_rng(seed)
    p = rng.permutation(8)
    negative, positive = p[:4], p[4:]
    label = np.zeros(8, np.int64)
    label[positive] = 1
    phases = []
    for j in range(4):
        combos = [(s, (s + int(off)) % 8, int(label[off]))
                  for off in [negative[j], positive[j]] for s in range(8)]
        blocks = []
        for n in [NTRAIN, NEVAL]:
            ids = np.tile(np.arange(16), n // 16)
            rng.shuffle(ids)
            c = np.asarray(combos)[ids]
            x = np.concatenate([np.eye(8)[c[:, 0]], np.eye(8)[c[:, 1]]], axis=1)
            x += rng.normal(0, .05, x.shape)
            blocks.append((torch.tensor(x, dtype=torch.float32), torch.tensor(c[:, 2])))
        phases.append(blocks)
    return phases

@torch.no_grad()
def evaluate(m, blocks):
    result = []
    for x, y in blocks:
        logits = m(x)
        result.append({'accuracy': float((logits.argmax(-1) == y).float().mean()),
                       'loss': float(F.cross_entropy(logits, y))})
    return result

def train(seed, architecture, regulation, lr, strength, reverse=False, joint=False):
    phases = data(seed)
    if reverse:
        phases = phases[::-1]
    m = Model(seed, architecture, tracking=regulation != 'none')
    opt = torch.optim.Adam(m.parameters(), lr=lr, foreach=False)
    generator = torch.Generator().manual_seed(seed + 10000)
    shuffle_generator = torch.Generator().manual_seed(seed + 20000)
    trajectory, damage, step_count = [], [], 0
    eval_blocks = [p[1] for p in phases]
    training_blocks = [p[0] for p in phases]
    if joint:
        training_blocks = [(torch.cat([p[0][0] for p in phases]),
                            torch.cat([p[0][1] for p in phases]))]
    started = time.perf_counter()
    for phase, (x, y) in enumerate(training_blocks):
        before = evaluate(m, eval_blocks)
        for epoch in range(EPOCHS):
            for ix in torch.randperm(len(y), generator=generator).split(BATCH):
                opt.zero_grad(set_to_none=True)
                logits, activity = m(x[ix], True)
                loss = F.cross_entropy(logits, y[ix])
                loss.backward()
                # A causal state: accumulated past mean-squared local response.
                # Multipliers apply AFTER Adam; scaling gradients alone can be
                # cancelled by Adam normalization. Shared uses same mean scale.
                if regulation != 'none':
                    scale = 1 / (1 + strength * m.exposure)
                    if regulation == 'shared':
                        scale = scale.mean().expand_as(scale)
                    elif regulation == 'shuffled':
                        scale = scale[torch.randperm(len(scale), generator=shuffle_generator)]
                    old_w = m.input.weight.detach().clone()
                    old_b = m.input.bias.detach().clone()
                opt.step()
                with torch.no_grad():
                    if regulation != 'none':
                        m.input.weight.copy_(old_w + scale[:, None] * (m.input.weight - old_w))
                        m.input.bias.copy_(old_b + scale * (m.input.bias - old_b))
                        m.exposure.add_(activity.detach().square().mean(0))
                step_count += 1
                if phase > 0 and epoch == 0 and len(damage) < phase:
                    # First current-task update: held-out probes, never feedback.
                    after = evaluate(m, eval_blocks)
                    damage.append({'phase': phase,
                        'old_loss_rise': st.mean(after[i]['loss'] - before[i]['loss'] for i in range(phase)),
                        'new_loss_drop': before[phase]['loss'] - after[phase]['loss']})
        trajectory.append(evaluate(m, eval_blocks))
    elapsed = time.perf_counter() - started
    diag = [trajectory[i][i]['accuracy'] for i in range(4)] if not joint else [p['accuracy'] for p in trajectory[-1]]
    final = [p['accuracy'] for p in trajectory[-1]]
    forgetting = st.mean(max(trajectory[t][i]['accuracy'] for t in range(i, 4)) - final[i]
                         for i in range(3)) if not joint else 0.
    parameters = sum(p.numel() for p in m.parameters())
    auxiliary = 0 if regulation == 'none' else m.exposure.numel() * 4
    result = {'seed': seed, 'order': 'reverse' if reverse else 'forward',
        'architecture': architecture, 'regulation': regulation,
        'lr': lr, 'strength': strength, 'joint': joint, 'variant': 'selected',
        'accuracy': st.mean(final), 'old_accuracy': st.mean(final[:3]),
        'new_accuracy': final[-1], 'acquisition': st.mean(diag),
        'forgetting': forgetting, 'trajectory': trajectory,
        'first_update_probes': damage, 'steps': step_count, 'seconds': elapsed,
        'parameters': parameters, 'auxiliary_bytes': auxiliary,
        'inference_bytes': parameters * 4,
        'training_persistent_bytes': parameters * 4 + auxiliary + sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)),
        'scale_min': float((1/(1+strength*m.exposure)).min()) if regulation != 'none' else 1.,
        'scale_mean': float((1/(1+strength*m.exposure)).mean()) if regulation != 'none' else 1.}
    return result, m

def verify():
    phases = data(700)
    for p in phases:
        x, y = p[0]
        assert float(y.float().mean()) == .5
        ids = x.reshape(-1, 2, 8).argmax(-1)
        for factor in range(2):
            assert torch.equal(torch.bincount(ids[:, factor]), torch.full((8,), NTRAIN//8))
    m = Model(700, 'branches')
    x = phases[0][0][0][:128]
    assert torch.allclose(m(x), m.flat_equivalent(x), atol=1e-6)
    m.zero_grad()
    F.cross_entropy(m(x), phases[0][0][1][:128]).backward()
    grad = {k: p.grad.clone() for k, p in m.named_parameters()}
    m.zero_grad()
    F.cross_entropy(m.flat_equivalent(x), phases[0][0][1][:128]).backward()
    assert all(torch.allclose(p.grad, grad[k], atol=1e-6) for k,p in m.named_parameters())
    return {'balanced_marginals': True, 'balanced_labels': True,
            'ordinary_grouped_forward_and_gradient_equivalent': True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['development', 'confirmation'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'development':
        assert not (OUT/'development.json').exists(), 'Refusing overwrite'
        (OUT/'source.py').write_text(Path(__file__).read_text())
        save(OUT/'verification.json', verify())
        save(OUT/'protocol.json', {
            'question': 'Does grouped nonlinear computation and activity-local plasticity reduce interference at similar resources?',
            'task': '4 phases; 8 shapes x 8 colours; 16 disjoint conjunctions per phase; balanced single-feature marginals and labels. Fixed binary label function, independent Gaussian observation noise. Not class-incremental or unseen-conjunction generalization.',
            'stream': 'No task IDs/boundary signals given to model, no optimizer/state resets, no replay, no route/cue banks; per-phase batching is experiment scheduler only.',
            'architecture': '16 groups x 4 ReLU branches summed /2 then 2-class head, versus 59-ReLU MLP. 1122 versus 1123 parameters. Linear-branch ablation is capacity diagnostic, not acquisition-matched.',
            'regulator': 'Accumulated mean-square hidden activity; incoming Adam update multiplied by 1/(1+lambda*past_exposure). Head unregulated. Shared=mean multipliers, shuffled=fresh permutation each update. Backprop remains global; activity state is local. No task reset. No top-down context or fading-memory claim.',
            'development': {'seeds': [700,701], 'order': 'forward', 'lr': LR, 'strength': STRENGTH},
            'selection': 'Each architecture/regulator independently maximizes harmonic mean of mean acquisition and final all-phase accuracy; ties choose lower strength then lower lr. Preserve all candidates. Select BEFORE independent confirmation.',
            'confirmation': 'Fresh 702-706 both orders; all selected arms and jointly trained architecture references; do not retune. Primary branched-local versus branched-shared; branch versus MLP and shuffled required for attribution.',
            'epochs': EPOCHS, 'train_per_phase': NTRAIN, 'test_per_phase': NEVAL,
            'batch': BATCH, 'noise_std': .05,
            'resource_note': 'FP32 weights and Adam moments counted; transient gradients/activations/Python overhead excluded from persistent table; update/forward timings separate. Exposure absent from deployment. Equivalent grouped MLP acknowledged.'})
        rows = []
        for architecture in ['branches', 'mlp', 'linear_branches']:
            regs = ['none', 'local', 'shared', 'shuffled'] if architecture != 'linear_branches' else ['none']
            for regulation in regs:
                for lr in LR:
                    for strength in ([0.] if regulation == 'none' else STRENGTH):
                        for seed in [700,701]:
                            row,_ = train(seed, architecture, regulation, lr, strength)
                            rows.append(row)
                        save(OUT/'development.json', rows)
                        recent = rows[-2:]
                        print('DEV', architecture, regulation, lr, strength,
                              'acc',round(st.mean(r['accuracy'] for r in recent)*100,2),
                              'acq',round(st.mean(r['acquisition'] for r in recent)*100,2),flush=True)
        selection = []
        for architecture,regulation in sorted({(r['architecture'],r['regulation']) for r in rows}):
            candidates = []
            for lr,strength in sorted({(r['lr'],r['strength']) for r in rows if (r['architecture'],r['regulation'])==(architecture,regulation)}):
                rr=[r for r in rows if (r['architecture'],r['regulation'],r['lr'],r['strength'])==(architecture,regulation,lr,strength)]
                a,b=st.mean(r['acquisition'] for r in rr),st.mean(r['accuracy'] for r in rr)
                candidates.append({'architecture': architecture,'regulation': regulation,'lr':lr,'strength':strength,'development_acquisition':a,'development_accuracy':b,'score':2*a*b/(a+b)})
            selection.append(max(candidates,key=lambda r:(r['score'],-r['strength'],-r['lr'])))
        save(OUT/'selection.json',selection)
        print('SELECTION',json.dumps(selection),flush=True)
    else:
        assert not (OUT/'confirmation.json').exists(), 'Refusing overwrite'
        selection=json.loads((OUT/'selection.json').read_text())
        # Added before any confirmation: slower Adam must not masquerade as
        # activity-local credit. Compare all rules at the local arm's settings.
        for architecture in ['branches','mlp']:
            local=next(r for r in selection if r['architecture']==architecture and r['regulation']=='local')
            for regulation in ['none','shared','shuffled']:
                strength=0. if regulation=='none' else local['strength']
                if not any(r['architecture']==architecture and r['regulation']==regulation and r['lr']==local['lr'] and r['strength']==strength for r in selection):
                    selection.append({'architecture':architecture,'regulation':regulation,'lr':local['lr'],'strength':strength,'variant':'local_setting_control'})
        save(OUT/'confirmation_addendum.json',{'written_before_any_confirmation':True,'reason':'Equal base LR and strength controls for local learning regulation; no new selected setting or sweep.','joint_reference':'Joint uses sequentially selected unregulated LR; not independently optimized, so a diagnostic only.','acquisition_gate':'Retention benefits require acquisition within 2 percentage points of corresponding control; report all effects regardless.'})
        save(OUT/'selection_lock.json',{'sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),'settings':selection,'seeds':list(range(702,707))})
        (OUT/'models').mkdir(exist_ok=True)
        rows=[]
        for seed in range(702,707):
            for reverse in [False,True]:
                for s in selection:
                    row,m=train(seed,s['architecture'],s['regulation'],s['lr'],s['strength'],reverse)
                    row['variant']=s.get('variant','selected')
                    rows.append(row)
                    torch.save(m.state_dict(),OUT/'models'/f"{seed}_{row['order']}_{s['architecture']}_{s['regulation']}_{row['variant']}.pt")
                for architecture in ['branches','mlp','linear_branches']:
                    s=next(r for r in selection if r['architecture']==architecture and r['regulation']=='none')
                    row,_=train(seed,architecture,'none',s['lr'],0.,reverse,joint=True)
                    rows.append(row)
                save(OUT/'confirmation.json',rows)
                print('CONFIRM',seed,'reverse' if reverse else 'forward',[(r['architecture'],r['regulation'],r['joint'],round(100*r['accuracy'],2),round(100*r['acquisition'],2)) for r in rows[-(len(selection)+3):]],flush=True)
        save(OUT/'completion.json',{'complete':True,'confirmation_runs':len(rows),'development_runs':len(json.loads((OUT/'development.json').read_text()))})

if __name__=='__main__':
    main()
