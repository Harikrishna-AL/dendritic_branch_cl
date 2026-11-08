"""Historical neuron-mask recovery in ordinary and trace-gated MLPs.

Run with python -m bdh_cl.scripts.plain_history_config. All recovery is
task-assisted diagnostics at fixed final weights, with ten outputs eligible.
"""
import json
import statistics as st
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts import controlled_branch_depth as branch

OUT = Path('results/plain_history_config_20261021')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))


class Flat(nn.Module):
    def __init__(self, seed, gated):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = nn.Linear(784, 100)
        self.head = nn.Linear(100, 10)
        self.register_buffer('utility', torch.zeros(100))
        self.gated = gated

    def mask(self, k=25):
        mask = torch.zeros(100)
        mask[self.utility.topk(k).indices] = 1
        return mask

    def forward(self, x, mask=None, record=False):
        raw = torch.tanh(self.hidden(x))
        if mask is None:
            mask = self.mask() if self.gated else torch.ones(100)
        h = raw * mask
        if record:
            h.retain_grad()
        logits = self.head(h)
        return (logits, raw, h) if record else logits


@torch.no_grad()
def logits(m, x, config=None):
    if isinstance(m, Flat):
        return torch.cat([m(b, mask=config) for b in x.split(512)])
    return torch.cat([m(b, routes=config) for b in x.split(512)])


def accuracy(pred, y):
    return float((pred == y).float().mean())


@torch.no_grad()
def evaluate(m, bank, ex, ey, order):
    before = {k: v.clone() for k, v in m.state_dict().items()}
    latest = logits(m, ex)
    predictions = [logits(m, ex, c).argmax(-1) for c in bank]
    correct = torch.empty_like(ey)
    wrong = torch.empty_like(ey)
    masked = torch.empty_like(ey)
    for i, pair in enumerate(order):
        ids = torch.isin(ey, torch.tensor(pair))
        correct[ids] = predictions[i][ids]
        wrong[ids] = predictions[(i + 1) % 5][ids]
        pair_logits = latest[ids].clone()
        disallowed = torch.ones(10, dtype=torch.bool)
        disallowed[list(pair)] = False
        pair_logits[:, disallowed] = -torch.inf
        masked[ids] = pair_logits.argmax(-1)
    old = ~torch.isin(ey, torch.tensor(order[-1]))
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    return {
        'automatic': accuracy(latest.argmax(-1), ey),
        'historical_config': accuracy(correct, ey),
        'historical_old': accuracy(correct[old], ey[old]),
        'historical_new': accuracy(correct[~old], ey[~old]),
        'wrong_config': accuracy(wrong, ey),
        'any_config_correct_label_oracle': float(torch.stack([p == ey for p in predictions]).any(0).float().mean()),
        'pair_output_mask_separate_diagnostic': accuracy(masked, ey),
        'config_by_pair': [[accuracy(p[torch.isin(ey, torch.tensor(pair))], ey[torch.isin(ey, torch.tensor(pair))]) for pair in order] for p in predictions],
        'evaluation_state_unchanged': True,
    }


def fit(seed, arm, optimizer, lr, x, y, ex, ey, order, k=25, keep=False):
    m = branch.make(seed, 'branch1_proxy') if arm == 'branch' else Flat(seed, arm == 'flat_trace')
    opt = torch.optim.Adam(m.parameters(), lr=lr) if optimizer == 'adam' else torch.optim.SGD(m.parameters(), lr=lr, momentum=.9)
    bank, acquisition, acquisition_config, stages = [], [], [], []
    for stage, pair in enumerate(order):
        ids = torch.nonzero(torch.isin(y, torch.tensor(pair))).flatten()
        for epoch in range(2):
            gen = torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch)
            for ix in ids[torch.randperm(len(ids), generator=gen)].split(128):
                if arm == 'branch':
                    branch.step(m, opt, x[ix], y[ix], 'branch1_proxy')
                else:
                    opt.zero_grad(set_to_none=True)
                    log, raw, h = m(x[ix], record=True)
                    loss = F.cross_entropy(log, y[ix])
                    loss.backward()
                    # First-order benefit of retaining a unit versus zeroing it.
                    # dL/dh exists for every head input, including inactive units.
                    utility = -(h.grad.detach() * raw.detach()).sum(0)
                    nn.utils.clip_grad_norm_(m.parameters(), 5)
                    opt.step()
                    m.utility.mul_(.95).add_(utility, alpha=.05)
        config = m.routes().clone() if arm == 'branch' else m.mask(k if arm == 'plain' else 25)
        bank.append(config)
        current = torch.isin(ey, torch.tensor(pair))
        acquisition.append(accuracy(logits(m, ex[current]).argmax(-1), ey[current]))
        acquisition_config.append(accuracy(logits(m, ex[current], config).argmax(-1), ey[current]))
        if keep:
            stages.append({key: v.clone() for key, v in m.state_dict().items()})
    result = evaluate(m, bank, ex, ey, order)
    result.update(seed=seed, arm=arm, optimizer=optimizer, lr=lr, k=k,
                  acquisition=st.mean(acquisition), acquisition_with_recorded_config=st.mean(acquisition_config),
                  acquisition_by_pair=acquisition, parameters=sum(p.numel() for p in m.parameters()),
                  model_numeric_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()),
                  bank_tensor_bytes=sum(c.numel()*c.element_size() for c in bank),
                  bank_compact_encoding_bytes=125 if arm == 'branch' else 63)
    if keep:
        # Checkpoints are audit artifacts. No historical weight enters evaluate().
        torch.save({'final': m.state_dict(), 'bank': bank, 'stage_checkpoints_analysis_only': stages},
                   OUT / f'{seed}_{optimizer}_{arm}_{"forward" if order == SPLITS else "reverse"}.pt')
    return result


def verify():
    m = Flat(1, False)
    x = torch.randn(7, 784)
    y = torch.arange(7)
    log, raw, h = m(x, record=True)
    loss = F.cross_entropy(log, y)
    loss.backward()
    direction = torch.randn_like(h)
    eps = .001
    finite = (F.cross_entropy(m.head(h.detach()+eps*direction), y)-F.cross_entropy(m.head(h.detach()-eps*direction), y))/(2*eps)
    assert torch.allclose(finite, (h.grad*direction).sum(), atol=.0002, rtol=.02)
    assert torch.equal(m(x), m(x, mask=torch.ones(100)))
    torch.manual_seed(2)
    clone = Flat(1, False)
    clone.load_state_dict(m.state_dict())
    opt1 = torch.optim.Adam(m.parameters(), lr=.003)
    opt2 = torch.optim.Adam(clone.parameters(), lr=.003)
    # Passive trace recording cannot alter the plain network's learning path.
    opt1.zero_grad(); z,r,h=m(x,record=True);F.cross_entropy(z,y).backward()
    utility=-(h.grad.detach()*r.detach()).sum(0);opt1.step();m.utility.mul_(.95).add_(utility,alpha=.05)
    opt2.zero_grad();F.cross_entropy(clone(x),y).backward();opt2.step()
    assert all(torch.equal(a,b) for a,b in zip(m.parameters(),clone.parameters()))
    return {'trace_gradient_finite_difference': True, 'all_ones_equals_plain_forward': True,
            'passive_trace_does_not_change_plain_parameter_updates': True}


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    save('verification.json', verify())
    save('protocol.json', {
        'question': 'Can historical neuron-selection configurations recover old skills at final fixed weights in plain or flat trace-gated MLPs?',
        'data': 'Full MNIST pixels/255, five class pairs, two epochs/pair, batch128, all ten outputs eligible, no replay/pretraining/reset.',
        'arms': 'Plain 784->100 tanh->10, passive usefulness EMA; flat_trace identical MLP with global top25 usefulness gating, no groups/pooling; branch1_proxy 25x4 reference.',
        'trace': 'EMA .95 of negative sum_batch(dCE/dh * raw_response), same first-order criterion as branch alternative scoring relative to zero. Supervised gradients, not biologically local.',
        'optimization': 'Adam or SGD momentum.9, clip5. LRs chosen on development validation: Adam .001/.003, SGD .03/.1/.3.',
        'selection': 'Seed550 stratified50k/10k. Maximize historical-config validation accuracy with acquisition tiebreak, separately per arm/optimizer. Plain additionally selects k25/50/75/100. No official-test selection.',
        'confirmation': 'Seeds551-553, forward/reverse order, full60k/10k. Three independent seeds; orders paired.',
        'diagnostic': 'True digit pair selects phase-end mask; final weights/head unchanged. Wrong cyclic config and any-config label oracle separate. Output masking separate. Historical weights saved only for audit; never used in config recovery.',
        'limits': 'Finite config family; negative result does not prove all computations overwritten. Flat changes training rule; positive result would not establish vanilla optimizer retention. Counts include extra traces/configs. Matched approximate parameter budget, not identical active width/head.',
    })
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1)/255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1)/255
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    gen = torch.Generator().manual_seed(550)
    valid = torch.cat([torch.nonzero(y == c).flatten()[torch.randperm(int((y == c).sum()), generator=gen)[:1000]] for c in range(10)])
    mask = torch.ones(len(y), dtype=torch.bool);mask[valid] = False
    fitx, fity, vx, vy = x[mask], y[mask], x[valid], y[valid]
    selection = {}
    for optimizer, rates in [('adam', [.001,.003]), ('sgd', [.03,.1,.3])]:
        for arm in ['plain', 'flat_trace', 'branch']:
            candidates=[]
            for lr in rates:
                for k in ([25,50,75,100] if arm == 'plain' else [25]):
                    r=fit(550,arm,optimizer,lr,fitx,fity,vx,vy,SPLITS,k)
                    candidates.append(r)
                    print('DEV',optimizer,arm,lr,k,round(100*r['historical_config'],2),flush=True)
            choice=max(candidates,key=lambda r:(r['historical_config'],r['acquisition']))
            selection[f'{optimizer}_{arm}']={'chosen':choice,'candidates':candidates}
            save('selection.json',selection)
    rows=[]
    for seed in [551,552,553]:
        for order_name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for optimizer in ['adam','sgd']:
                for arm in ['plain','flat_trace','branch']:
                    chosen=selection[f'{optimizer}_{arm}']['chosen']
                    r=fit(seed,arm,optimizer,chosen['lr'],x,y,tx,ty,order,chosen['k'],True)
                    r['order']=order_name;rows.append(r);save('results.json',rows)
                    print('TEST',seed,order_name,optimizer,arm,'automatic',round(100*r['automatic'],2),'history',round(100*r['historical_config'],2),'acquisition',round(100*r['acquisition'],2),flush=True)
    save('completion.json',{'complete':True,'confirmation_runs':len(rows),'development_runs':sum(len(v['candidates']) for v in selection.values())})


if __name__ == '__main__':
    main()
