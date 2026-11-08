"""DCNet-inspired task-aware branch pilot, not a DCNet reproduction.

Known training boundaries; task-specific protected branch configurations;
fixed orthogonal targets and optional aggregation compensation. Inference
accepts inputs only and compares every learned head. No old observations.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.cifar100 import load


class BranchBank(nn.Module):
    def __init__(self, seed, width=128, tasks=10, inputs=3072):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.tasks = width, tasks
        self.first = nn.Linear(inputs, width * tasks)
        self.register_buffer('codes', torch.linalg.qr(torch.randn(width, width)).Q[:, :100].T.contiguous())
        self.register_buffer('routes', torch.full((tasks, width), -1, dtype=torch.long))
        self.register_buffer('used', torch.zeros(width, tasks, dtype=torch.bool))
        self.register_buffer('aggregation', torch.zeros(tasks))
        self.stages = 0

    @torch.no_grad()
    def allocate(self, x):
        # Current-task data only. Select a high-response unused branch per neuron.
        response = self.first(x).tanh().reshape(-1, self.width, self.tasks)
        utility = response.square().mean(0).masked_fill(self.used, -torch.inf)
        route = utility.argmax(-1)
        self.routes[self.stages].copy_(route)
        self.used[torch.arange(self.width), route] = True
        ids = torch.arange(self.width) * self.tasks + route
        layer = nn.Linear(self.first.in_features, self.width)
        layer.weight.copy_(self.first.weight[ids])
        layer.bias.copy_(self.first.bias[ids])
        return layer, ids

    @torch.no_grad()
    def commit(self, layer, ids, aggregation):
        self.first.weight[ids] = layer.weight
        self.first.bias[ids] = layer.bias
        self.aggregation[self.stages] = aggregation
        self.stages += 1

    def scores(self, x):
        # No labels, true task IDs, or state writes in this inference function.
        response = self.first(x).tanh().reshape(-1, self.width, self.tasks)
        hidden = response[:, torch.arange(self.width)[None], self.routes[:self.stages]]
        z = F.normalize(hidden, dim=-1)
        task_codes = self.codes[:10*self.stages].reshape(self.stages, 10, self.width)
        return torch.einsum('btd,tcd->btc', z, task_codes)

    def predict(self, x, rule='head_probability'):
        cosine = self.scores(x)
        score = (cosine / .1).softmax(-1) if rule == 'head_probability' else cosine
        return score.flatten(1).argmax(-1)


def supervised_contrastive(z, y, temperature):
    sim = z @ z.T / temperature
    diagonal = torch.eye(len(y), dtype=torch.bool)
    positive = (y[:, None] == y[None]) & ~diagonal
    logp = sim - sim.masked_fill(diagonal, -torch.inf).logsumexp(-1)[:, None]
    valid = positive.any(-1)
    if not valid.any():
        return z.sum() * 0
    return -(logp.masked_fill(~positive, 0).sum(-1) / positive.sum(-1).clamp_min(1))[valid].mean()


@torch.no_grad()
def evaluate(model, x, y):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    outputs = torch.cat([model.scores(b) for b in x.split(256)])
    result = {}
    for name, score in [('head_probability', (outputs/.1).softmax(-1)), ('cosine', outputs)]:
        pred = score.flatten(1).argmax(-1)
        result[name] = {'accuracy': float((pred == y).float().mean()),
                        'task_accuracy': [float((pred[y//10 == t] == y[y//10 == t]).float().mean()) for t in range(model.stages)],
                        'routing_accuracy': float((pred//10 == y//10).float().mean())}
    # True task used only for this explicitly labelled diagnostic.
    oracle = outputs[torch.arange(len(y)), y//10].argmax(-1) + (y//10)*10
    result['oracle_task_diagnostic'] = float((oracle == y).float().mean())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(model.predict(x[:11]), (outputs[:11]/.1).softmax(-1).flatten(1).argmax(-1))
    return result


def checks():
    model = BranchBank(17, inputs=12)
    x = torch.rand(15, 12)
    layer, ids = model.allocate(x)
    model.commit(layer, ids, .5)
    oldw, oldb = model.first.weight[ids].clone(), model.first.bias[ids].clone()
    layer2, ids2 = model.allocate(x)
    assert not torch.isin(ids2, ids).any()
    with torch.no_grad():
        layer2.weight.add_(1)
    model.commit(layer2, ids2, .4)
    assert torch.equal(oldw, model.first.weight[ids]) and torch.equal(oldb, model.first.bias[ids])
    assert torch.allclose(model.codes @ model.codes.T, torch.eye(100), atol=1e-5)
    evaluate(model, x, torch.arange(15)%20)
    return {'orthogonal_codes': True, 'disjoint_routes': True, 'protected_weights_unchanged': True,
            'prediction_input_only': True, 'evaluation_state_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', default='701,702,703')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--arms', default='ioe_dac,ioe')
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2))
    save('verification.json', checks())
    (out/'source.py').write_text(Path(__file__).read_text())
    train, valid, test, order = load('data_cache/cifar100')
    x = torch.from_numpy(train[0].copy()).float().flatten(1)/255
    y = torch.from_numpy(train[1].copy()).long()
    tx = torch.from_numpy(test[0].copy()).float().flatten(1)/255
    ty = torch.from_numpy(test[1].copy()).long()
    # Fixed data-independent centering: no lookahead at future tasks.
    mean = torch.full((3072,), .5)
    x.sub_(mean); tx.sub_(mean)
    save('protocol.json', {'dataset':'Split CIFAR100, 10x10, 45000 fit/5000 unused validation/10000 official test',
        'class_order':order, 'seeds':args.seeds, 'epochs':args.epochs,
        'architecture':'Raw pixels -> 128 groups of 10 tanh branches. Fixed random orthonormal class directions. No pretrained representation.',
        'allocation':'At known boundary pick one unused branch/group by mean squared current-task response on first 512 current examples. No reuse; fixed capacity for ten tasks.',
        'protection':'Train extracted current rows only; write back at boundary. Previously committed rows and targets immutable.',
        'loss':'Current-head vMF cross entropy temperature .1 + squared distance to fixed class target; ioe_dac adds .1 supervised contrastive loss after first epoch, temperature clamped [.03,.3] relative to historical concentration.',
        'inference':'Primary: concatenate per-task softmax class probabilities and global argmax. Secondary: concatenate unnormalized cosine scores and argmax. No true task/label supplied. Both fixed before test, no selection.',
        'metrics':'Average of seen-class accuracy after each task and final all100 accuracy; oracle task diagnostic separate.',
        'normalization':'Fixed pixel centering by .5, no dataset fitting or future-task lookahead.',
        'no_replay':'Only current-class data used in each update. No retained examples/features or joint calibration. Full dataset tensors held by stream harness, never exposed again after class block.',
        'differences_from_dcnet':'Adapted branch backbone with disjoint hard allocation, fixed QR targets, simpler compaction and fixed 5-epoch budget. Not original HAT, ResNet, basis-generator or full DCNet implementation. Capacity/compute not matched to prior24x4 model.',
        'memory':'Persistent actual state_dict tensor bytes plus input centering. Peak training RSS and inference allocation not measured. Optimizer/current extracted layer temporary.'})
    rows=[]
    for seed in map(int,args.seeds.split(',')):
        for arm in args.arms.split(','):
            model=BranchBank(seed); history=[]; start=time.perf_counter()
            for task in range(10):
                ids=torch.nonzero(y//10==task).flatten(); cx=x[ids]; cy=y[ids]
                old=model.first.weight.clone(); oldbias=model.first.bias.clone(); protected=model.used.flatten().clone()
                layer, selected=model.allocate(cx[:512]); opt=torch.optim.Adam(layer.parameters(),lr=.001)
                temperature=.1
                for epoch in range(args.epochs):
                    generator=torch.Generator().manual_seed(seed*10000+task*100+epoch)
                    for batch in torch.randperm(len(cy),generator=generator).split(128):
                        z=F.normalize(layer(cx[batch]).tanh(),dim=-1); target=model.codes[cy[batch]]
                        logits=z@model.codes[task*10:(task+1)*10].T/.1
                        loss=F.cross_entropy(logits,cy[batch]%10)+(z-target).square().sum(-1).mean()
                        if arm=='ioe_dac' and epoch>0:
                            loss=loss+.1*supervised_contrastive(z,cy[batch],temperature)
                        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(layer.parameters(),5);opt.step()
                    with torch.no_grad():
                        concentration=float((F.normalize(layer(cx[:512]).tanh(),dim=-1)*model.codes[cy[:512]]).sum(-1).mean())
                        reference=float(model.aggregation[:task].mean()) if task else concentration
                        temperature=max(.03,min(.3,.1*max(concentration,.01)/max(reference,.01)))
                model.commit(layer,selected,concentration)
                assert torch.equal(old[protected],model.first.weight[protected])
                assert torch.equal(oldbias[protected],model.first.bias[protected])
                seen=ty//10<=task; result=evaluate(model,tx[seen],ty[seen]);history.append(result)
                save('progress.json',{'seed':seed,'arm':arm,'task':task+1,'history':history})
                print(seed,arm,'task',task+1,'primary',round(result['head_probability']['accuracy']*100,2),'cosine',round(result['cosine']['accuracy']*100,2),'oracle',round(result['oracle_task_diagnostic']*100,2),flush=True)
            row={'seed':seed,'arm':arm,'history':history,'average_incremental':{r:statistics.mean(h[r]['accuracy'] for h in history) for r in ['head_probability','cosine']},
                 'final':history[-1],'persistent_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values())+mean.numel()*mean.element_size(),'seconds':time.perf_counter()-start}
            rows.append(row);save('results.json',rows)
            torch.save({'model':model.state_dict(),'mean':mean,'stages':model.stages},out/f'{seed}_{arm}.pt')
    summary={}
    for arm in args.arms.split(','):
        rr=[r for r in rows if r['arm']==arm];summary[arm]={}
        for rule in ['head_probability','cosine']:
            values=[r['final'][rule]['accuracy']*100 for r in rr]
            summary[arm][rule]={'final_mean_percent':statistics.mean(values),'final_sd_percent':statistics.stdev(values) if len(values)>1 else None,'average_incremental_percent':statistics.mean(r['average_incremental'][rule]*100 for r in rr)}
        summary[arm]['persistent_bytes']=rr[0]['persistent_bytes']
    save('summary.json',summary);save('completion.json',{'complete':True,'runs':len(rows)})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
