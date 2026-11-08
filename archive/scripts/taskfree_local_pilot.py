"""No task IDs: branch-response leaders on observable temporal regimes."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('dense','direct_topk','all_local','leader_local','fixed_local')

def make_data(seed,n,ambiguous=False):
    result=[]
    for regime in range(2):
        rng=torch.Generator().manual_seed(seed+regime*100)
        rho=torch.tensor([.9,.1] if regime==0 else [.1,.9])
        if ambiguous:rho=torch.tensor([.9,.9])
        state=torch.randn(n,2,generator=rng)
        history=[]
        for _ in range(9):
            history.append(state+.05*torch.randn(n,2,generator=rng))
            state=state*rho+torch.sqrt(1-rho.square())*torch.randn(n,2,generator=rng)
        # Predict next latent value of the slow channel; history identifies regime imperfectly.
        result.append((torch.stack(history[:8],1).flatten(1),history[8][:,regime]))
    return result

class LocalLayer(nn.Module):
    def __init__(self,inputs,branch_graph=False):
        super().__init__()
        self.branch_graph=branch_graph
        self.weight=nn.Parameter(torch.randn(32,4,inputs)/inputs**.5)
        self.bias=nn.Parameter(torch.zeros(32,4))
    def forward(self,x,arm,disable=False):
        a=F.relu(F.linear(x,self.weight.flatten(0,1),self.bias.flatten()).reshape(-1,32,4))
        score=a.flatten(1)
        probability=F.softmax(score/.2,dim=-1)
        winner=score.argmax(-1)
        leader=(F.one_hot(winner,128).float()+probability-probability.detach()).reshape(-1,32,4).sum(-1)
        if arm=='dense':mask=torch.ones(len(x),32)
        else:
            local=arm in ('all_local','leader_local','fixed_local')
            if arm=='fixed_local':
                message=torch.zeros_like(score if self.branch_graph else leader);message[:,0]=1
            elif arm=='all_local':message=score if self.branch_graph else a.mean(-1)
            elif arm=='leader_local':message=(hard_branch(probability,winner) if self.branch_graph else leader)
            else:message=a.mean(-1)
            if local and not disable:
                for _ in range(32 if self.branch_graph else 8):
                    message=(message+message.roll(1,1)+message.roll(-1,1))/3
            # Exact forward budget, differentiable surrogate for recruitment learning.
            if local and self.branch_graph:message=message.reshape(-1,32,4).sum(-1)
            ix=message.topk(16,dim=-1).indices
            hard=torch.zeros_like(message).scatter(1,ix,1)
            soft=16*F.softmax(message/.2,dim=-1)
            mask=hard+soft-soft.detach()
        return a.mean(-1)*mask,mask,winner

def hard_branch(probability,winner):
    return F.one_hot(winner,128).float()+probability-probability.detach()

class Model(nn.Module):
    def __init__(self,arm='dense',branch_graph=False):
        super().__init__();self.arm=arm;self.disable=False
        self.layers=nn.ModuleList([LocalLayer(16,branch_graph),LocalLayer(32,branch_graph)])
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def encode(self,x):
        states=[]
        for layer in self.layers:
            x,m,w=layer(x,self.arm,self.disable);states.append((m,w))
        return x,states
    def forward(self,x):return self.encode(x)[0]@self.readout

def fit(model,x,y,epochs,seed):
    opt=torch.optim.Adam(model.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ix in torch.randperm(len(x),generator=gen).split(64):
            opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x[ix]),y[ix])
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()

@torch.no_grad()
def evaluate(model,data):
    result=[]
    for x,y in data:
        z,states=model.encode(x);pred=z@model.readout
        model.disable=True;off=model(x);model.disable=False
        result.append({'mse':F.mse_loss(pred,y).item(),'no_communication_mse':F.mse_loss(off,y).item(),
            'prediction_delta':(pred-off).abs().mean().item(),
            'winner': [w.tolist() for _,w in states],
            'mean_masks':[m.mean(0).tolist() for m,_ in states]})
    # Balanced regime-decoding purity from winners, for analysis only; not learned/tested classifier.
    purity=[]
    for layer in range(2):
        counts=[torch.bincount(torch.tensor(r['winner'][layer]),minlength=128) for r in result]
        purity.append(torch.maximum(*counts).sum().item()/sum(len(r['winner'][layer]) for r in result))
    return {'tasks':result,'winner_regime_purity':purity}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--reverse',action='store_true');p.add_argument('--branch-graph',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);order=[1,0] if args.reverse else [0,1]
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':ARMS,'order':order,
        'data':'8-step history of two stationary AR(1) streams; rho .9/.1 swapped between regimes; predict next value of slow stream, observed with noise .05',
        'context':'no task IDs, no one-hot cue, no labels/rewards in input; only temporal history',
        'training':'1024 train per regime, 20 epochs each; known schedule for batching/optimizer reset only; labels provide regression loss',
        'testing':'1024 independent test windows per regime; random-regime mixed joint control; no test-time feedback or weight updates',
        'local':('128 individual branch compartments, single branch seed, 32 actual nearest-neighbor rounds, sum received messages within each neuron then top16 neurons' if args.branch_graph else '32 neuron groups, single leader seed, eight nearest-neighbor rounds, top16 neurons'),
        'capacity':'same allocated branch weights and fixed readout; routed arms 16 forward-active neurons, dense32; straight-through gradients for leader/recruitment',
        'selection':'fixed settings; no tuning on test; validation diagnostics only',
        'limits':'task boundaries known to optimizer; input regimes intentionally distinguishable statistically, not arbitrary hidden task switches; global argmax and topk are centralized; diffusion is local, not full decentralized election; label-conditioned objective, no router pretraining'},indent=2))
    (out/'source.py').write_text(Path(__file__).read_text());records=[];baseline=[]
    for seed in map(int,args.seeds.split(',')):
        data=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        for t,(x,y) in enumerate(test):
            baseline.append({'seed':seed,'regime':t,'zero':y.square().mean().item(),
                'oracle_last':F.mse_loss(.9*x.reshape(-1,8,2)[:,-1,t],y).item()})
        torch.manual_seed(seed);initial=Model(branch_graph=args.branch_graph)
        for arm in ARMS:
            model=copy.deepcopy(initial);model.arm=arm;joint=copy.deepcopy(model);states=[];stages=[]
            for task in order:
                fit(model,*data[task],args.epochs,seed+100+task)
                stages.append(evaluate(model,val));states.append(copy.deepcopy(model.state_dict()))
            fit(joint,torch.cat([d[0] for d in data]),torch.cat([d[1] for d in data]),args.epochs,seed+100)
            row={'seed':seed,'arm':arm,'validation':stages,'test':[]}
            for state in states:
                model.load_state_dict(state);row['test'].append(evaluate(model,test))
            row['joint_test']=evaluate(joint,test)
            torch.save({'stages':states,'joint':joint.state_dict()},out/f'{seed}_{arm}.pt')
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'old/new',*[round(row['test'][-1]['tasks'][t]['mse'],3) for t in order],flush=True)
    (out/'baselines.json').write_text(json.dumps(baseline,indent=2))
    lines=['# Task-ID-free local communication pilot','','Mean MSE across three seeds, lower is better. Purity is descriptive winner-to-regime association, not a held-out task decoder.','','| Arm | Old just learned | Old final | New final | Joint avg | Diffusion-off final avg | Winner purity L1 / L2 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        def avg(f):return sum(f(r) for r in rows)/len(rows)
        vals=[avg(lambda r:r['test'][0]['tasks'][order[0]]['mse']),avg(lambda r:r['test'][-1]['tasks'][order[0]]['mse']),avg(lambda r:r['test'][-1]['tasks'][order[1]]['mse']),avg(lambda r:sum(d['mse'] for d in r['joint_test']['tasks'])/2),avg(lambda r:sum(d['no_communication_mse'] for d in r['test'][-1]['tasks'])/2)]
        purity=' / '.join(f'{avg(lambda r:r["test"][-1]["winner_regime_purity"][i]):.3f}' for i in range(2))
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in vals)+f' | {purity} |')
    lines+=['',f'Zero baseline: {sum(b["zero"] for b in baseline)/len(baseline):.3f}; oracle slow-stream last-value baseline: {sum(b["oracle_last"] for b in baseline)/len(baseline):.3f}. Oracle uses hidden regime for analysis only.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
