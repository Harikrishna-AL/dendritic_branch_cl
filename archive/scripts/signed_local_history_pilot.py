"""History-only branch leadership with signed nearest-neighbor messages."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.taskfree_local_pilot import make_data

ARMS=('dense','independent','all_signed','leader_signed','fixed_signed')

class Layer(nn.Module):
    def __init__(self,inputs):
        super().__init__();self.input=nn.Linear(inputs,128)
        self.keys=nn.Parameter(torch.randn(128,4)*.2)
    def forward(self,x,arm,disabled=False):
        z=self.input(x);response=torch.tanh(z)
        winner=response.abs().argmax(-1)
        probability=F.softmax(response.abs()/.2,-1)
        route=F.one_hot(winner,128).float()+probability-probability.detach()
        if arm=='fixed_signed':route=torch.zeros_like(route);route[:,0]=1
        keys=F.normalize(self.keys,dim=-1)
        message=response[:,:,None]*keys[None]
        if arm in ('leader_signed','fixed_signed'):message=message*route[:,:,None]
        if arm=='dense' or disabled:drive=torch.zeros_like(response)
        elif arm=='independent':drive=response
        else:
            # Only neighboring branch states can affect each synchronous update.
            for _ in range(32):message=(message+message.roll(1,1)+message.roll(-1,1))/3
            # Rescale singleton source dilution; fixed in advance, not tuned on test.
            scale=16 if arm in ('leader_signed','fixed_signed') else 1
            drive=scale*(message*keys[None]).sum(-1)
        gate=torch.sigmoid(4*drive)
        activity=F.relu(z)*gate
        return activity.reshape(-1,32,4).mean(-1),{'winner':winner,'gate':gate,'drive':drive}

class Model(nn.Module):
    def __init__(self,arm='dense'):
        super().__init__();self.arm=arm;self.disabled=False
        self.layers=nn.ModuleList([Layer(16),Layer(32)])
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def encode(self,x):
        states=[]
        for layer in self.layers:x,state=layer(x,self.arm,self.disabled);states.append(state)
        return x,states
    def forward(self,x):return self.encode(x)[0]@self.readout

def stream_batches(data,epochs,seed,order):
    # Task metadata constructs the environment schedule only; batches expose x/y alone.
    gen=torch.Generator().manual_seed(seed)
    for task in order:
        x,y=data[task]
        for _ in range(epochs):
            for ix in torch.randperm(len(x),generator=gen).split(64):yield x[ix],y[ix]

@torch.no_grad()
def evaluate(model,data):
    tasks=[]
    for x,y in data:
        z,states=model.encode(x);on=z@model.readout
        model.disabled=True;off=model(x);model.disabled=False
        tasks.append({'mse':F.mse_loss(on,y).item(),'messages_off_mse':F.mse_loss(off,y).item(),
            'prediction_delta':(on-off).abs().mean().item(),
            'layers':[{'winner':s['winner'].tolist(),'excited_fraction':(s['gate']>.51).float().mean().item(),
                       'inhibited_fraction':(s['gate']<.49).float().mean().item(),
                       'gate_delta':(s['gate']-.5).abs().mean().item()} for s in states]})
    return tasks

def fit_stream(model,batches,checkpoints,evaluator):
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    states=[];metrics=[]
    for step,(x,y) in enumerate(batches,1):
        opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x),y)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        # The observer records checkpoints; it does not reset optimizer/model or supply IDs.
        if step in checkpoints:
            states.append(copy.deepcopy(model.state_dict()));metrics.append(evaluator(model))
    return states,metrics

def diagnostic_decode(validation,test):
    acc=[]
    for layer in range(2):
        counts=[torch.bincount(torch.tensor(t['layers'][layer]['winner']),minlength=128) for t in validation]
        mapping=(counts[1]>counts[0]).long();hits=total=0
        for task,t in enumerate(test):
            w=torch.tensor(t['layers'][layer]['winner']);hits+=(mapping[w]==task).sum().item();total+=len(w)
        acc.append(hits/total)
    return acc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--reverse',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);order=[1,0] if args.reverse else [0,1]
    protocol={**vars(args),'arms':ARMS,'order':order,
        'input':'16 continuous values: eight historical observations of two AR1 processes; no IDs, context vector, target history, or boundary flag',
        'task':'rho .9/.1 swapped between regimes; predict next slow-channel observation; same generator and data seeds as taskfree_local_pilot',
        'learning':'single Adam optimizer throughout stream, no boundary resets; batches contain x and regression y only; regime IDs used solely to construct environment and analyze checkpoints',
        'leader':'largest absolute signed tanh branch response; one branch per layer; abs criterion admits inhibitory/suppressed leader; straight-through softmax .2 gradient',
        'local':'128 branch compartments, learned four-dimensional keys; signed response times source key; 32 nearest-neighbor averaging rounds; received key compatibility modulates gate above/below .5',
        'controls':'dense gate .5, independent local response gates, all-source signed messaging, fixed-source messaging; fixed output projection; allocated capacity identical, effective activity differs',
        'sizes':'1024 train /512 validation /1024 test windows per regime;20 epochs; joint control matches steps and presentations',
        'selection':'all constants specified before results; no test selection',
        'limits':'supervised prediction targets during training; explicit global leader election; ReLU feedforward and signed modulatory inhibition, not biophysical inhibitory neurons; no replay, protection or pretraining; labels never supplied at inference'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    records=[]
    for seed in map(int,args.seeds.split(',')):
        data=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        torch.manual_seed(seed);initial=Model();steps=args.epochs*16
        for arm in ARMS:
            model=copy.deepcopy(initial);model.arm=arm;joint=copy.deepcopy(model)
            states,validation=fit_stream(model,stream_batches(data,args.epochs,seed+100,order),[steps,2*steps],lambda m:evaluate(m,val))
            jointdata=[(torch.cat([d[0] for d in data]),torch.cat([d[1] for d in data]))]
            fit_stream(joint,stream_batches(jointdata,args.epochs,seed+100,[0]),[],lambda m:None)
            row={'seed':seed,'arm':arm,'validation':validation,'test':[]}
            for state in states:model.load_state_dict(state);row['test'].append(evaluate(model,test))
            row['joint_test']=evaluate(joint,test);row['winner_regime_decode']=diagnostic_decode(validation[-1],row['test'][-1])
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            torch.save({'stages':states,'joint':joint.state_dict()},out/f'{seed}_{arm}.pt')
            print(seed,arm,'old/new',*[round(row['test'][-1][t]['mse'],3) for t in order],flush=True)
    lines=['# Signed branch communication, continuous optimizer','','MSE means across paired seeds; lower is better. Decode uses validation-derived mapping on independent test.','','| Arm | Old just learned | Old final | New final | Joint avg | Messages-off final avg | Winner decode L1 / L2 | Excited / inhibited fractions |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        def avg(f):return sum(f(r) for r in rows)/len(rows)
        v=[avg(lambda r:r['test'][0][order[0]]['mse']),avg(lambda r:r['test'][-1][order[0]]['mse']),avg(lambda r:r['test'][-1][order[1]]['mse']),avg(lambda r:sum(t['mse'] for t in r['joint_test'])/2),avg(lambda r:sum(t['messages_off_mse'] for t in r['test'][-1])/2)]
        decode=' / '.join(f'{avg(lambda r:r["winner_regime_decode"][i])*100:.1f}%' for i in range(2))
        excited=avg(lambda r:sum(s['excited_fraction'] for t in r['test'][-1] for s in t['layers'])/4)
        inhibited=avg(lambda r:sum(s['inhibited_fraction'] for t in r['test'][-1] for s in t['layers'])/4)
        lines.append(f'| {arm} | '+' | '.join(f'{x:.3f}' for x in v)+f' | {decode} | {excited:.3f} / {inhibited:.3f} |')
    lines+=['','Same task baselines: zero output ~1.000; oracle ~0.199; history-only heuristic ~0.295. These baselines come from the preceding run with identical held-out seeds.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
