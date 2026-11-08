"""Evidence-anchored local correction, trained across inference horizons."""
import argparse,copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.taskfree_local_pilot import make_data
from bdh_cl.scripts.signed_local_history_pilot import stream_batches

class Layer(nn.Module):
    def __init__(self,inputs):
        super().__init__();self.activity=nn.Linear(inputs,128);self.context=nn.Linear(inputs,512)
        self.evidence=nn.Linear(4,4,bias=False);self.self_mix=nn.Linear(4,4,bias=False)
        self.neighbor_mix=nn.Linear(4,4,bias=False);self.receiver=nn.Parameter(torch.randn(128,4)*.2)
    def forward(self,x,rounds,exchange):
        initial=torch.tanh(self.context(x).reshape(-1,128,4));c=initial
        residual=torch.zeros(len(x),device=x.device)
        for _ in range(rounds):
            neighbors=(c.roll(1,1)+c.roll(-1,1))/2 if exchange else c
            # Evidence is reintroduced every step; correction is toward an evidence-conditioned proposal.
            proposal=torch.tanh(self.evidence(initial)+self.self_mix(c)+self.neighbor_mix(neighbors))
            correction=proposal-c
            c=c+.25*correction
            residual=correction.square().mean((1,2)).sqrt()
        gate=torch.sigmoid((c*self.receiver[None]).sum(-1))
        h=(F.relu(self.activity(x))*gate).reshape(-1,32,4).mean(-1)
        return h,{'context':c,'residual':residual}

class Model(nn.Module):
    def __init__(self,exchange=True):
        super().__init__();self.exchange=exchange;self.rounds=8
        self.layers=nn.ModuleList([Layer(16),Layer(32)])
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def encode(self,x,rounds=None):
        states=[]
        for layer in self.layers:x,s=layer(x,self.rounds if rounds is None else rounds,self.exchange);states.append(s)
        return x,states
    def forward(self,x):return self.encode(x)[0]@self.readout

def fit(model,batches,variable,seed,checkpoints):
    opt=torch.optim.Adam(model.parameters(),lr=.003);rng=torch.Generator().manual_seed(seed)
    states=[];horizons=(4,8,16,32)
    for step,(x,y) in enumerate(batches,1):
        model.rounds=horizons[int(torch.randint(4,(1,),generator=rng))] if variable else 8
        opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x),y)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step in checkpoints:states.append(copy.deepcopy(model.state_dict()))
    model.rounds=8
    return states

@torch.no_grad()
def mismatch(model,x,rounds=8):
    pred,states=model.encode(x,rounds)
    return pred@model.readout,states,torch.stack([s['residual'] for s in states]).mean(0)

@torch.no_grad()
def policy(model,x,mode,threshold):
    rounds=int(mode) if mode!='adaptive' else 8
    pred,states,score=mismatch(model,x,rounds)
    contexts=[s['context'] for s in states];trigger=torch.zeros(len(x),dtype=torch.bool)
    if mode=='adaptive':
        trigger=score>threshold
        if trigger.any():
            longer,ls,_=mismatch(model,x[trigger],32);pred=pred.clone();pred[trigger]=longer
            contexts=[c.clone() for c in contexts]
            for i,s in enumerate(ls):contexts[i][trigger]=s['context']
    return pred,contexts,trigger,score

@torch.no_grad()
def decode(val,test):
    a=torch.cat(val).flatten(1);b=torch.cat(test).flatten(1)
    y=torch.cat([-torch.ones(len(val[0])),torch.ones(len(val[1]))])
    target=torch.cat([torch.zeros(len(test[0])),torch.ones(len(test[1]))]).bool()
    mean=a.mean(0);std=a.std(0).clamp_min(.01)
    a=torch.cat([(a-mean)/std,torch.ones(len(a),1)],1);b=torch.cat([(b-mean)/std,torch.ones(len(b),1)],1)
    coef=torch.linalg.solve(a.T@a+10*torch.eye(a.shape[1]),a.T@y)
    return ((b@coef>0)==target).float().mean().item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seeds',default='0,1,2');args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records=[]
    arms=[('self_fixed',False,False),('neighbor_fixed',True,False),('self_variable',False,True),('neighbor_variable',True,True)]
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':arms,
        'task':'same history-only AR1 regimes, same seed offsets/data sizes; targets only in regression loss; no task IDs at prediction',
        'update':'evidence-conditioned proposal minus current branch context; residual correction rate .25; raw projected evidence clamped and reintroduced every round',
        'training':'single Adam throughout sequential stream; fixed8 vs uniformly sampled4/8/16/32 per batch; matched presentations and optimizer steps, variable has more communication compute; no replay/protection',
        'trigger':'mean proposal-state residual after8 rounds;95th percentile on initial-regime training inputs at checkpoint1, frozen thereafter; no IDs/targets in trigger',
        'policies':['4','8','16','32','64','adaptive'],
        'diagnostics':'first/final/joint MSE, trigger rates, state residual, validation-fitted layer2 cue decoder on independent test',
        'scope':'correction is latent proposal residual, not direct task-error feedback; random-horizon training uses supervised prediction loss; input anchoring and varying horizon are separated by fixed/variable controls',
        'selection':'no test selection; adaptive8->32 preset;32 within variable training horizon,64 extrapolation',
        'compute':'adaptive selected inputs recomputed from scratch, so costs8+32 although chosen total horizon is32'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        data=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        torch.manual_seed(seed);initial=Model();steps=args.epochs*16
        for arm,exchange,variable in arms:
            model=copy.deepcopy(initial);model.exchange=exchange;joint=copy.deepcopy(model)
            states=fit(model,stream_batches(data,args.epochs,seed+100,[0,1]),variable,seed+400,[steps,2*steps])
            fit(joint,stream_batches([(torch.cat([d[0] for d in data]),torch.cat([d[1] for d in data]))],args.epochs,seed+100,[0]),variable,seed+400,[])
            model.load_state_dict(states[0]);threshold=torch.quantile(mismatch(model,data[0][0])[2],.95).item()
            torch.save({'stages':states,'joint':joint.state_dict()},out/f'{seed}_{arm}.pt')
            for stage,state in [('first',states[0]),('final',states[1]),('joint',joint.state_dict())]:
                model.load_state_dict(state)
                for mode in ('4','8','16','32','64','adaptive'):
                    features=[];tasks=[]
                    for dataset in (val,test):
                        layer=[]
                        for x,y in dataset:
                            pred,c,trigger,residual=policy(model,x,mode,threshold);layer.append(c[1])
                            if dataset is test:tasks.append({'mse':F.mse_loss(pred,y).item(),'trigger_rate':trigger.float().mean().item(),'residual':residual.mean().item()})
                        features.append(layer)
                    records.append({'seed':seed,'arm':arm,'stage':stage,'policy':mode,'threshold':threshold,'tasks':tasks,'cue_decode':decode(*features)})
            (out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'completed',flush=True)
    lines=['# Evidence-anchored recurrent correction','','Means across three paired seeds; MSE lower is better. All inference weights frozen.','','| Arm | Inference rounds | Old final MSE | New final MSE | Joint avg MSE | Cue decode | Old/new trigger |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,_,_ in arms:
        for mode in ('8','32','adaptive'):
            rows=[r for r in records if r['arm']==arm and r['stage']=='final' and r['policy']==mode]
            jointrows=[r for r in records if r['arm']==arm and r['stage']=='joint' and r['policy']==mode]
            def avg(f):return sum(f(r) for r in rows)/len(rows)
            old,new=[avg(lambda r:r['tasks'][i]['mse']) for i in (0,1)]
            jointmean=sum(sum(t['mse'] for t in r['tasks'])/2 for r in jointrows)/len(jointrows)
            trigger=' / '.join(f'{avg(lambda r:r["tasks"][i]["trigger_rate"])*100:.1f}%' for i in (0,1))
            lines.append(f'| {arm} | {mode} | {old:.3f} | {new:.3f} | {jointmean:.3f} | {avg(lambda r:r["cue_decode"])*100:.1f}% | {trigger} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
