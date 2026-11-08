"""Context-dependent regression through a fixed, shared output projection."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.local_orchestrator_pilot import Layer

class SignalNet(torch.nn.Module):
    def __init__(self, arm='context', gain=1.):
        super().__init__()
        self.arm,self.gain=arm,gain
        self.layers=torch.nn.ModuleList([Layer(10),Layer(32)])
        # Balanced fixed projection, same across seeds and arms; no learned head.
        self.register_buffer('readout',torch.cat([torch.ones(16),-torch.ones(16)])/4)
        self.decoder=torch.nn.Linear(32,8)
    def encode(self,x):
        diagnostic=[]
        for layer in self.layers:
            x,g,w=layer(x,self.arm,4,self.gain)
            diagnostic.append((g,w))
        return x,diagnostic
    def forward(self,x): return self.encode(x)[0]@self.readout

def make_data(seed,n):
    g=torch.Generator().manual_seed(seed)
    latent=torch.rand(n,2,generator=g)*2-1
    noisy=latent[:,:,None]+.2*torch.randn(n,2,4,generator=g)
    result=[]
    for task in range(2):
        cue=F.one_hot(torch.full((n,),task),2).float()
        result.append((torch.cat([noisy.flatten(1),cue],1),latent[:,task]))
    return result

def train(model,x,y,epochs,seed,pretrain=False):
    params=list(model.layers.parameters())+(list(model.decoder.parameters()) if pretrain else [])
    opt=torch.optim.Adam(params,lr=.003)
    rng=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ix in torch.randperm(len(x),generator=rng).split(64):
            opt.zero_grad(set_to_none=True)
            if pretrain:
                loss=F.mse_loss(model.decoder(model.encode(x[ix])[0]),x[ix,:8])
            else: loss=F.mse_loss(model(x[ix]),y[ix])
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,5);opt.step()

@torch.no_grad()
def evaluate(model,data):
    result=[]
    for x,y in data:
        pred=model(x);z,diag=model.encode(x)
        saved=model.gain;model.gain=0;off=model(x);model.gain=saved
        flipped=x.clone();flipped[:,-2:]=flipped[:,-2:].flip(1)
        layers=[];incoming=x
        for layer in model.layers:
            outgoing,g,w=layer(incoming,model.arm,4,model.gain)
            _,go,_=layer(incoming,model.arm,4,0)
            hist=torch.bincount(w,minlength=128).float()/len(x)
            layers.append({'gate_delta':(g-go).abs().mean().item(),
                'winner_max_share':hist.max().item(),'winner_hist':hist.tolist()})
            incoming=outgoing
        result.append({'mse':F.mse_loss(pred,y).item(),'off_mse':F.mse_loss(off,y).item(),
            'r2':1-F.mse_loss(pred,y).item()/y.var(unbiased=False).item(),
            'message_prediction_delta':(pred-off).abs().mean().item(),
            'cue_prediction_delta':(pred-model(flipped)).abs().mean().item(),
            'layers':layers})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--epochs',type=int,default=20)
    args=p.parse_args();torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    variants=[('context',0),('local',1),('local',4),('winner',1),('winner',4)]
    protocol={**vars(args),'variants':variants,'data':'two independent uniform latent signals, four noisy observations each (sigma .2), explicit two-element context cue; same observations across contexts',
        'sizes':{'train_per_context':1024,'validation_per_context':512,'test_per_context':1024,'pretrain':1024},
        'readout':'fixed balanced signed sum of final 32 neurons divided by 4; never trained',
        'pretrain':'scratch or eight-epoch reconstruction of eight noisy observations, separate first-context training draws only; no future-context exposure',
        'selection':'gain per arm/pretraining condition using mean final sequential validation MSE across paired seeds; lower gain breaks ties',
        'controls':'joint training at same total downstream data presentations; observed signal mean baseline; zero prediction baseline',
        'limits':'explicit context, soft gates, synthetic two-task pilot; pretraining adds steps; no protection/replay; fixed readout still constrains representation'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    (out/'layer_source.py').write_text(Path(__file__).with_name('local_orchestrator_pilot.py').read_text())
    records=[];models={};testsets={}
    for seed in map(int,args.seeds.split(',')):
        data=make_data(seed+1000,1024);val=make_data(seed+2000,512)
        testsets[seed]=make_data(seed+3000,1024)
        pre=make_data(seed+4000,1024)[0]
        torch.manual_seed(seed);initial=SignalNet()
        for pretraining in ('scratch','autoencoder'):
            for arm,gain in variants:
                model=copy.deepcopy(initial);model.arm=arm;model.gain=gain
                if pretraining=='autoencoder':train(model,*pre,8,seed+50,True)
                joint=copy.deepcopy(model)
                stages=[];states=[]
                for task,(x,y) in enumerate(data):
                    train(model,x,y,args.epochs,seed+100+task)
                    stages.append(evaluate(model,val));states.append(copy.deepcopy(model.state_dict()))
                train(joint,torch.cat([d[0] for d in data]),torch.cat([d[1] for d in data]),args.epochs,seed+100)
                key=f'{seed}_{pretraining}_{arm}_{gain}'
                models[key]=(states,copy.deepcopy(joint.state_dict()))
                torch.save(models[key],out/f'{key}.pt')
                records.append({'seed':seed,'pretraining':pretraining,'arm':arm,'gain':gain,'validation':stages,'joint_validation':evaluate(joint,val)})
                print(key,'final validation mse',round(sum(d['mse'] for d in stages[-1])/2,4),flush=True)
    chosen={}
    for pretraining in ('scratch','autoencoder'):
        for arm in ('context','local','winner'):
            candidates=[r for r in records if r['arm']==arm and r['pretraining']==pretraining]
            gains=sorted(set(r['gain'] for r in candidates))
            def score(g):
                rows=[r for r in candidates if r['gain']==g]
                return sum(sum(d['mse'] for d in r['validation'][-1])/2 for r in rows)/len(rows)
            chosen[f'{pretraining}_{arm}']=min(gains,key=lambda g:(score(g),g))
    (out/'selection.json').write_text(json.dumps(chosen,indent=2))
    baselines={}
    for seed,data in testsets.items():
        baselines[seed]=[{'zero_mse':y.square().mean().item(),'observed_mean_mse':F.mse_loss(x[:,4*t:4*t+4].mean(1),y).item()} for t,(x,y) in enumerate(data)]
    for r in records:
        if r['gain']!=chosen[f'{r["pretraining"]}_{r["arm"]}']:continue
        key=f'{r["seed"]}_{r["pretraining"]}_{r["arm"]}_{r["gain"]}'
        states,jointstate=models[key];model=SignalNet(r['arm'],r['gain'])
        r['test']=[]
        for state in states:
            model.load_state_dict(state);r['test'].append(evaluate(model,testsets[r['seed']]))
        model.load_state_dict(jointstate);r['joint_test']=evaluate(model,testsets[r['seed']])
    (out/'results.json').write_text(json.dumps(records,indent=2));(out/'baselines.json').write_text(json.dumps(baselines,indent=2))
    lines=['# Fixed-readout signal pilot','','Lower MSE is better. Means across paired seeds; gain selected on validation.','','| Initial training | Mechanism | Gain | Old MSE just learned | Old MSE final | New MSE | Joint avg MSE | Message-off final avg |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pretraining in ('scratch','autoencoder'):
        for arm in ('context','local','winner'):
            rows=[r for r in records if r['pretraining']==pretraining and r['arm']==arm and 'test' in r]
            def avg(f):return sum(f(r) for r in rows)/len(rows)
            vals=[avg(lambda r:r['test'][0][0]['mse']),avg(lambda r:r['test'][-1][0]['mse']),avg(lambda r:r['test'][-1][1]['mse']),avg(lambda r:sum(d['mse'] for d in r['joint_test'])/2),avg(lambda r:sum(d['off_mse'] for d in r['test'][-1])/2)]
            lines.append(f'| {pretraining} | {arm} | {chosen[f"{pretraining}_{arm}"]} | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    zero=sum(d['zero_mse'] for ds in baselines.values() for d in ds)/6
    observed=sum(d['observed_mean_mse'] for ds in baselines.values() for d in ds)/6
    lines+=['',f'Zero-output baseline MSE: {zero:.4f}. Context-cued observed-mean baseline: {observed:.4f}.','', 'No classifier head is trained. Any sequential degradation arises in the trainable internal computation. Pre-training adds steps and is exploratory; no matched extra-training control in this pilot.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
