"""Collective local context versus independent branches, without supplied task IDs."""
import argparse,copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.taskfree_local_pilot import make_data
from bdh_cl.scripts.signed_local_history_pilot import stream_batches,fit_stream
ARMS=('independent','average','learned')

class Layer(nn.Module):
    def __init__(self,inputs):
        super().__init__();self.activity=nn.Linear(inputs,128)
        self.context=nn.Linear(inputs,128*4)
        self.self_mix=nn.Linear(4,4,bias=False)
        self.left_mix=nn.Linear(4,4,bias=False)
        self.right_mix=nn.Linear(4,4,bias=False)
        self.receiver=nn.Parameter(torch.randn(128,4)*.2)
    def communicate(self,c,arm,disabled=False):
        for _ in range(8):
            if arm=='independent' or disabled:l,r=c,c
            else:l,r=c.roll(1,1),c.roll(-1,1)
            if arm=='average':center=(c+l+r)/3;l,r=center,center
            else:center=c
            update=torch.tanh(self.self_mix(center)+self.left_mix(l)+self.right_mix(r))
            c=.75*c+.25*update
        return c
    def forward(self,x,arm,disabled=False):
        initial=torch.tanh(self.context(x).reshape(-1,128,4))
        c=self.communicate(initial,arm,disabled)
        gate=torch.sigmoid((c*self.receiver[None]).sum(-1))
        h=(F.relu(self.activity(x))*gate).reshape(-1,32,4).mean(-1)
        return h,{'context':c,'initial':initial,'gate':gate}

class Model(nn.Module):
    def __init__(self,arm='independent'):
        super().__init__();self.arm=arm;self.disabled=False
        self.layers=nn.ModuleList([Layer(16),Layer(32)])
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def encode(self,x):
        states=[]
        for l in self.layers:x,s=l(x,self.arm,self.disabled);states.append(s)
        return x,states
    def forward(self,x):return self.encode(x)[0]@self.readout

@torch.no_grad()
def measure(model,data):
    rows=[]
    for x,y in data:
        z,states=model.encode(x);on=z@model.readout
        model.disabled=True;off=model(x);model.disabled=False
        rows.append({'mse':F.mse_loss(on,y).item(),'exchange_off_mse':F.mse_loss(off,y).item(),
                     'prediction_delta':(on-off).abs().mean().item(),
                     'gate_mean':[s['gate'].mean().item() for s in states]})
    return rows

@torch.no_grad()
def cue_decode(model,validation,test):
    # All regime labels below are observer-only, never training losses or routing inputs.
    extracted=[]
    for data in (validation,test):
        layers=[[],[]]
        for x,_ in data:
            _,states=model.encode(x)
            for i,s in enumerate(states):layers[i].append(s['context'])
        extracted.append([torch.cat(parts) for parts in layers])
    y=torch.cat([-torch.ones(len(validation[0][0])),torch.ones(len(validation[1][0]))])
    target=torch.cat([torch.zeros(len(test[0][0])),torch.ones(len(test[1][0]))]).bool()
    results=[]
    for trainctx,testctx in zip(*extracted):
        def ridge(a,b):
            mean=a.mean(0);std=a.std(0).clamp_min(.01)
            a=(a-mean)/std;b=(b-mean)/std
            a=torch.cat([a,torch.ones(len(a),1)],1);b=torch.cat([b,torch.ones(len(b),1)],1)
            coeff=torch.linalg.solve(a.T@a+10*torch.eye(a.shape[1]),a.T@y)
            return b@coeff,a@coeff
        pred,_=ridge(trainctx.flatten(1),testctx.flatten(1))
        # Select single branch using validation-only leave-half-out scores.
        half=len(validation[0][0])//2
        indices=torch.cat([torch.arange(half),torch.arange(len(validation[0][0]),len(validation[0][0])+half)])
        other=torch.cat([torch.arange(half,len(validation[0][0])),torch.arange(len(validation[0][0])+half,len(y))])
        branch_scores=[]
        for branch in range(128):
            a=trainctx[:,branch];mean=a[indices].mean(0);std=a[indices].std(0).clamp_min(.01)
            a=(a-mean)/std;a=torch.cat([a,torch.ones(len(a),1)],1)
            coeff=torch.linalg.solve(a[indices].T@a[indices]+10*torch.eye(5),a[indices].T@y[indices])
            branch_scores.append(((a[other]@coeff>0)==(y[other]>0)).float().mean().item())
        selected=max(range(128),key=lambda b:branch_scores[b])
        single,_=ridge(trainctx[:,selected],testctx[:,selected])
        results.append({'combined_accuracy':((pred>0)==target).float().mean().item(),
                        'single_branch_accuracy':((single>0)==target).float().mean().item(),
                        'selected_branch':selected})
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--epochs',type=int,default=20)
    args=p.parse_args();torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':ARMS,
      'input':'eight-step observation history only; same AR1 regimes and saved seeds as preceding runs',
      'context':'each branch learns four-dimensional input projection; eight rounds of residual tanh updates',
      'controls':'same allocated AND trainable modules in all arms; independent feeds own context into all mixers; average feeds neighborhood average to all mixers; learned feeds self/left/right separately into learned mixers',
      'gate':'each branch reads its own updated context using learned receiver weights; sigmoid gate modulates activity',
      'training':'single Adam throughout sequential stream, 1024 windows/regime,20epochs each; fixed readout; no IDs/boundaryflags/replay/protection',
      'diagnostics':'ridge decode from concatenated branch contexts vs validation-selected individual branch, train on validation and score independent test; regime labels never enter network loss; decoding stable across checkpoints also measured',
      'selection':'constants fixed before test; no tuning; joint matches presentations',
      'limits':'combined diagnostic decoder is global and supervised by observer, not used by model; all branch seeds see full incoming layer input; no explicit consensus objective; parameter count matched but local-information access differs'},indent=2))
    records=[]
    for seed in map(int,args.seeds.split(',')):
        train=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        torch.manual_seed(seed);initial=Model();steps=args.epochs*16
        for arm in ARMS:
            model=copy.deepcopy(initial);model.arm=arm;joint=copy.deepcopy(model)
            states,validation=fit_stream(model,stream_batches(train,args.epochs,seed+100,[0,1]),[steps,steps*2],lambda m:measure(m,val))
            fit_stream(joint,stream_batches([(torch.cat([d[0] for d in train]),torch.cat([d[1] for d in train]))],args.epochs,seed+100,[0]),[],lambda m:None)
            row={'seed':seed,'arm':arm,'validation':validation,'test':[],'cue_decode':[]}
            for state in states:
                model.load_state_dict(state);row['test'].append(measure(model,test));row['cue_decode'].append(cue_decode(model,val,test))
            row['joint_test']=measure(joint,test);row['joint_cue_decode']=cue_decode(joint,val,test)
            torch.save({'stages':states,'joint':joint.state_dict()},out/f'{seed}_{arm}.pt')
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'old/new',*[round(t['mse'],3) for t in row['test'][-1]],'decode',[round(d['combined_accuracy'],3) for d in row['cue_decode'][-1]],flush=True)
    lines=['# Collective branch context','','Means across three paired seeds. MSE lower is better; decode chance 50%.','','| Exchange | Old just learned | Old final | New final | Joint avg | Exchange-off avg | Combined cue L1 / L2 | Single-branch cue L1 / L2 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        def avg(f):return sum(f(r) for r in rows)/len(rows)
        values=[avg(lambda r:r['test'][0][0]['mse']),avg(lambda r:r['test'][-1][0]['mse']),avg(lambda r:r['test'][-1][1]['mse']),avg(lambda r:sum(t['mse'] for t in r['joint_test'])/2),avg(lambda r:sum(t['exchange_off_mse'] for t in r['test'][-1])/2)]
        def pair(key):return ' / '.join(f'{avg(lambda r:r["cue_decode"][-1][i][key])*100:.1f}%' for i in (0,1))
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in values)+f' | {pair("combined_accuracy")} | {pair("single_branch_accuracy")} |')
    lines+=['','Combined decoder has 512 context coordinates; single branch has four. Greater decoding capacity is a diagnostic confound. Regime labels are observer-only. Baselines on same draws: zero ~1.000, oracle ~0.199, history heuristic ~0.295.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
