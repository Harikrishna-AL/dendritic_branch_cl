"""Fixed-readout circuit recruitment: explicit positive control and learned leaders."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.orchestrator_signal_pilot import make_data

ARMS=('dense','separated','raw_leader','context_leader')

class CircuitLayer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(32,4,inputs)/inputs**.5)
        self.bias=nn.Parameter(torch.zeros(32,4))
        self.compatibility=nn.Parameter(torch.randn(32,4,2)*.05)
        # A leader recruits 16 contiguous neurons through eight local hops.
        # Circular tie breaking keeps exactly 16 neurons and fixed circuit size.
        templates=torch.zeros(32,32)
        for leader in range(32):
            for offset in range(-7,9):templates[leader,(leader+offset)%32]=1
        self.register_buffer('templates',templates)
        group=(torch.arange(32)%4>=2).long()
        self.register_buffer('groups',group)

    def forward(self,x,cue,arm):
        a=F.relu(F.linear(x,self.weight.flatten(0,1),self.bias.flatten()).reshape(-1,32,4))
        if arm=='dense':
            mask=torch.ones(len(x),32,device=x.device);winner=a.flatten(1).argmax(-1)//4
        elif arm=='separated':
            mask=(cue.argmax(-1)[:,None]==self.groups[None]).float()
            winner=(a.mean(-1)*mask).argmax(-1)
        else:
            # Raw leader follows activation; context leader uses explicit cue affinity.
            branchscore=a if arm=='raw_leader' else torch.einsum('bc,ndc->bnd',cue,self.compatibility)
            score=branchscore.flatten(1)
            probability=F.softmax(score/.2,dim=-1)
            branchwinner=score.argmax(-1)
            hard=F.one_hot(branchwinner,128).float()
            # Straight-through: one actual leader, soft gradient to learn selection.
            route=(hard+probability-probability.detach()).reshape(-1,32,4).sum(-1)
            mask=route@self.templates
            winner=branchwinner//4
        return (a.mean(-1)*mask),mask,winner

class CircuitNet(nn.Module):
    def __init__(self,arm='dense'):
        super().__init__();self.arm=arm
        self.layers=nn.ModuleList([CircuitLayer(10),CircuitLayer(32)])
        # Each group has equal positive/negative readout coordinates.
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def encode(self,x):
        cue=x[:,-2:];states=[]
        for layer in self.layers:
            x,mask,winner=layer(x,cue,self.arm);states.append((mask,winner))
        return x,states
    def forward(self,x):return self.encode(x)[0]@self.readout

def fit(model,x,y,epochs,seed):
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ix in torch.randperm(len(x),generator=rng).split(64):
            opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x[ix]),y[ix])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()

@torch.no_grad()
def measure(model,data):
    result=[]
    for x,y in data:
        z,states=model.encode(x);pred=z@model.readout
        flipped=x.clone();flipped[:,-2:]=flipped[:,-2:].flip(1)
        result.append({'mse':F.mse_loss(pred,y).item(),
            'cue_delta':(pred-model(flipped)).abs().mean().item(),
            'layers':[{'mask_mean':m.mean(0).tolist(),
                       'winner_hist':(torch.bincount(w,minlength=32).float()/len(w)).tolist()} for m,w in states]})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--reverse',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    schedule=[1,0] if args.reverse else [0,1]
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':ARMS,'schedule':schedule,
        'task':'same noisy two-signal regression, explicit cue, fixed balanced readout',
        'data':'1024 train, 512 validation, 1024 test per context; seeds 1000/2000/3000 + run seed',
        'capacity':'two layers, 32 neurons x 4 branches; all arms have identical allocated parameters; dense uses 32 neurons, others exactly 16',
        'separated':'fixed non-overlapping 16-neuron groups per cue, with balanced fixed readout; masked rows receive zero gradients; no protection penalty',
        'leaders':'one winning branch per layer; recruits 16 contiguous neurons on ring, reachable in eight local hops; straight-through softmax gradient temperature .2; no separate message-amplitude pathway',
        'context_leader':'branch cue-affinity score, supplied same explicit cue at each layer; raw_leader uses activation score',
        'joint':'same total example presentations and optimizer updates; sequential Adam reset at boundary, joint no reset',
        'selection':'fixed settings before test; no hyperparameter search',
        'limitations':'new circuit recruitment mechanism, longer reach than prior pilot; explicit task cue; oracle separation control; hard masks with surrogate gradients; soft backward can update forward-inactive rows in learned arms; no pretraining/replay'},indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    records=[]
    for seed in map(int,args.seeds.split(',')):
        data=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        torch.manual_seed(seed);initial=CircuitNet()
        for arm in ARMS:
            model=copy.deepcopy(initial);model.arm=arm;joint=copy.deepcopy(model)
            states=[];validation=[]
            for task in schedule:
                fit(model,*data[task],args.epochs,seed+100+task)
                validation.append(measure(model,val));states.append(copy.deepcopy(model.state_dict()))
            fit(joint,torch.cat([d[0] for d in data]),torch.cat([d[1] for d in data]),args.epochs,seed+100)
            row={'seed':seed,'arm':arm,'validation':validation,'test':[]}
            for state in states:
                model.load_state_dict(state);row['test'].append(measure(model,test))
            row['joint_test']=measure(joint,test)
            torch.save({'stages':states,'joint':joint.state_dict()},out/f'{seed}_{arm}.pt')
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'old/new',*[round(row['test'][-1][t]['mse'],4) for t in schedule],flush=True)
    lines=['# Context circuit recruitment','','Mean ± sample standard deviation across seeds. Lower MSE is better.','','| Arm | Old MSE just learned | Old MSE final | New MSE final | Joint avg MSE | Circuit overlap L1 / L2 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        def cell(f):
            v=torch.tensor([f(r) for r in rows]);return f'{v.mean():.4f} ± {v.std().item() if len(v)>1 else 0:.4f}'
        overlaps=[]
        for i in range(2):
            vals=[]
            for r in rows:
                a,b=[torch.tensor(d['layers'][i]['mask_mean']) for d in r['test'][-1]]
                vals.append(float(torch.minimum(a,b).sum()/a.sum()))
            overlaps.append(sum(vals)/len(vals))
        lines.append(f'| {arm} | '+ ' | '.join([cell(lambda r:r['test'][0][schedule[0]]['mse']),cell(lambda r:r['test'][-1][schedule[0]]['mse']),cell(lambda r:r['test'][-1][schedule[1]]['mse']),cell(lambda r:sum(d['mse'] for d in r['joint_test'])/2)])+' | '+' / '.join(f'{o:.2f}' for o in overlaps)+' |')
    lines+=['','Overlap is overlap of mean neuron-selection profiles divided by selected population size. Zero means separated, one means fully shared. Dense has twice as many forward-active neurons as routed arms.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
