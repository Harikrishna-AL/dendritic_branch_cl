"""Grant-core pilot: persistent fading state in branches, causal continuous streams."""
import argparse,copy,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ARMS=('slow_only','slow_competition','branch_memory','branch_memory_competition','neuron_memory_competition')

class Model(nn.Module):
    def __init__(self,arm='slow_only',streams=16,decay=.95):
        super().__init__();self.arm=arm;self.decay=decay
        self.weight=nn.Parameter(torch.randn(32,4,16)/4)
        self.bias=nn.Parameter(torch.zeros(32,4))
        self.register_buffer('memory',torch.zeros(streams,32,4))
        self.register_buffer('readout',torch.tensor([1.,-1.]*16)/4)
    def predict(self,x,erase=False):
        a=F.relu(F.linear(x,self.weight.flatten(0,1),self.bias.flatten()).reshape(-1,32,4))
        state=torch.zeros_like(self.memory) if erase else self.memory.clone()
        if self.arm.startswith('slow'):score=a
        elif self.arm=='neuron_memory_competition':score=a+.5*state.mean(-1,keepdim=True)
        else:score=a+.5*state
        if 'competition' in self.arm:
            winner=score.argmax(-1)
            mask=F.one_hot(winner,4).float()
            h=(a*mask).sum(-1)
        else:
            winner=score.argmax(-1);mask=torch.ones_like(a)/4;h=a.mean(-1)
        if not self.arm.startswith('slow'):h=h*(1+.5*(state*mask).sum(-1))
        return h@self.readout,a.detach(),winner
    @torch.no_grad()
    def commit(self,activity,erase=False):
        if self.arm.startswith('slow') or erase:self.memory.zero_();return
        observation=torch.tanh(activity)
        if self.arm=='neuron_memory_competition':observation=observation.mean(-1,keepdim=True).expand_as(observation)
        self.memory.mul_(self.decay).add_(observation,alpha=1-self.decay)


def stream(seed,length,block,streams=16,order=(0,1,0,1)):
    rng=torch.Generator().manual_seed(seed)
    state=torch.randn(streams,2,generator=rng)
    history=[]
    # Burn-in under first environment, to supply causal recent observations.
    for _ in range(8):
        history.append(state+.05*torch.randn(streams,2,generator=rng))
        rho=torch.tensor([.9,.1] if order[0]==0 else [.1,.9])
        state=rho*state+torch.sqrt(1-rho.square())*torch.randn(streams,2,generator=rng)
    for t in range(length):
        task=order[min(t//block,len(order)-1)]
        rho=torch.tensor([.9,.1] if task==0 else [.1,.9])
        observation=state+.05*torch.randn(streams,2,generator=rng)
        history=history[1:]+[observation]
        nextstate=rho*state+torch.sqrt(1-rho.square())*torch.randn(streams,2,generator=rng)
        target=nextstate[:,task]+.05*torch.randn(streams,generator=rng)
        yield torch.stack(history,1).flatten(1),target,task
        state=nextstate

@torch.no_grad()
def evaluate(model,seed,block,erase=False,order=(0,1,0,1)):
    saved=model.memory.clone();model.memory.zero_()
    phases=[{'errors':[],'winners':[]} for _ in order]
    for t,(x,y,task) in enumerate(stream(seed,block*len(order),block,len(model.memory),order)):
        pred,a,w=model.predict(x,erase);model.commit(a,erase)
        phase=t//block;phases[phase]['errors'].append((pred-y).square().mean().item())
        phases[phase]['winners'].append(w.flatten().tolist())
    model.memory.copy_(saved)
    return [{'regime':order[i],'early_mse':sum(p['errors'][:32])/32,
             'late_mse':sum(p['errors'][32:])/len(p['errors'][32:]),
             'all_mse':sum(p['errors'])/block,
             'neuron_branch_usage':F.one_hot(torch.tensor(p['winners'][32:]).reshape(-1,32),4).float().mean(0).tolist(),
             'branch_usage':torch.bincount(torch.tensor(p['winners'][32:]).flatten(),minlength=4).float().div((block-32)*len(model.memory)*32).tolist()} for i,p in enumerate(phases)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--block',type=int,default=256);p.add_argument('--arms',default=','.join(ARMS));args=p.parse_args();arms=args.arms.split(',');torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records=[]
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':arms,
      'schedule':'continuous stationary AR1 streams; A then B then A then B; rho.9/.1 swapped; predict next slow-channel observation',
      'input':'eight recent observations only;16 independent stream lanes; neither task IDs nor boundary flags supplied to model/optimizer',
      'slow_learning':'one Adam lr.001 throughout; optimizer update every8 steps; branch state persists across both updates and switches, no gradients through writes',
      'fast_memory':'EMA of tanh branch activation,decay.95 per timestep;read previous state to select/modulate activity,then write current activity;no target in state',
      'controls':'same trainable weights; slow_only has no functional fast state; neuron state stores one value per neuron, branch state four; competition hard branch winner; allocated buffer matched but distinct effective state differs',
      'evaluation':'weights frozen after each training block; independent A/B/A/B stream plus reverse B/A/B/A;zero memory only at start of independent evaluation,never at regime boundaries;live versus erased each step',
      'limits':'competition changes pooling as well as sparsity; slow-only competition controls changed pooling;no tuning/pretraining/replay/protection;statistics exploratory'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        torch.manual_seed(seed);initial=Model()
        for arm in arms:
            model=copy.deepcopy(initial);model.arm=arm;opt=torch.optim.Adam(model.parameters(),lr=.001)
            losses=[];stages=[];online=[]
            for t,(x,y,task) in enumerate(stream(seed+1000,args.block*4,args.block)):
                pred,a,w=model.predict(x);loss=F.mse_loss(pred,y);losses.append(loss)
                online.append(float(loss.detach()));model.commit(a)
                if (t+1)%8==0:
                    opt.zero_grad(set_to_none=True);torch.stack(losses).mean().backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses=[]
                if (t+1)%args.block==0:
                    stage=(t+1)//args.block
                    stages.append({'stage':stage,'online_mse':sum(online[-args.block:])/args.block,
                        'live':evaluate(model,seed+3000,128),'erased':evaluate(model,seed+3000,128,True),
                        'reverse':evaluate(model,seed+4000,128,order=(1,0,1,0))})
                    torch.save(model.state_dict(),out/f'{seed}_{arm}_stage{stage}.pt')
            records.append({'seed':seed,'arm':arm,'stages':stages});(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'final live/erased',round(sum(v['late_mse'] for v in stages[-1]['live'])/4,3),round(sum(v['late_mse'] for v in stages[-1]['erased'])/4,3),flush=True)
    lines=['# Persistent branch short-term memory','','Means across three seeds. Frozen final weights; late MSE excludes first32 observations after each switch.','','| Arm | A late MSE | B late MSE | Live avg | Erased avg | Reverse avg |','|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        rows=[r['stages'][-1] for r in records if r['arm']==arm]
        def avg(f):return sum(f(r) for r in rows)/len(rows)
        a=avg(lambda r:(r['live'][0]['late_mse']+r['live'][2]['late_mse'])/2)
        b=avg(lambda r:(r['live'][1]['late_mse']+r['live'][3]['late_mse'])/2)
        values=[a,b,avg(lambda r:sum(v['late_mse'] for v in r['live'])/4),avg(lambda r:sum(v['late_mse'] for v in r['erased'])/4),avg(lambda r:sum(v['late_mse'] for v in r['reverse'])/4)]
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
