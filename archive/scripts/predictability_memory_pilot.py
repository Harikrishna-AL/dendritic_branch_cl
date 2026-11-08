"""Proof of mechanism, not emergent discovery: fading branch predictability traces."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.branch_short_memory_pilot import stream
ARMS=('average','instant_competition','window_competition','branch_predictability','central_matched_memory','fixed_branch')

class Model(nn.Module):
    def __init__(self,arm='average',streams=16):
        super().__init__();self.arm=arm
        self.slope=nn.Parameter(torch.zeros(2));self.bias=nn.Parameter(torch.zeros(2))
        self.register_buffer('cross',torch.zeros(streams,2));self.register_buffer('energy',torch.zeros(streams,2))
    def predict(self,x,erase=False):
        history=x.reshape(-1,8,2);current=history[:,-1]
        branch=current*self.slope+self.bias
        if self.arm=='average':mask=torch.full_like(branch,.5)
        elif self.arm=='fixed_branch':mask=torch.zeros_like(branch);mask[:,0]=1
        else:
            if self.arm=='window_competition':
                score=(history[:,:-1]*history[:,1:]).sum(1)/(history[:,:-1].square().sum(1)+.01)
            elif self.arm=='instant_competition':
                score=(history[:,-2]*current)/(history[:,-2].square()+.1)
            else:score=torch.zeros_like(branch) if erase else self.cross/(self.energy+.01)
            mask=F.one_hot(score.argmax(-1),2).float()
        return (branch*mask).sum(-1),mask
    @torch.no_grad()
    def commit(self,x,erase=False):
        if erase:self.cross.zero_();self.energy.zero_();return
        h=x.reshape(-1,8,2);previous,current=h[:,-2],h[:,-1]
        self.cross.mul_(.95).add_(previous*current,alpha=.05)
        self.energy.mul_(.95).add_(previous.square(),alpha=.05)
    def reset(self):self.cross.zero_();self.energy.zero_()


def train(model,seed,block,order):
    opt=torch.optim.Adam(model.parameters(),lr=.01);losses=[];snapshots=[]
    for t,(x,y,_) in enumerate(stream(seed,block*len(order),block,16,order)):
        pred,_=model.predict(x);losses.append(F.mse_loss(pred,y));model.commit(x)
        if (t+1)%8==0:
            opt.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();opt.step();losses=[]
        if (t+1)%block==0:snapshots.append(copy.deepcopy(model.state_dict()))
    return snapshots

@torch.no_grad()
def evaluate(model,seed,erase=False,order=(0,1,0,1),block=256):
    model.reset();rows=[{'error':[],'correct':[],'mask':[]} for _ in order]
    for t,(x,y,task) in enumerate(stream(seed,block*len(order),block,16,order)):
        pred,mask=model.predict(x,erase);model.commit(x,erase)
        row=rows[t//block];row['error'].append((pred-y).square().mean().item())
        row['correct'].append((mask.argmax(-1)==task).float().mean().item());row['mask'].append(mask.mean(0).tolist())
    return [{'regime':order[i],'early_mse':statistics.mean(r['error'][:32]),'late_mse':statistics.mean(r['error'][32:]),
             'route_accuracy':statistics.mean(r['correct'][32:]),'usage':torch.tensor(r['mask'][32:]).mean(0).tolist()} for i,r in enumerate(rows)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2');p.add_argument('--train-block',type=int,default=1024);p.add_argument('--arms',default=','.join(ARMS));args=p.parse_args();arms=args.arms.split(',')
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':arms,
      'hypothesis':'fading temporal-statistic state+competition can select stable predictors without task ID',
      'structure':'two input-channel branches, each one learned slope/bias; sensory partition predetermined, not learned task separation',
      'memory':'EMA lag-one covariance and input energy per branch; decay.95; previous state read, current observed pair written afterward; no future target in state',
      'selection':'largest normalized lag correlation; hand-designed criterion aligned to predicting slower process, not automatic arbitrary-task inference',
      'central_control':'same independent covariance/energy traces housed in a central router; functionally identical positive-memory implementation, disclosed equivalence',
      'slow_learning':'one Adam lr.01,update every8 timesteps,no reset at boundaries',
      'controls':{'single_steps':4096,'joint_steps':4096,'sequential_steps_per_regime':args.train_block},
      'evaluation':'frozen weights,independent recurringA/B/A/B and reverse;memory resets only at independent evaluation start;erased eachstep intervention',
      'success_criteria':'late oldA and newB MSE<.30;>=20% reduction versusaverage andinstant baseline;erasure degrades;confirm five independent seeds and reverseevaluation;no test tuning',
      'limits':'proof-of-mechanism only;architectural channel separation given;memory criterion handengineered;does not show dendrites outperform state-matched central inference;no novelty claim'},indent=2))
    records=[]
    for seed in map(int,args.seeds.split(',')):
        initial=Model()
        for arm in arms:
            initial.arm=arm
            row={'seed':seed,'arm':arm,'learnability':{}}
            for label,block,order in [('singleA',4096,(0,)),('singleB',4096,(1,)),('joint',128,(0,1)*16),('sequential',args.train_block,(0,1))]:
                model=copy.deepcopy(initial);states=train(model,seed+1000,block,order)
                if label=='sequential':
                    row['stages']=[]
                    for state in states:
                        model.load_state_dict(state);row['stages'].append(evaluate(model,seed+3000))
                    model.load_state_dict(states[-1]);row['live']=evaluate(model,seed+3000)
                    model.load_state_dict(states[-1]);row['erased']=evaluate(model,seed+3000,True)
                    model.load_state_dict(states[-1]);row['reverse']=evaluate(model,seed+4000,order=(1,0,1,0))
                    torch.save(states,out/f'{seed}_{arm}.pt')
                else:
                    row['learnability'][label]=evaluate(model,seed+5000,order=order if label!='joint' else (0,1,0,1))
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'live/erased',round(statistics.mean(r['late_mse'] for r in row['live']),3),round(statistics.mean(r['late_mse'] for r in row['erased']),3),flush=True)
    lines=['# Predictability-memory mechanism','','Means across paired seeds. Predetermined channel branches; no task IDs. Central matched memory is mathematically identical and intentionally controls specificity.','','| Arm | Old A MSE | New B MSE | Joint MSE | Live avg | Erased avg | Route accuracy |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        rs=[r for r in records if r['arm']==arm]
        def avg(f):return statistics.mean(f(r) for r in rs)
        a=avg(lambda r:statistics.mean(t['late_mse'] for t in r['live'] if t['regime']==0))
        b=avg(lambda r:statistics.mean(t['late_mse'] for t in r['live'] if t['regime']==1))
        values=[a,b,avg(lambda r:statistics.mean(t['late_mse'] for t in r['learnability']['joint'])),avg(lambda r:statistics.mean(t['late_mse'] for t in r['live'])),avg(lambda r:statistics.mean(t['late_mse'] for t in r['erased']))]
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in values)+f' | {avg(lambda r:statistics.mean(t["route_accuracy"] for t in r["live"]))*100:.1f}% |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
