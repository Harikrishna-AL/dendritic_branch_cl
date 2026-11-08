"""Random sensory mixtures: branch-local predictive memory without assigned channels."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.branch_short_memory_pilot import stream
ARMS=('average','window_competition','memory_competition')
class Model(nn.Module):
    def __init__(self,arm='average',streams=16):
        super().__init__();self.arm=arm
        self.weight=nn.Parameter(torch.randn(32,2)*.3);self.bias=nn.Parameter(torch.zeros(32))
        self.register_buffer('previous',torch.zeros(streams,32));self.register_buffer('cross',torch.zeros(streams,32));self.register_buffer('energy',torch.zeros(streams,32))
        self.register_buffer('seen',torch.tensor(False))
    def predict(self,x,erase=False):
        h=x.reshape(-1,8,2);q=F.linear(h[:,-1],self.weight,self.bias)
        if self.arm=='average':mask=torch.ones_like(q)/32
        else:
            if self.arm=='window_competition':
                historical=F.linear(h,self.weight,self.bias)
                score=(historical[:,:-1]*historical[:,1:]).sum(1)/(historical[:,:-1].square().sum(1)+.01)
            else:score=torch.zeros_like(q) if erase else self.cross/(self.energy+.01)
            mask=F.one_hot(score.argmax(-1),32).float()
        return (q*mask).sum(-1),q.detach(),mask.argmax(-1)
    @torch.no_grad()
    def commit(self,q,erase=False):
        if erase:self.reset();return
        if self.seen:
            self.cross.mul_(.95).add_(self.previous*q,alpha=.05)
            self.energy.mul_(.95).add_(self.previous.square(),alpha=.05)
        self.previous.copy_(q);self.seen.fill_(True)
    def reset(self):self.previous.zero_();self.cross.zero_();self.energy.zero_();self.seen.fill_(False)

def train(model,seed,block,order):
    opt=torch.optim.Adam(model.parameters(),lr=.01);losses=[];states=[]
    for t,(x,y,_) in enumerate(stream(seed,block*len(order),block,16,order)):
        pred,q,_=model.predict(x);losses.append(F.mse_loss(pred,y));model.commit(q)
        if (t+1)%8==0:
            opt.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();opt.step();losses=[]
        if (t+1)%block==0:states.append(copy.deepcopy(model.state_dict()))
    return states
@torch.no_grad()
def evaluate(model,seed,erase=False,order=(0,1,0,1)):
    model.reset();errors=[[] for _ in order];routes=[[] for _ in order]
    for t,(x,y,_) in enumerate(stream(seed,256*len(order),256,16,order)):
        pred,q,w=model.predict(x,erase);model.commit(q,erase)
        errors[t//256].append((pred-y).square().mean().item());routes[t//256].append(w.tolist())
    return [{'regime':order[i],'mse':statistics.mean(e[32:]),'early_mse':statistics.mean(e[:32]),'usage':(torch.bincount(torch.tensor(routes[i][32:]).flatten(),minlength=32).float()/(224*16)).tolist()} for i,e in enumerate(errors)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2');args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records=[]
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':ARMS,
        'structure':'32 scalar branches with random learned mixtures of both channels; no sensory channel assigned to a branch;96trainable parameters in every arm',
        'rule':'previous branch outputs, covariance and energy traces decay.95; hard winner chooses largest normalized lagcorrelation; state read before target-free write; ordinary Adam regression only winner receives direct output gradient',
        'input':'eight observation steps;no task IDs; branchprediction uses latest observations;window control uses all8 for routing;memory observes longer stream',
        'schedule':'A4096 thenB4096; continuous optimizer lr.01 every8steps;singleA4096/singleB4096/joint4096 controls',
        'evaluation':'frozenweights A/B/A/B and reverse;memory reset onlyatindependentstart;erasedeachstepcontrol',
        'criteria':'old and new MSE<.30;better thanwindow by20%;replicate fiveheldoutseeds;report allnegative outcomes',
        'limits':'linear branches and hard engineered predictability competition;task specifically asks forpredictablechannel;no dendrite-specific superiority or noveltyclaim;no taskIDs but role ofpredictability supplied by design'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        torch.manual_seed(seed);initial=Model()
        for arm in ARMS:
            row={'seed':seed,'arm':arm,'controls':{}}
            for label,block,order in [('singleA',4096,(0,)),('singleB',4096,(1,)),('joint',128,(0,1)*16),('sequential',4096,(0,1))]:
                model=copy.deepcopy(initial);model.arm=arm;states=train(model,seed+1000,block,order)
                if label=='sequential':
                    row['stage1']=evaluate(model,seed+3000) # overwritten below with stage1weights
                    model.load_state_dict(states[0]);row['stage1']=evaluate(model,seed+3000)
                    model.load_state_dict(states[-1]);row['live']=evaluate(model,seed+3000)
                    model.load_state_dict(states[-1]);row['erased']=evaluate(model,seed+3000,True)
                    model.load_state_dict(states[-1]);row['reverse']=evaluate(model,seed+4000,order=(1,0,1,0))
                    row['weights']=model.weight.detach().tolist();torch.save(states,out/f'{seed}_{arm}.pt')
                else:row['controls'][label]=evaluate(model,seed+5000,order=order if label!='joint' else (0,1,0,1))
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'live/erased',round(statistics.mean(t['mse'] for t in row['live']),3),round(statistics.mean(t['mse'] for t in row['erased']),3),flush=True)
    lines=['# Learned input mixtures with predictive branch memory','','Mean across seeds. Lower MSE better. No assigned sensory branches.','','| Arm | OldA | NewB | Joint | Erased avg | A/B branch overlap |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        def avg(f):return statistics.mean(f(r) for r in rows)
        def overlap(r):
            a=(torch.tensor(r['live'][0]['usage'])+torch.tensor(r['live'][2]['usage']))/2;b=(torch.tensor(r['live'][1]['usage'])+torch.tensor(r['live'][3]['usage']))/2
            return torch.minimum(a,b).sum().item()
        values=[avg(lambda r:statistics.mean(t['mse'] for t in r['live'] if t['regime']==i)) for i in (0,1)]
        values+=[avg(lambda r:statistics.mean(t['mse'] for t in r['controls']['joint'])),avg(lambda r:statistics.mean(t['mse'] for t in r['erased']))]
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in values)+f' | {avg(overlap):.3f} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
