"""Full-moment shared contexts, plus validation-selected branch-memory improvements."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.deep_predictive_memory_pilot import Model as Base,Memory,train,evaluate
from bdh_cl.scripts.branch_short_memory_pilot import stream

class CenteredMemory(Memory):
    def __init__(self,width,streams=16):
        super().__init__(width,streams);self.register_buffer('mean',torch.zeros(streams,width))
    @torch.no_grad()
    def write(self,q,start=0):
        super().write(q,start);self.mean.mul_(.95).add_(q,alpha=.05)
    def reset(self):super().reset();self.mean.zero_()

class Moments(nn.Module):
    def __init__(self,d,decay=.95,streams=16):
        super().__init__();self.decay=decay
        self.register_buffer('previous',torch.zeros(streams,d))
        self.register_buffer('mean_previous',torch.zeros(streams,d));self.register_buffer('mean_current',torch.zeros(streams,d))
        self.register_buffer('cross',torch.zeros(streams,d,d));self.register_buffer('energy',torch.zeros(streams,d,d))
        self.register_buffer('seen',torch.tensor(False));self.register_buffer('mass',torch.tensor(0.))
    @torch.no_grad()
    def write(self,x):
        if self.seen:
            a=self.decay;b=1-a
            self.mass.mul_(a).add_(b)
            self.cross.mul_(a).add_(self.previous[:,:,None]*x[:,None,:],alpha=b)
            self.energy.mul_(a).add_(self.previous[:,:,None]*self.previous[:,None,:],alpha=b)
            self.mean_previous.mul_(a).add_(self.previous,alpha=b);self.mean_current.mul_(a).add_(x,alpha=b)
        self.previous.copy_(x);self.seen.fill_(True)
    def scores(self,weight,bias):
        if not self.seen:return self.previous.new_zeros(self.previous.shape[0],weight.shape[0])
        cross=torch.einsum('nd,bdk,nk->bn',weight,self.cross,weight)
        energy=torch.einsum('nd,bdk,nk->bn',weight,self.energy,weight)
        cross=cross+F.linear(self.mean_previous+self.mean_current,weight)*bias+bias.square()*self.mass
        energy=energy+2*F.linear(self.mean_previous,weight)*bias+bias.square()*self.mass
        return cross/(energy.clamp_min(0)+.01)
    def reset(self):
        for b in self.buffers():b.zero_()

class Model(Base):
    def __init__(self,variant='baseline',streams=16):
        super().__init__('branch_memory',streams);self.variant=variant
        if variant=='centered':self.memories=nn.ModuleList([CenteredMemory(32,streams),CenteredMemory(32,streams)])
        if variant in ('neuron_shared','layer_shared'):
            del self.memories
            # Shared contexts use full lag/variance matrices, not a weak scalar average.
            decays=[.95]*8 if variant=='neuron_shared' else [.8,.85,.9,.92,.95,.97,.98,.99]
            self.shared_first=nn.ModuleList([Moments(2,d,streams) for d in decays]);self.shared_second=Moments(8,.95,streams)
    def activation(self,x):
        if self.variant=='linear':return x
        if self.variant=='softsign':return F.softsign(x)
        return torch.tanh(x)
    def local_scores(self,q,layer,erase):
        if erase:return torch.zeros_like(q)
        m=self.memories[layer]
        if self.variant=='centered':return (m.cross-m.mean.square())/((m.energy-m.mean.square()).clamp_min(0)+.01)
        return m.cross/(m.energy+.01)
    def predict(self,x,erase=False):
        current=x.reshape(-1,8,2)[:,-1];q1=self.first(current)
        shared=self.variant in ('neuron_shared','layer_shared')
        if shared:
            scores=[m.scores(self.first.weight,self.first.bias) for m in self.shared_first]
            if self.variant=='neuron_shared':s1=torch.stack([s[:,4*i:4*i+4] for i,s in enumerate(scores)],1)
            else:s1=torch.stack(scores).mean(0).reshape(-1,8,4)
            if erase:s1=torch.zeros_like(s1)
        else:s1=self.local_scores(q1,0,erase).reshape(-1,8,4)
        winner1=s1.argmax(-1);mask=F.one_hot(winner1,4).float()
        hidden=self.activation((q1.reshape(-1,8,4)*mask).sum(-1));q2=self.second(hidden)
        s2=self.shared_second.scores(self.second.weight,self.second.bias) if shared else self.local_scores(q2,1,erase)
        if erase:s2=torch.zeros_like(s2)
        winner2=s2.argmax(-1);pred=q2.gather(1,winner2[:,None]).squeeze(1)
        return pred,(current.detach(),hidden.detach()) if shared else (q1.detach(),q2.detach()),(winner1,winner2)
    @torch.no_grad()
    def commit(self,q,erase=False):
        if erase:self.reset();return
        if self.variant in ('neuron_shared','layer_shared'):
            for m in self.shared_first:m.write(q[0])
            self.shared_second.write(q[1])
        else:
            for m,value in zip(self.memories,q):m.write(value)
    def reset(self):
        if self.variant in ('neuron_shared','layer_shared'):
            for m in self.shared_first:m.reset()
            self.shared_second.reset()
        else:
            for m in self.memories:m.reset()

VARIANTS=('baseline','centered','linear','softsign','neuron_shared','layer_shared')
def mean_error(phases):return statistics.mean(p['mse'] for p in phases)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2');p.add_argument('--block',type=int,default=4096)
    p.add_argument('--variants',default=','.join(VARIANTS));p.add_argument('--selected',default=None);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);variants=args.variants.split(',')
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),
        'architecture':'same384prediction parameters as deep pilot;shared controls analytically read full covariance/lag matrices using learned branch weights',
        'memory_budget':'265float values perstream cap;baseline192,centered256,shared265;shared controls have more effective state than baseline,not exact-equal budgets;actualstate reported',
        'sharing':'neuron_shared same input-moment context for each neuron branches;layer_shared all neurons read one multiscale context bank;full cross-terms avoid weak scalar pooling',
        'candidate_improvements':'centered autocorrelation;linear orsoftsign firstactivation;each separate factor;no addedparamcount',
        'selection':'development seeds0-2,twoorders;rankbranchvariants by meanvalidation MSE on seed+2000;test untouched untilselection;confirm freshseeds3-7 with selectedvariant andunchangedbaseline',
        'learning':'samecontinuous Adam .003;no taskIDs/boundaryreset/replay/protection;4096steps/regime',
        'limits':'input-moment shared controls differ in representation coordinate and analytic reconstruction,not only spatial memory location;lower/upperdepthactivation differs forlinear/softsign;budget cap matched ratherthanexacteffective count'},indent=2))
    records=[];models={}
    for seed in map(int,args.seeds.split(',')):
        for order in ((0,1),(1,0)):
            for variant in variants:
                torch.manual_seed(seed);model=Model(variant)
                snapshots=train(model,seed+1000,args.block,order)
                validation=evaluate(model,seed+2000)
                record={'seed':seed,'order':order,'variant':variant,'validation':validation,
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'float_state_per_stream':sum(b.numel() for b in model.buffers() if b.dtype==torch.float32)//16}
                key=f'{seed}_{order[0]}_{variant}';torch.save(snapshots,out/f'{key}.pt');models[key]=snapshots[-1]
                records.append(record);(out/'validation.json').write_text(json.dumps(records,indent=2))
                print(key,'validation',round(mean_error(validation),3),flush=True)
    branch=[v for v in variants if v not in ('neuron_shared','layer_shared')]
    ranking=sorted(branch,key=lambda v:statistics.mean(mean_error(r['validation']) for r in records if r['variant']==v))
    (out/'selection.json').write_text(json.dumps({'branch_ranking':ranking,'selected':args.selected or (ranking[0] if ranking else None)},indent=2))
    # Test only after the development selection is committed; never reselect on test.
    for r in records:
        torch.manual_seed(r['seed']);model=Model(r['variant']);key=f'{r["seed"]}_{r["order"][0]}_{r["variant"]}'
        model.load_state_dict(models[key]);r['test']=evaluate(model,r['seed']+3000)
        model.load_state_dict(models[key]);r['erased']=evaluate(model,r['seed']+3000,True)
    (out/'results.json').write_text(json.dumps(records,indent=2))
    lines=['# Shared-context controls and branch-memory improvements','','Settings selected on validation before test. Means over seeds and both orders.','','| Variant | Old MSE | New MSE | Avg | Erased avg | State floats/lane |','|---|---:|---:|---:|---:|---:|']
    for v in variants:
        rows=[r for r in records if r['variant']==v]
        old=statistics.mean(p['mse'] for r in rows for p in r['test'] if p['regime']==r['order'][0]);new=statistics.mean(p['mse'] for r in rows for p in r['test'] if p['regime']==r['order'][1])
        lines.append(f'| {v} | {old:.3f} | {new:.3f} | {statistics.mean(mean_error(r["test"]) for r in rows):.3f} | {statistics.mean(mean_error(r["erased"]) for r in rows):.3f} | {rows[0]["float_state_per_stream"]} |')
    lines+=['',f'Validation-selected branch variant: {args.selected or (ranking[0] if ranking else 'none: controls only')}. Shared controls use the same265float budget cap but different effective counts.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
