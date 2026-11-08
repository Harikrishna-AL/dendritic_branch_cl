"""Controlled recurring regimes: direct retention and target-delay quality memory."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.electricity_usefulness_pilot import Useful
from bdh_cl.scripts.deep_predictive_memory_pilot import Model as Deep,Memory
from bdh_cl.scripts.shared_context_memory_pilot import Moments
from bdh_cl.scripts.branch_short_memory_pilot import stream

VARIANTS=('predict_winner','quality_soft','quality_all','uniform_all','history','mlp','gru')
class Branch(Useful):
    def __init__(self,variant):
        nn.Module.__init__(self);self.arm='branch_tanh';self.variant=variant
        self.first=nn.Linear(2,32);self.second=nn.Linear(8,32)
        self.context=nn.ModuleList([Memory(32)] + ([Memory(32)] if variant.startswith('predict') else []))
        if variant.startswith('quality'):self.register_buffer('error',torch.ones(16,32))
    def forecast(self,x):return super().forecast(x.reshape(-1,8,2)[:,-1])

class History(Deep):
    def __init__(self):super().__init__('history_only')
    def forecast(self,x):
        pred,q,w=self.predict(x);self.commit(q);return pred,(x.detach().clone(),w[0].clone(),w[1].clone())
    def ticket_prediction(self,ticket):
        x,w1,w2=ticket;q=self.first(x.reshape(-1,8,2)[:,-1]);h=torch.tanh(q.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1))
        return self.second(h).gather(1,w2[:,None]).squeeze(1)

class Shared(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(2,32);self.second=nn.Linear(8,32)
        self.context=nn.ModuleList([Moments(2),Moments(8)])
    def reset(self):
        for m in self.context:m.reset()
    def forecast(self,x):
        x=x.reshape(-1,8,2)[:,-1];q=self.first(x)
        w1=self.context[0].scores(self.first.weight,self.first.bias).reshape(-1,8,4).argmax(-1)
        h=torch.tanh(q.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1));out=self.second(h)
        w2=self.context[1].scores(self.second.weight,self.second.bias).argmax(-1)
        self.context[0].write(x);self.context[1].write(h.detach())
        return out.gather(1,w2[:,None]).squeeze(1),(x.detach().clone(),w1.clone(),w2.clone())
    def ticket_prediction(self,ticket):
        x,w1,w2=ticket;q=self.first(x);h=torch.tanh(q.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1))
        return self.second(h).gather(1,w2[:,None]).squeeze(1)

class Window(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        if variant=='mlp':self.net=nn.Sequential(nn.Linear(16,22),nn.Tanh(),nn.Linear(22,1))
        else:self.rnn=nn.GRU(2,9,batch_first=True);self.head=nn.Linear(9,1)
    def reset(self):pass
    def ticket_prediction(self,ticket):
        x=ticket[0]
        if self.variant=='mlp':return self.net(x).squeeze(-1)
        h,_=self.rnn(x.reshape(-1,8,2));return self.head(h[:,-1]).squeeze(-1)
    def forecast(self,x):return self.ticket_prediction((x,)),(x.detach().clone(),)

def make(v):
    return Shared() if v=='shared' else History() if v=='history' else Window(v) if v in ('mlp','gru') else Branch(v)

def fit(m,opt,seed,order,block,batch=None,pending=None,capture=False):
    batch=[] if batch is None else copy.deepcopy(batch);snap=None
    for t,(x,y,_) in enumerate(stream(seed,block*len(order),block,order=order)):
        if pending is not None:
            ticket,target=pending
            if getattr(m,'feedback',False):m.observe(ticket,target)
            batch.append((ticket,target))
        if len(batch)>=8:
            opt.zero_grad(set_to_none=True)
            losses=[m.training_loss(ticket,target) if hasattr(m,'training_loss') else F.mse_loss(m.ticket_prediction(ticket),target) for ticket,target in batch]
            torch.stack(losses).mean().backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();batch=[]
        with torch.no_grad():_,ticket=m.forecast(x)
        pending=(ticket,y.detach().clone())
        if capture and t+1==block:snap={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'batch':copy.deepcopy(batch),'pending':copy.deepcopy(pending)}
    return snap

@torch.no_grad()
def audit(model,seed,order,carry=False):
    m=copy.deepcopy(model)
    if not carry:m.reset()
    weights=[p.clone() for p in m.parameters()];pending=None;phases=[[] for _ in order]
    for t,(x,y,_) in enumerate(stream(seed,256*len(order),256,order=order)):
        if pending is not None and getattr(m,'feedback',False):m.observe(*pending)
        pred,ticket=m.forecast(x);phases[t//256].append((pred-y).square().mean().item());pending=(ticket,y.clone())
    assert all(torch.equal(a,b) for a,b in zip(weights,m.parameters()))
    return [{'regime':r,'all_mse':statistics.mean(errors),'first32_mse':statistics.mean(errors[:32]),'late_mse':statistics.mean(errors[32:])} for r,errors in zip(order,phases)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='8,9,10');p.add_argument('--block',type=int,default=4096);p.add_argument('--variants',default=','.join(VARIANTS));args=p.parse_args();variants=args.variants.split(',')
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records=[]
    (out/'source.py').write_text(Path(__file__).read_text());(out/'protocol.json').write_text(json.dumps({**vars(args),'variants':variants,
      'task':'two continuously evolving AR1channels rho.9/.1 swapped acrossregimes; predict nextslowchannel;16lanes; samecausal8stepwindow available; no taskIDs/boundaries',
      'learning':'oneAdam.003 every8 matured targets,clip5;4096/regime; delayone timestep; quality gets supervisedenvironmentforecast target only whenobserved; no resetof optimizer/faststate/batch/pending atswitch',
      'feedback':'sameoutput usefulness variants as electricity; all-output training versus winner; target-freepredictability baseline; quality cue receives factual labels, notoracleIDs; supervisedfeedback potentiallytask-informative',
      'controls':'history384params,MLP397,windowGRU361; GRU fullBPTT through8observedsteps rather than one-steptruncation; branch384; runtimebudgets differ',
      'retention':'sameheldout A/B/A/B and reverse afterA/afterB/aftermoreA; weights frozen, memory/causalfeedbackallowed; no slowrelearning. Whole-stream andlate excluding32 both reported.',
      'moreA':'same additional4096 updates on newA observations, continuing optimizer,batch,pending,faststate fromafterA',
      'limits':'engineered observablepredictabilitytask,notreal-world electricity; natural electricityaudit foundbackwardtransfer; no hyperselection ornewclaimofnovelty'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        for first in (0,1):
            for v in variants:
                torch.manual_seed(seed);m=make(v);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.003)
                snap=fit(m,opt,seed+1000,(first,1-first),args.block,capture=True)
                afterB=copy.deepcopy(m.state_dict());r={'seed':seed,'first':first,'variant':v,'parameters':sum(p.numel() for p in m.parameters())}
                order=(first,1-first)*2
                r['after_B']=audit(m,seed+5000,order);r['return_carry']=audit(m,seed+5000,(first,),True)
                m.load_state_dict(snap['model']);r['after_A']=audit(m,seed+5000,order)
                opt.load_state_dict(snap['optimizer']);fit(m,opt,seed+9000,(first,),args.block,snap['batch'],snap['pending']);r['after_moreA']=audit(m,seed+5000,order)
                def old(phases,key):return statistics.mean(p[key] for p in phases if p['regime']==first)
                def new(phases,key):return statistics.mean(p[key] for p in phases if p['regime']!=first)
                for key in ('all_mse','late_mse'):
                    r['forgetting_'+key]=old(r['after_B'],key)-old(r['after_A'],key)
                    r['new_gain_'+key]=new(r['after_A'],key)-new(r['after_B'],key)
                records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2))
                print(seed,first,v,'forget',round(r['forgetting_all_mse'],3),'newgain',round(r['new_gain_all_mse'],3),flush=True)
    lines=['# Controlled-regime retention','','Three new seeds, both training orders. Positive forgetting means worse old-task MSE after new-task learning. All audit weights frozen; fast feedback causal.','','| Model | Old before | Old after | Forgetting | New after | New gain | Late forgetting |','|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        rs=[r for r in records if r['variant']==v]
        def val(stage,old=True):return statistics.mean(p['all_mse'] for r in rs for p in r[stage] if (p['regime']==r['first'])==old)
        vals=[val('after_A'),val('after_B'),statistics.mean(r['forgetting_all_mse'] for r in rs),val('after_B',False),statistics.mean(r['new_gain_all_mse'] for r in rs),statistics.mean(r['forgetting_late_mse'] for r in rs)]
        lines.append('| '+v+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
