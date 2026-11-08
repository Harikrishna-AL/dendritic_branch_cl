"""Forecast usefulness vs activation predictability, with delayed factual feedback."""
import argparse,collections,copy,json,statistics,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.electricity_memory_pilot import Forecast,features,summarize,prepare

CANDIDATES=('predict_winner','predict_all','quality_winner','quality_all','quality_soft','uniform_all')
class Useful(Forecast):
    def __init__(self,variant,streams=16):
        super().__init__('branch_tanh',streams);self.variant=variant
        if not variant.startswith('predict'):self.context=nn.ModuleList([self.context[0]])
        if variant.startswith('quality'):self.register_buffer('error',torch.ones(streams,32))
    @property
    def feedback(self):return self.variant.startswith('quality')
    def reset(self):
        super().reset()
        if hasattr(self,'error'):self.error.fill_(1)
    def forecast(self,x):
        q1=self.first(x);m=self.context[0];w1=(m.cross/(m.energy+.01)).reshape(-1,8,4).argmax(-1)
        hidden=torch.tanh(q1.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1));q2=self.second(hidden)
        if self.variant.startswith('predict'):
            m2=self.context[1];w2=(m2.cross/(m2.energy+.01)).argmax(-1);weights=F.one_hot(w2,32).float()
        elif self.variant=='uniform_all':weights=torch.full_like(q2,1/32)
        elif self.variant=='quality_soft':weights=torch.softmax(-self.error/.1,dim=-1)
        else:weights=F.one_hot(self.error.argmin(-1),32).float()
        pred=(q2*weights).sum(-1)
        with torch.no_grad():
            self.context[0].write(q1.detach())
            if self.variant.startswith('predict'):self.context[1].write(q2.detach())
        return pred,(x.detach().clone(),w1.detach().clone(),weights.detach().clone(),q2.detach().clone())
    @torch.no_grad()
    def observe(self,ticket,target):
        if self.feedback:self.error.mul_(.95).add_((ticket[3]-target[:,None]).square(),alpha=.05)
    def ticket_heads(self,ticket):
        x,w1=ticket[:2];q1=self.first(x);hidden=torch.tanh(q1.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1));return self.second(hidden)
    def ticket_prediction(self,ticket):return (self.ticket_heads(ticket)*ticket[2]).sum(-1)
    def training_loss(self,ticket,target):
        if self.variant in ('predict_all','quality_all','quality_soft','uniform_all'):
            q=self.ticket_heads(ticket);return F.mse_loss(q,target[:,None].expand_as(q))
        return F.mse_loss(self.ticket_prediction(ticket),target)

def cohort(offset):
    if offset==0:return prepare('data/electricity/electricity.zip','data/electricity/hourly16.pt')
    cache=Path(f'data/electricity/hourly16_offset{offset}.pt')
    if cache.exists():return torch.load(cache,weights_only=False)
    with zipfile.ZipFile('data/electricity/electricity.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('LD2011_2014.txt') and not n.startswith('__MACOSX'))
        with z.open(name) as f:df=pd.read_csv(f,sep=';',decimal=',',usecols=list(range(65)))
    v=df.iloc[:,1:].astype('float32');v.index=pd.to_datetime(df.iloc[:,0]);h=v.resample('h',closed='right',label='right').mean().loc['2012-01-01':'2014-12-31 23:00:00']
    c=h.iloc[:720];eligible=[name for name in h if (c[name]>0).mean()>.8 and c[name].std()>.01];names=eligible[offset:offset+16];assert len(names)==16
    a=h[names].to_numpy(copy=True)
    for t in np.flatnonzero((h.to_numpy()==0).all(1)):
        if t:a[t]=a[t-1]
    mean=a[:720].mean(0);std=a[:720].std(0)
    d={'y':torch.tensor((a-mean)/std,dtype=torch.float32),'year':h.index.year.to_numpy(),'month':h.index.month.to_numpy(),'clients':names,'mean':mean.tolist(),'std':std.tolist(),'selection':f'eligible positions{offset}:{offset+16} selected only from first720h2012; no future screening'}
    assert torch.isfinite(d['y']).all();torch.save(d,cache);return d

def run(model,opt,d,start,end,updates,score=True):
    y=d['y'];years=d['year'];months=d['month'];queue=collections.deque();batch=[];rows=[]
    feedback=getattr(model,'feedback',False)
    for t in range(start,end):
        while queue and queue[0][0]<=t:
            when,ticket,base=queue.popleft();target=y[when]-base
            if feedback:model.observe(ticket,target)
            if updates:batch.append((ticket,target))
        if updates and len(batch)>=8:
            opt.zero_grad(set_to_none=True)
            losses=[model.training_loss(ticket,target) if hasattr(model,'training_loss') else F.mse_loss(model.ticket_prediction(ticket),target) for ticket,target in batch]
            torch.stack(losses).mean().backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();batch=[]
        with torch.no_grad():residual,ticket=model.forecast(features(y,t));prediction=y[t]+residual
        if t+24>=len(y):continue
        if updates or feedback:queue.append((t+24,ticket,y[t].clone()))
        if score and years[t+24]==years[t]:
            e=(prediction-y[t+24]).square();rows.append({'year':int(years[t+24]),'month':int(months[t+24]),'mse':e.mean().item(),'mae':(prediction-y[t+24]).abs().mean().item(),'per_client_squared_error':e.tolist()})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--offset',type=int,default=0);p.add_argument('--seeds',default='0,1,2');p.add_argument('--variants',default=','.join(CANDIDATES));p.add_argument('--confirm',action='store_true');p.add_argument('--selected');args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);d=cohort(args.offset)
    if args.offset:
        original=cohort(0);assert not set(original['clients'])&set(d['clients'])
    trainend=int(np.flatnonzero(d['year']==2013)[0]);teststart=int(np.flatnonzero(d['year']==2014)[0]);end=len(d['y'])-24 if args.confirm else teststart
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'clients':d['clients'],
      'selection':'rank branch candidates on original cohort2013 online MSE across seeds0-2; commit selected before accessing new cohort2014; confirm locked setting with seeds3-7 and eligible clients16:32',
      'isolation':'same1120slowparameters and first-layer predictability competition; change only output cue and/or all-output training; original selected-head training versus mean per-head auxiliary loss; uniform/predict_all control dense-head training',
      'fast_feedback':'EMA decay.95 of squared factual forecast error per output branch, updated only24h after forecast; saved contemporaneous predictions not recomputed errors; no taskIDs/boundaries; target is observed future electricity, not external tasklabel',
      'information':'quality uses delayed observed-load forecast errors; it is not the earlier target-free activation-only rule; frozen means frozen slowweights, quality still gets causal feedback',
      'memory':'predict192float/client;quality128;uniform96; quality ticket adds32 stored counterfactual outputs/client for24h; full operating memory not exactly matched; delayqueue/batch storage included in limitations',
      'soft':'softmax(-error/.1) weighted ensemble, hardmin otherwise; no feedback/adaptation hyperparameter search',
      'learning':'Adam.001 every8 matured tickets, clip5; no resets across2013/2014;2012prefix calibration720h; all customer selection uses calibration only',
      'limits':'this only tests outputbranch usefulness, hiddenbranch scoring still temporal; allheadtraining changes learning coverage; no retentionclaim; originalcohort2014 never evaluated here; newclientsnotanindependent dataset'},indent=2))
    records=[]
    for seed in map(int,args.seeds.split(',')):
        for variant in args.variants.split(','):
            torch.manual_seed(seed);model=Forecast(variant) if variant in ('mlp','gru') else Useful(variant);model.reset();opt=torch.optim.Adam(model.parameters(),lr=.001)
            run(model,opt,d,720,trainend,True,False)
            state=copy.deepcopy(model.state_dict());opstate=copy.deepcopy(opt.state_dict());recent=copy.deepcopy(model.recent)
            r={'seed':seed,'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'runtime_buffer_bytes':sum(b.numel()*b.element_size() for b in model.buffers())}
            torch.save({'model':state,'optimizer':opstate,'recent':recent},out/f'{seed}_{variant}_2012.pt')
            for mode,updates in [('frozen',False),('online',True)]:
                model.load_state_dict(state);model.recent=copy.deepcopy(recent);opt.load_state_dict(opstate)
                r[mode]=summarize(run(model,opt,d,trainend,end,updates))
            records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2))
            year='2014' if args.confirm else '2013';print(seed,variant,year,round(r['online'][year]['mse'],4),flush=True)
    if not args.confirm:
        ranks=sorted(args.variants.split(','),key=lambda v:statistics.mean(r['online']['2013']['mse'] for r in records if r['variant']==v))
        (out/'selection.json').write_text(json.dumps({'ranking':ranks,'selected':ranks[0],'metric':'originalcohort2013 online MSE; frozen not selection metric'},indent=2))
    lines=['# Delayed usefulness memory','','Means over initialization seeds; no future targets enter predictions. Frozen slow weights still permit causal error-memory feedback.','','| Variant | Parameters | 2013 frozen | 2013 online'+(' | 2014 frozen | 2014 online' if args.confirm else '')+' |','|---|---:|---:|---:'+('|---:|---:' if args.confirm else '')+'|']
    for v in args.variants.split(','):
        rs=[r for r in records if r['variant']==v];vals=[statistics.mean(r[mode][year]['mse'] for r in rs) for year in (('2013','2014') if args.confirm else ('2013',)) for mode in ('frozen','online')]
        lines.append(f'| {v} | {rs[0]["parameters"]} | '+' | '.join(f'{a:.4f}' for a in vals)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
