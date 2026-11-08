"""Small real-data transfer pilot with delayed-label chronological forecasting."""
import argparse,copy,json,zipfile,hashlib,collections,statistics,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.deep_predictive_memory_pilot import Memory
from bdh_cl.scripts.shared_context_memory_pilot import Moments

ARMS=('branch_tanh','branch_linear','shared','history','mlp','gru','linear')

class Forecast(nn.Module):
    def __init__(self,arm,streams=16):
        super().__init__();self.arm=arm
        if arm in ('branch_tanh','branch_linear','shared','history'):
            self.first=nn.Linear(25,32);self.second=nn.Linear(8,32)
            if arm=='shared':self.context=nn.ModuleList([Moments(25,streams=streams),Moments(8,streams=streams)])
            else:self.context=nn.ModuleList([Memory(32,streams),Memory(32,streams)])
        elif arm=='mlp':self.net=nn.Sequential(nn.Linear(25,40),nn.Tanh(),nn.Linear(40,1))
        elif arm=='linear':self.net=nn.Linear(25,1)
        elif arm=='gru':
            self.cell=nn.GRUCell(25,10);self.head=nn.Linear(10,1);self.register_buffer('hidden',torch.zeros(streams,10))
    def act(self,q):return q if self.arm=='branch_linear' else torch.tanh(q)
    def routed(self,x,w1,w2):
        q1=self.first(x);hidden=self.act(q1.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1));q2=self.second(hidden)
        return q2.gather(1,w2[:,None]).squeeze(1),q1,hidden,q2
    def forecast(self,x):
        # Returns a detached training ticket; no graph is retained through delayed updates.
        if self.arm=='gru':
            prior=self.hidden.clone();h=self.cell(x,prior);pred=self.head(h).squeeze(-1)
            with torch.no_grad():self.hidden.copy_(h.detach())
            return pred,(x.detach().clone(),prior)
        if self.arm in ('mlp','linear'):return self.net(x).squeeze(-1),(x.detach().clone(),)
        q1=self.first(x)
        if self.arm=='shared':s1=self.context[0].scores(self.first.weight,self.first.bias)
        elif self.arm=='history':
            # Reconstruct branch response sequence from causal recent windows stored externally.
            h=self.recent+[x];historical=self.first(torch.stack(h[-8:],1))
            s1=(historical[:,:-1]*historical[:,1:]).sum(1)/(historical[:,:-1].square().sum(1)+.01)
        else:s1=self.context[0].cross/(self.context[0].energy+.01)
        w1=s1.reshape(-1,8,4).argmax(-1)
        hidden=self.act(q1.reshape(-1,8,4).gather(2,w1[:,:,None]).squeeze(-1));q2=self.second(hidden)
        if self.arm=='shared':s2=self.context[1].scores(self.second.weight,self.second.bias)
        elif self.arm=='history':
            hh=self.act(historical.reshape(-1,historical.shape[1],8,4).gather(3,w1[:,None,:,None].expand(-1,historical.shape[1],-1,1)).squeeze(-1))
            qs=self.second(hh);s2=(qs[:,:-1]*qs[:,1:]).sum(1)/(qs[:,:-1].square().sum(1)+.01)
        else:s2=self.context[1].cross/(self.context[1].energy+.01)
        w2=s2.argmax(-1);pred=q2.gather(1,w2[:,None]).squeeze(1)
        with torch.no_grad():
            if self.arm=='shared':self.context[0].write(x);self.context[1].write(hidden.detach())
            elif self.arm=='history':self.recent=(self.recent+[x.detach().clone()])[-7:]
            else:self.context[0].write(q1.detach());self.context[1].write(q2.detach())
        return pred,(x.detach().clone(),w1.detach().clone(),w2.detach().clone())
    def ticket_prediction(self,ticket):
        x=ticket[0]
        if self.arm=='gru':return self.head(self.cell(x,ticket[1])).squeeze(-1)
        if self.arm in ('linear','mlp'):return self.net(x).squeeze(-1)
        return self.routed(x,ticket[1],ticket[2])[0]
    def reset(self):
        if self.arm=='gru':self.hidden.zero_()
        if hasattr(self,'context'):
            for c in self.context:c.reset()
        self.recent=[]

def features(y,t):
    # Observed24-hour window plus last week's matching hour for the future target.
    return torch.cat((y[t-23:t+1].T,y[t-144,:,None]),1)

def prepare(archive,cache):
    cache=Path(cache)
    if cache.exists():return torch.load(cache,weights_only=False)
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('LD2011_2014.txt') and not n.startswith('__MACOSX'))
        with z.open(name) as f:df=pd.read_csv(f,sep=';',decimal=',',usecols=list(range(65)))
    stamp=pd.to_datetime(df.iloc[:,0]);values=df.iloc[:,1:].astype('float32');values.index=stamp
    hourly=values.resample('h',closed='right',label='right').mean()
    hourly=hourly.loc['2012-01-01':'2014-12-31 23:00:00']
    calibration=hourly.iloc[:720];eligible=[c for c in hourly.columns if (calibration[c]>0).mean()>.8 and calibration[c].std()>.01]
    selected=eligible[:16];assert len(selected)==16,'not enough active training-prefix clients'
    a=hourly[selected].to_numpy(copy=True);assert np.isfinite(a).all()
    # All-client zero hours are documented clock-change artifacts. Impute from past only.
    artifact=(hourly.to_numpy()==0).all(1)
    for t in np.flatnonzero(artifact):
        if t:a[t]=a[t-1]
    mean=a[:720].mean(0);std=a[:720].std(0);assert (std>0).all()
    y=torch.tensor((a-mean)/std,dtype=torch.float32)
    payload={'y':y,'year':hourly.index.year.to_numpy(),'month':hourly.index.month.to_numpy(),
             'clients':selected,'mean':mean.tolist(),'std':std.tolist(),'artifact_hours':int(artifact.sum()),
             'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
             'source':'https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014',
             'selection':'first16 of first64 clients active>80% in first720h2012; train-prefix-only normalization; right-closed hourly means'}
    torch.save(payload,cache);return payload

def run(model,opt,y,years,months,start,end,updates,score=True):
    queue=collections.deque();batch=[];rows=[];nupdates=0
    for t in range(start,end):
        # At time t, delayed targets for forecasts made t-24 are now observed.
        while queue and queue[0][0]<=t:
            maturity,ticket,base=queue.popleft()
            if updates:batch.append((ticket,y[maturity]-base))
        if updates and len(batch)>=8:
            opt.zero_grad(set_to_none=True)
            loss=torch.stack([F.mse_loss(model.ticket_prediction(ticket),target) for ticket,target in batch]).mean()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();batch=[];nupdates+=1
        x=features(y,t)
        with torch.no_grad():residual,ticket=model.forecast(x);prediction=y[t]+residual
        if t+24>=len(y):continue
        if updates:queue.append((t+24,ticket,y[t].clone()))
        if score and years[t+24]==years[t]:
            errors=(prediction-y[t+24]).square()
            rows.append({'target_time':t+24,'year':int(years[t+24]),'month':int(months[t+24]),
                         'mse':errors.mean().item(),'mae':(prediction-y[t+24]).abs().mean().item(),
                         'per_client_squared_error':errors.tolist()})
    # Pending tickets and <8 mature examples are dropped at independent replay starts.
    # No training target past end is used. Optimizer and runtime state persist across years in online run.
    return rows,nupdates

def summarize(rows):
    return {str(year):{'mse':statistics.mean(r['mse'] for r in rows if r['year']==year),
            'mae':statistics.mean(r['mae'] for r in rows if r['year']==year),
            'monthly_mse':{str(m):statistics.mean(r['mse'] for r in rows if r['year']==year and r['month']==m) for m in range(1,13)},
            'per_client_mse':np.mean([r['per_client_squared_error'] for r in rows if r['year']==year],axis=0).tolist(),
            'forecasts':sum(r['year']==year for r in rows)} for year in sorted({r['year'] for r in rows})}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2');p.add_argument('--arms',default=','.join(ARMS));args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=prepare('data/electricity/electricity.zip','data/electricity/hourly16.pt');y=data['y'];years=data['year'];months=data['month']
    end2012=int(np.flatnonzero(years==2013)[0]);test=int(np.flatnonzero(years==2014)[0]);end=len(y)-24
    protocol={**vars(args),'data':{k:v for k,v in data.items() if k not in ('y','year','month')},
      'horizon':'24hours; same25 observed values for all learned models; predict residual to daily persistence y[t]',
      'split':'first720h2012 calibration only; remaining2012 delayed-label training once in chronological order; 2013validation,2014test',
      'modes':'frozen continuous2013/2014 with no slowupdates; online continuous2013/2014 with labels delayed24h; score prediction before target; no year resets',
      'optimizer':'singleAdam .001 every8 matured hourly tickets; clipping5; no hyperparameter search; no replay except necessary24h label-delay queue',
      'baseline':'daily y[t], weekly y[t-144] for targett+24; ridge fit only2012; MLP1081,GRU1121,branch1120parameters; GRU hidden detached eachstep,25inputwindow,one-step truncatedgradients',
      'memory':'branch192floats/client; shared fullmoments1477floats/client plus2mass scalars; GRU10/client; history7x25/client; budgets not equal',
      'causality':'tickets cache observedfeatures and contemporaneous routes/GRUstate; recompute maturedloss with currentweights and savedrouting; no future-label memorywrites',
      'limits':'16clients chosenfromfirst64; state tracked per knownseries,not inferredcustomerID; real driftforecasting doesnot establish taskseparation or reducedforgetting; no tuned state-of-art claim; clock artifacts imputed from past; monthly scoring isanalysis only'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    baselines={}
    X=torch.stack([features(y,t) for t in range(720,end2012-24)]).reshape(-1,25).double();target=torch.stack([y[t+24]-y[t] for t in range(720,end2012-24)]).flatten().double()
    design=torch.cat((X,torch.ones(len(X),1)),1);ridge=torch.linalg.solve(design.T@design+torch.diag(torch.tensor([1.]*25+[0.],dtype=torch.float64)),design.T@target)
    for label in ('daily','weekly','ridge'):
        rs=[]
        for t in range(end2012,end):
            if years[t+24]!=years[t]:continue
            prediction=y[t] if label=='daily' else y[t-144] if label=='weekly' else y[t]+(features(y,t).double()@ridge[:25]+ridge[-1]).float()
            errors=(prediction-y[t+24]).square();rs.append({'year':int(years[t+24]),'month':int(months[t+24]),'mse':errors.mean().item(),'mae':(prediction-y[t+24]).abs().mean().item(),'per_client_squared_error':errors.tolist()})
        baselines[label]=summarize(rs)
    (out/'baselines.json').write_text(json.dumps(baselines,indent=2));records=[]
    for seed in map(int,args.seeds.split(',')):
        for arm in args.arms.split(','):
            torch.manual_seed(seed);model=Forecast(arm);model.reset();opt=torch.optim.Adam(model.parameters(),lr=.001)
            _,updates=run(model,opt,y,years,months,720,end2012,True,False)
            state=copy.deepcopy(model.state_dict());optimizer=copy.deepcopy(opt.state_dict());recent=copy.deepcopy(model.recent)
            row={'seed':seed,'arm':arm,'parameters':sum(p.numel() for p in model.parameters()),'state_bytes':sum(b.numel()*b.element_size() for b in model.buffers()),'train_updates':updates}
            torch.save({'model':state,'optimizer':optimizer,'recent':recent},out/f'{seed}_{arm}_2012.pt')
            for mode,update in (('frozen',False),('online',True)):
                model.load_state_dict(state);model.recent=copy.deepcopy(recent);opt.load_state_dict(optimizer)
                rs,n=run(model,opt,y,years,months,end2012,end,update);row[mode]=summarize(rs);row[mode+'_updates']=n
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,arm,'2014 frozen/online',round(row['frozen']['2014']['mse'],3),round(row['online']['2014']['mse'],3),flush=True)
    lines=['# Electricity: first real-data transfer pilot','','Standardized per-client error;16 clients;24h horizon; means across seeds. All forecasts are counted. Runtime memory and optimizer persist across years.','','| Model | Parameters | 2013 frozen MSE | 2014 frozen MSE | 2013 online MSE | 2014 online MSE |','|---|---:|---:|---:|---:|---:|']
    for label,r in baselines.items():lines.append(f'| {label} | {26 if label=="ridge" else 0} | {r["2013"]["mse"]:.3f} | {r["2014"]["mse"]:.3f} | {r["2013"]["mse"]:.3f} | {r["2014"]["mse"]:.3f} |')
    for arm in args.arms.split(','):
        rs=[r for r in records if r['arm']==arm];vals=[statistics.mean(r[mode][year]['mse'] for r in rs) for mode in ('frozen','online') for year in ('2013','2014')]
        lines.append(f'| {arm} | {rs[0]["parameters"]} | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','Frozen rows test retained forecasting weights with runtime memory; online rows additionally allow slow learning on past labels. Online2013 is an adaptation prefix before2014, not a held-out static validation score. No settings were selected using either2013 or2014 outcomes. Shared full-moment memory receives much more state; GRU uses detached hidden state and the same25-feature observed window.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
