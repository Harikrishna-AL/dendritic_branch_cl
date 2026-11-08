"""Local three-factor coded teaching, branch usefulness, and local cue synapses.

No backpropagation, covariance matrix, prototype/example/configuration archive.
Cue synapses are branch-resident accumulated coactivity statistics, which are
mathematically associative input memories; they are not memory-free.
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS


class LocalCoded:
    def __init__(self,seed,arm,rate,tau,neurons=24,cue_dim=196):
        self.arm=arm;self.rate=rate;self.tau=tau
        self.n=neurons if arm.startswith('branch') else (4*neurons if arm=='flat' else 119)
        self.k=4 if arm.startswith('branch') else 1
        self.units=self.n*self.k
        generator=torch.Generator().manual_seed(seed)
        self.weight=torch.randn(785,self.units,generator=generator)*.02
        self.codes=torch.empty(10,self.n)
        for n in range(self.n):
            order=torch.randperm(10,generator=generator)
            self.codes[:,n]=-.8;self.codes[order[:5],n]=.8
        self.error=torch.full((self.n,self.k),.64)
        self.usage=torch.zeros(self.units)
        self.cue_dim=cue_dim
        self.cue_sum=torch.zeros(self.units,cue_dim)
        self.cue_count=torch.zeros(self.units)
        self.presentations=0;self.rho=.95;self.active_count=neurons

    def data_features(self,x):return torch.cat([F.normalize(x,dim=1),torch.ones(len(x),1)],1)
    def cues(self,x):
        pooled=x if self.cue_dim==784 else x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196)
        return (pooled>pooled.mean(1,keepdim=True)).float()

    def selection(self,b):
        if self.arm=='branch_activity':j=b.abs().mean(0).argmax(-1)
        elif self.k==4:j=self.error.argmin(-1)
        elif self.arm=='flat':return self.error.flatten().topk(self.active_count,largest=False).indices
        else:return torch.arange(self.units)
        return torch.arange(self.n)*self.k+j

    @torch.no_grad()
    def learn(self,x,y):
        z=self.data_features(x);b=torch.tanh(z@self.weight).reshape(-1,self.n,self.k)
        active=self.selection(b)
        desired=self.codes[y]
        targets=desired[:,:,None].expand_as(b).flatten(1)
        response=b.flatten(1)
        # Apical teaching signal desired-current; presynaptic input and local
        # tanh derivative form eligibility. Active-branch gate is the third factor.
        residual=targets[:,active]-response[:,active]
        eligibility=residual*(1-response[:,active].square())
        rate=self.rate/(1+self.usage[active]/self.tau) if math.isfinite(self.tau) else torch.full_like(self.usage[active],self.rate)
        self.weight[:,active]+=((z.T@eligibility)/len(x))*rate[None]
        reliable=(residual.abs()<.2).float()
        self.usage[active]+=reliable.sum(0)
        if self.arm!='plain':
            self.cue_sum[active]+=reliable.T@self.cues(x)
            self.cue_count[active]+=reliable.sum(0)
        # Inactive branches receive the SAME neuron-local teaching signal;
        # no alternative full-network/classifier prediction is made.
        alpha=len(x)/(self.presentations+len(x)) if self.rho=='cumulative' else 1-self.rho
        self.error.mul_(1-alpha).add_((b-desired[:,:,None]).square().mean(0),alpha=alpha)
        self.presentations+=len(x)

    @torch.no_grad()
    def predict(self,x,rule='cosine',teacher=False,oracle=None):
        b=torch.tanh(self.data_features(x)@self.weight)
        if self.arm=='plain':hidden=b;return (hidden@self.codes.T).argmax(1)
        if oracle is not None:
            if self.k==4:active=torch.arange(self.n)[None]*self.k+oracle
            else:active=oracle
        elif teacher:
            active=self.selection(b.reshape(-1,self.n,self.k))[None].expand(len(x),-1)
        else:
            q=self.cues(x)
            if rule=='cosine':scores=F.normalize(q,dim=1)@F.normalize(self.cue_sum,dim=1).T
            else:
                p=(self.cue_sum+1)/(self.cue_count[:,None]+2)
                p=p.clamp(1e-6,1-1e-6)
                scores=q@(p/(1-p)).log().T+(1-p).log().sum(1)[None]
                scores+=((self.cue_count+1)/(self.cue_count.sum()+self.units)).log()[None]
            scores=scores.masked_fill(self.cue_count[None]==0,-1e6)
            if self.k==4:active=torch.arange(self.n)[None]*self.k+scores.reshape(-1,self.n,self.k).argmax(-1)
            else:active=scores.topk(self.active_count,dim=1).indices
        chosen=b.gather(1,active)
        neuron=active//self.k
        class_code=self.codes.T[neuron]
        return (chosen[:,:,None]*class_code).sum(1).argmax(1)

    def bytes(self):
        data=[self.weight,self.codes,self.error,self.usage]
        if self.arm!='plain':data+=[self.cue_sum,self.cue_count]
        return sum(t.numel()*t.element_size() for t in data)


@torch.no_grad()
def evaluate(model,x,y,pairs,rule='cosine',teacher=False,history=None):
    pred=[]
    for i in range(0,len(y),512):
        oracle=None
        if history is not None:
            ids=torch.zeros(len(y[i:i+512]),dtype=torch.long)
            for j,pair in enumerate(pairs):ids[(y[i:i+512]==pair[0])|(y[i:i+512]==pair[1])]=j
            oracle=torch.stack(history)[ids]
            if model.k==4:oracle=oracle%4
        pred.append(model.predict(x[i:i+512],rule,teacher,oracle))
    pred=torch.cat(pred)
    return {'accuracy':float((pred==y).float().mean()),'pairs':[float((pred[(y==a)|(y==b)]==y[(y==a)|(y==b)]).float().mean()) for a,b in pairs]}


def train(seed,arm,rate,tau,x,y,mask,pairs):
    model=LocalCoded(seed,arm,rate,tau);history=[];start=time.perf_counter()
    for stage,pair in enumerate(pairs):
        ix=torch.nonzero(mask&((y==pair[0])|(y==pair[1])),as_tuple=True)[0]
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            order=ix[torch.randperm(len(ix),generator=rng)]
            for ids in order.split(128):model.learn(x[ids],y[ids])
        # Diagnostic only; no future model operation reads this history.
        history.append(model.selection(torch.zeros(1,model.n,model.k)).clone())
    return model,history,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--datasets',default='mnist,fashion_mnist');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    arms=['branch','branch_activity','flat','plain']
    protocol={'datasets':args.datasets.split(','),'dev_seeds':[190,200],'confirmation_seeds':[[191,192,193],[201,202,203]],
              'arms':arms,'rates':[.1,1.,10.],'usage_timescales':[1000.,10000.,'infinity'],'cue_rules':['cosine','bernoulli'],
              'training':'Two current-pair epochs, normalized784pixel inputs. Local SGD updates selected branches only; no optimizer/backprop. Apical desired response is fixed balanced random class code from supervised TRAINING label. Feedback is per-neuron desired-current response, NOT a scalar reward and NOT inferred biological circuitry.',
              'short_term':'EMA.95 of neuron-local branch squared response error selects one branch per neuron. Branch-activity control selects largest mean absolute response without error memory. Flat control chooses24of96independent units by same error memory. Plain control has144ordinary units with same update rule and larger first layer.',
              'long_term':'Data synapses and branch-local cue coactivity sums/counts. Cue weights are mathematically associative mean/input statistics, not a memory-free mechanism. Binary pooled pixels coactivate reliable selected branches (abs response error<.2). No example/prototype bank, whole-config archive, global covariance matrix or stored task ID. No label enters inference.',
              'protection':'Successful branch usage damps future learning rate as eta/(1+usage/tau). No boundary consolidation/reset. tau infinity is unprotected control. Fixed coded output eliminates drifting learned classifier but changes the original architecture; this must be disclosed.',
              'matching':'Branch and flat have96data units and local cue synapses, differing only grouped vs global competition and coded output tying. Plain119units matches total ongoing numeric storage within0.2%; exact bytes reported. No compute-match claim. Fixed codes count in memory. No central random cue expansion.',
              'selection':'Fresh stratified50k/10k training split; rate/tau/cue rule independentlyselected per arm on validation. Confirm fresh3seeds both orders only if some branch developmentaccuracy>=60%. All negative candidates retained; not officialtest tuning.',
              'scope':'Local-learning feasibility/mechanism pilot; no novelty/biological/conference-ready claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text());selections={};rows=[]
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['dev_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;candidates=[];selected={}
        for arm in arms:
            for rate in protocol['rates']:
                for tau in [1000.,10000.,math.inf]:
                    model,_,seconds=train(seed,arm,rate,tau,x,y,mask,list(SPLITS))
                    for rule in (['cosine'] if arm=='plain' else protocol['cue_rules']):
                        result=evaluate(model,x[val],y[val],list(SPLITS),rule)
                        candidates.append({'arm':arm,'rate':rate,'tau':tau if math.isfinite(tau) else 'infinity','rule':rule,'accuracy':result['accuracy'],'numeric_bytes':model.bytes(),'seconds':seconds})
            selected[arm]=max([r for r in candidates if r['arm']==arm],key=lambda r:r['accuracy'])
            print('dev',dataset,arm,round(selected[arm]['accuracy']*100,2),selected[arm],flush=True)
        selections[dataset]={'selected':selected,'candidates':candidates};(out/'selection.json').write_text(json.dumps(selections,indent=2))
        if selected['branch']['accuracy']<.6:
            print('gate fails',dataset,'branch<60%; no officialtest selection or confirmation',flush=True);continue
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                for arm in arms:
                    setting=selected[arm];tau=math.inf if setting['tau']=='infinity' else setting['tau']
                    model,history,seconds=train(seed,arm,setting['rate'],tau,x,y,torch.ones(len(y),dtype=torch.bool),pairs)
                    result=evaluate(model,tx,ty,pairs,setting['rule'])
                    teacher=evaluate(model,tx,ty,pairs,setting['rule'],teacher=True)
                    oracle=evaluate(model,tx,ty,pairs,setting['rule'],history=history) if arm!='branch_activity' else {'accuracy':None}
                    rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':arm,**result,'teacher_accuracy':teacher['accuracy'],'oracle_accuracy':oracle['accuracy'],'old_accuracy':st.mean(result['pairs'][:-1]),'seconds':seconds,'numeric_bytes':model.bytes()})
                    print(dataset,order,seed,arm,round(result['accuracy']*100,2),'oracle',None if oracle['accuracy'] is None else round(oracle['accuracy']*100,2),flush=True)
                    (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Local coded branch-learning pilot','','No covariance, global gradient backpropagation, example/configuration archive or task ID. Per-neuron class-code teaching signals remain explicit supervised feedback. Fixed coded classifier and branch-resident associative coactivity memory are changes to the original model.','', '| Dataset | Architecture | Best training-validation accuracy | Officialtest confirmation |','|---|---|---:|---:|']
    for dataset,selection in selections.items():
        for arm in arms:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            test=f'{100*st.mean(r["accuracy"] for r in rs):.2f}%' if rs else 'Not run: branch-development gate failed'
            lines.append(f'| {dataset} | {arm} | {100*selection["selected"][arm]["accuracy"]:.2f}% | {test} |')
    lines+=['','All validation candidates and exact numeric memory/compute are saved. Branch vs activity control tests recent-error routing; branch vs flat tests grouped competition. Controls are independently tuned. These are feasibility experiments, not evidence for publication readiness. Cues are local persistent input associations, not absence of memory.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
