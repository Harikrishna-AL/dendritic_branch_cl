"""Concurrent local class-density feedback and informative-flat-unit control.

Both density and branch updates are acquired from the SAME current stream.
No covariance, replay, task identity, archived configuration or inference label.
"""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.local_coded_branch_pilot import LocalCoded
from bdh_cl.scripts.local_coded_feedback import ClassAssociation
from bdh_cl.scripts.local_cue_resolution import cache


class VarianceFlat(LocalCoded):
    def __init__(self,seed,setting):
        super().__init__(seed,'flat',setting['rate'],setting['tau'],setting['neurons'],setting['cue_dim'])
        self.active_count=setting['active_count'];self.rho=0.
        self.mean=torch.zeros(self.units);self.second=torch.full((self.units,),.64)
        self.penalty=setting['variance_penalty']

    def selection(self,b):
        variance=(self.second-self.mean.square()).clamp_min(0)
        return (self.error.flatten()-self.penalty*variance).topk(self.active_count,largest=False).indices

    def learn(self,x,y):
        super().learn(x,y)
        self.mean.copy_(self.codes[y].mean(0));self.second.copy_(self.codes[y].square().mean(0))

    def bytes(self):return super().bytes()+self.mean.numel()*8


def train(seed,arm,setting,x,y,mask,pairs):
    if arm=='variance_flat':model=VarianceFlat(seed,setting)
    else:
        model=LocalCoded(seed,arm,setting['rate'],setting['tau'],setting['neurons'],setting['cue_dim'])
        model.rho=0.;model.active_count=setting.get('active_count',setting['neurons'])
    density=ClassAssociation(784,'bernoulli');informative=[]
    for stage,pair in enumerate(pairs):
        ids=torch.nonzero(mask&((y==pair[0])|(y==pair[1])),as_tuple=True)[0]
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for batch in ids[torch.randperm(len(ids),generator=rng)].split(128):
                bx=x[batch];by=y[batch]
                b=torch.tanh(model.data_features(bx)@model.weight).reshape(-1,model.n,model.k)
                active=model.selection(b);unitcodes=model.codes.repeat_interleave(model.k,dim=1)
                # Analysis only: how much class-discriminating code is recruited?
                informative.append(float((unitcodes[pair[0],active]!=unitcodes[pair[1],active]).float().mean()))
                model.learn(bx,by);density.learn(bx,by)
    return model,density,st.mean(informative)


@torch.no_grad()
def density_scores(model,x):
    q=model.features(x);p=((model.sums+1)/(model.counts[:,None]+2)).clamp(1e-6,1-1e-6)
    return q@(p/(1-p)).log().T+(1-p).log().sum(1)[None]+((model.counts+1)/(model.counts.sum()+10)).log()[None]


@torch.no_grad()
def scores(model,response,cue,beta):
    codes=model.codes.repeat_interleave(model.k,dim=1)
    if beta=='independent':
        if model.k==4:active=torch.arange(model.n)[None]*4+cue.reshape(-1,model.n,4).argmax(-1)
        else:active=cue.topk(model.active_count,dim=1).indices
        return -(response.gather(1,active)[:,None]-codes.T[active].transpose(1,2)).square().sum(-1)
    value=-(response[:,None]-codes[None]).square()+beta*cue[:,None]
    if model.k==4:score=value.reshape(-1,10,model.n,4).max(-1).values.sum(-1)
    else:score=value.topk(model.active_count,dim=-1).values.sum(-1)
    return score-score.max(-1,keepdim=True).values


@torch.no_grad()
def evaluate(model,cached,density,y,beta,alpha):
    response,cue=cached;pred=[]
    for i in range(0,len(y),512):
        if alpha=='direct':combined=density[i:i+512]
        else:combined=scores(model,response[i:i+512],cue[i:i+512],beta)+alpha*density[i:i+512]
        pred.append(combined.argmax(-1))
    pred=torch.cat(pred)
    return float((pred==y).float().mean())


@torch.no_grad()
def sweep(model,cached,density,y,betas,alphas):
    # Every alpha shares exactly the same expensive branch energy. Cache it
    # once per beta; predictions/selection are identical to repeated evaluate.
    response,cue=cached;results=[]
    for beta in betas:
        energy=torch.cat([scores(model,response[i:i+512],cue[i:i+512],beta) for i in range(0,len(y),512)])
        for alpha in alphas:
            combined=density if alpha=='direct' else energy+alpha*density
            results.append((beta,alpha,float((combined.argmax(-1)==y).float().mean())))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'datasets':['mnist','fashion_mnist'],'dev_seeds':[250,260],'confirmation_seeds':[[251,252,253],[261,262,263]],
              'branch_settings':'Inherited from localcue-resolution VALIDATION:64neurons x4branches, eta1, tau10000, rho0, cue196 MNIST/784Fashion. No architecture choice from officialtest results.',
              'class_feedback':'Local raw784pixel Bernoulli class-coactivity synapses accumulated concurrently. Class hypotheses plus branch response compatibility jointly rank all10classes. Additional31,400bytes, no covariance. This is classifier-cue/energy fusion, which is explicitly a separate mechanism; no claim branch-only result.',
              'flat_control':'Inherited same256data units and contextdimension; independently select activecount64/128/256, rates.1/1/10, tau1000/10000 and variancepenalty0/.1/1/10 using previous-batch teaching-signal variance. Low-error competition may otherwise prefer constant/uninformative code bits. This is also local and boundary-free.',
              'inference_grid':'beta0/.001/.01/.1/1/independent; densityweight0/.01/.1/1/10/direct-only. Select on fresh training validation. Direct-only is included so branch reliance is not forced. No labels or pair masks enter inference.',
              'stream':'Two current-pair epochs, no pasttask revisit. Branch and density see same CURRENT sample once each update. Oracle configurations not used. Fresh3seeds x2orders confirmation per dataset. No conference-ready/novel inference claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text());selections={};rows=[]
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['dev_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False
        base={'neurons':64,'cue_dim':196 if dataset=='mnist' else 784,'rate':1.,'tau':10000.,'active_count':64}
        betas=[0.,.001,.01,.1,1.,'independent'];alphas=[0.,.01,.1,1.,10.,'direct']
        candidates=[]
        model,density,informative=train(seed,'branch',base,x,y,mask,list(SPLITS));cached=cache(model,x[val]);ds=density_scores(density,x[val]);ds-=ds.max(-1,keepdim=True).values
        for beta,alpha,a in sweep(model,cached,ds,y[val],betas,alphas):
            candidates.append({'arm':'branch',**base,'beta':beta,'alpha':alpha,'accuracy':a,'informative_fraction':informative})
        branch=max(candidates,key=lambda r:r['accuracy']);print('dev',dataset,'branch',round(branch['accuracy']*100,2),flush=True)
        # Train and decode flat independently, including same posterior fusion.
        flat_train_candidates=[]
        for count in [64,128,256]:
            for penalty in [0.,.1,1.,10.]:
                for rate in [.1,1.,10.]:
                    for tau in [1000.,10000.]:
                        setting={**base,'active_count':count,'variance_penalty':penalty,'rate':rate,'tau':tau}
                        model,density,informative=train(seed,'variance_flat',setting,x,y,mask,list(SPLITS));cached=cache(model,x[val]);ds=density_scores(density,x[val]);ds-=ds.max(-1,keepdim=True).values
                        for beta,alpha,a in sweep(model,cached,ds,y[val],betas,alphas):
                            candidates.append({'arm':'variance_flat',**setting,'beta':beta,'alpha':alpha,'accuracy':a,'informative_fraction':informative})
                        flat_train_candidates.append(max([r for r in candidates if r['arm']=='variance_flat' and r['active_count']==count and r['variance_penalty']==penalty and r['rate']==rate and r['tau']==tau],key=lambda r:r['accuracy']))
                (out/'development_progress.json').write_text(json.dumps({'dataset':dataset,'candidates':candidates},indent=2))
                print('dev',dataset,'flat',count,penalty,'best',round(max(r['accuracy'] for r in flat_train_candidates)*100,2),flush=True)
        flat=max([r for r in candidates if r['arm']=='variance_flat'],key=lambda r:r['accuracy'])
        selected={'branch':branch,'variance_flat':flat};selections[dataset]={'selected':selected,'candidates':candidates}
        (out/'selection.json').write_text(json.dumps(selections,indent=2));print('selected',dataset,selected,flush=True)
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                for arm in ['branch','variance_flat']:
                    setting=selected[arm];model,density,informative=train(seed,arm,setting,x,y,torch.ones(len(y),dtype=torch.bool),pairs)
                    cached=cache(model,tx);ds=density_scores(density,tx);ds-=ds.max(-1,keepdim=True).values
                    for name,alpha in [(arm,setting['alpha']),(arm+'_no_density',0.)]:
                        a=evaluate(model,cached,ds,ty,setting['beta'],alpha)
                        rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':name,'accuracy':a,'informative_fraction':informative,'numeric_bytes':model.bytes()+31400})
                        print(dataset,order,seed,name,round(a*100,2),flush=True)
                    if arm=='branch':rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'direct','accuracy':evaluate(model,cached,ds,ty,0.,'direct'),'numeric_bytes':31400})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Local class feedback and informative-competition control','','Class density and branch memory acquired concurrently without covariance/replay/taskIDs. Density adds31,400bytes; this is explicitly feedback/energy fusion, not a branch-only result. Flat competition is independently strengthened with local teaching-signal variance.','', '| Dataset | Architecture | Fresh test mean |','|---|---|---:|']
    for dataset in protocol['datasets']:
        for arm in ['branch','branch_no_density','variance_flat','variance_flat_no_density','direct']:
            a=[100*r['accuracy'] for r in rows if r['dataset']==dataset and r['arm']==arm]
            lines.append(f'| {dataset} | {arm} | {st.mean(a):.2f}% |')
    lines+=['','Three independent seeds, both paired taskorders. All models hypothesize classes themselves; no true inference label or historical route is consulted. Local class-density learner is much smaller. Gains from combining two predictors must be distinguished from an inherently dendritic anti-forgetting mechanism. No novelty/conference-ready claim.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
