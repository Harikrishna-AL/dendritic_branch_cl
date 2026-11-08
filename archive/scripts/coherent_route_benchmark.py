"""Can learned local branch coactivity resolve incompatible independent choices?

Exact ring max-sum is an engineering inference control: messages traverse
neighbor edges, but no biological implementation is claimed. No route archive.
"""
import argparse
import itertools
import json
import statistics as st
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.conference_gate_benchmark import BranchRouter,Direct,fit,score,BUDGET,width_for,direct_bytes,BUILD
from bdh_cl.scripts.slow_route_consolidation import recruit,diagnostics


def edge_potential(links,kind,alpha=1.):
    counts=links.double()+alpha
    if kind=='conditional':return (counts/counts.sum(-1,keepdim=True)).log().float()
    joint=counts/counts.sum((-1,-2),keepdim=True)
    return (joint/(joint.sum(-1,keepdim=True)*joint.sum(-2,keepdim=True))).log().float()


@torch.no_grad()
def ring_map(unary,edges,beta):
    batch,n,k=unary.shape
    if beta==0:return unary.argmax(-1)
    # Four fixed starting-state hypotheses in parallel; local max-sum messages.
    identity=torch.eye(k,dtype=torch.bool)
    dp=unary[:,0,None,:].expand(batch,k,k).masked_fill(~identity[None],float('-inf'))
    previous=[]
    for i in range(1,n):
        value,pointer=(dp[:,:,:,None]+beta*edges[i-1][None,None]).max(2)
        dp=value+unary[:,i,None,:];previous.append(pointer)
    closed=dp+beta*edges[-1].T[None]
    best=closed.flatten(1).argmax(-1);start=best//k;end=best%k
    path=torch.empty(batch,n,dtype=torch.long);path[:,-1]=end
    ids=torch.arange(batch)
    for i in range(n-1,0,-1):path[:,i-1]=previous[i-1][ids,start,path[:,i]]
    return path


def validate_map():
    generator=torch.Generator().manual_seed(999)
    for n in [2,3,4]:
        unary=torch.randn(7,n,3,generator=generator);edges=torch.randn(n,3,3,generator=generator)
        routes=torch.tensor(list(itertools.product(range(3),repeat=n)))
        exact=torch.zeros(7,len(routes))
        for i in range(n):exact+=unary[:,i,routes[:,i]]+.7*edges[i,routes[:,i],routes[:,(i+1)%n]][None]
        expected=routes[exact.argmax(1)];actual=ring_map(unary,edges,.7)
        assert torch.equal(expected,actual)
    assert torch.equal(ring_map(unary,edges,0.),unary.argmax(-1))


class PooledDirect(Direct):
    def __init__(self,seed,settings):
        from bdh_cl.scripts.slow_route_consolidation import cue
        self.cue=cue;self.width=width_for(lambda w:196*w*4+8*(w+1)**2+16*(w+1)*10)
        rng=torch.Generator().manual_seed(seed+10000)
        self.projection=torch.randn(196,self.width,generator=rng)
        self.gram=torch.zeros(self.width+1,self.width+1,dtype=torch.float64)
        self.cross=torch.zeros(self.width+1,10,dtype=torch.float64)
        self.head=torch.zeros(self.width+1,10,dtype=torch.float64)
        self.settings=settings;self.presentations=0;self.replays=0

    def features(self,x):
        z=torch.tanh(self.settings['scale']*(self.cue(x)@self.projection))
        return torch.cat([z,torch.ones(len(z),1)],1)


@torch.no_grad()
def cache(model,x):
    unary=[];branches=[]
    for bx in x.split(512):
        unary.append(model.features(bx))
        branches.append(torch.tanh(model.model.first(bx).reshape(-1,24,4)))
    return torch.cat(unary),torch.cat(branches)


@torch.no_grad()
def evaluate_cached(model,features,branches,y,settings):
    choices=[];pred=[];start=time.perf_counter()
    edges=edge_potential(model.links,settings.get('kind','conditional'))
    for offset in range(0,len(y),512):
        scores=(features[offset:offset+512]@model.head.float()).reshape(-1,24,4)
        if settings['mode']=='map':j=ring_map(scores,edges,settings['beta'])
        else:j=recruit(scores,model.links,settings.get('gamma',0.))
        hidden=branches[offset:offset+512].gather(2,j[:,:,None]).squeeze(-1)
        pred.append(model.model.head(hidden).argmax(-1));choices.append(j)
    pred=torch.cat(pred)
    return {'accuracy':float((pred==y).float().mean()),'choices':torch.cat(choices),'inference_seconds_cached':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);validate_map()
    BUILD['pooled_direct']=PooledDirect
    protocol={'budget_bytes':BUDGET,'datasets':['mnist','fashion_mnist'],'dev_seeds':[150,160],'confirmation_seeds':[[151,152,153],[161,162,163]],
              'ridges':[.1,1.,10.],'betas':[.01,.05,.1,.5,1.,2.],'edges':['conditional','pmi'],
              'motivation':'Preceding same-budget benchmark showed poorer branch accuracy. Exact replay audit on one fresh seed each found oracle final-weight accuracy88.15% MNIST and93.91% Fashion versus87.66% and74.84% automatic routing. Tests whether incoherent per-neuron selection accounts for the Fashion gap.',
              'inference':'Exact MAP of ring factor graph using current input unary scores and24x4x4online coactivity counts. Four starting-state hypotheses, forward max-sum neighbor messages and backward choice messages. No stored whole route or task ID. This is an engineering inference algorithm, not demonstrated biological local dynamics.',
              'controls':'Independent branches and preceding3-loop heuristic separately validation-selected; direct analytic classifier raw784or pooled196 cues selected independently at SAME ongoing numeric budget. Pooled direct added because earlier work showed pooling improves ordinary analytic control; no handicap from raw-only features.',
              'development':'Fresh50k/10k training split; select ridge, beta and edge normalization using validation only. Direct selects input preprocessing, scale and ridge on validation. Fresh3seeds per dataset and both orders. Same original branch training, optimizer/memory budgets and current-example exposure.',
              'scope':'Coherent decoding effect only. Full covariance router remains. Historical routes are recorded only for oracle diagnostics; never read by MAP or learned memory.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    rows=[];selections={}
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['dev_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False
        model,_=fit('branch',seed,{'ridge':1.,'gamma':0.},x,y,mask,list(SPLITS),x[val],y[val],False)
        features,branches=cache(model,x[val]);candidates=[]
        for ridge in protocol['ridges']:
            model.settings['ridge']=ridge;model.refresh()
            settings=[{'mode':'independent','gamma':0.},{'mode':'heuristic','gamma':.5}]+[
                {'mode':'map','kind':kind,'beta':beta} for kind in protocol['edges'] for beta in protocol['betas']]
            for setting in settings:
                result=evaluate_cached(model,features,branches,y[val],setting);result.pop('choices')
                candidates.append({'ridge':ridge,**setting,**result})
            print('dev',dataset,'ridge',ridge,'best',round(max(r['accuracy'] for r in candidates)*100,2),flush=True)
        selected={mode:max([r for r in candidates if r['mode']==mode],key=lambda r:r['accuracy']) for mode in ['independent','heuristic','map']}
        direct_candidates=[]
        for arm in ['direct','pooled_direct']:
            for scale in [.5,1.,2.]:
                for ridge in [.1,1.,10.]:
                    _,result=fit(arm,seed,{'scale':scale,'ridge':ridge},x,y,mask,list(SPLITS),x[val],y[val],False)
                    direct_candidates.append({'arm':arm,'scale':scale,'ridge':ridge,'accuracy':result['accuracy']})
        direct=max(direct_candidates,key=lambda r:r['accuracy'])
        selections[dataset]={'selected':selected,'direct':direct,'route_candidates':candidates,'direct_candidates':direct_candidates}
        (out/'selection.json').write_text(json.dumps(selections,indent=2))
        print('selected',dataset,selected,'direct',direct,flush=True)
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                model,result=fit('branch',seed,{'ridge':1.,'gamma':0.},x,y,torch.ones(len(y),dtype=torch.bool),pairs,tx,ty)
                history=torch.tensor(result['historical_routes_analysis_only']);features,branches=cache(model,tx)
                before=[v.clone() for v in model.arrays() if v is not model.head]
                for arm,setting in selected.items():
                    model.settings['ridge']=setting['ridge'];model.refresh()
                    evaluated=evaluate_cached(model,features,branches,ty,setting)
                    diag=diagnostics(model.model,tx,ty,evaluated['choices'],history,pairs)
                    rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':arm,'settings':setting,**diag,'numeric_bytes':model.bytes(),'cached_inference_seconds':evaluated['inference_seconds_cached']})
                    print(dataset,order,seed,arm,round(diag['accuracy']*100,2),'oracle',round(diag['oracle_accuracy']*100,2),flush=True)
                assert all(torch.equal(a,b) for a,b in zip(before,[v for v in model.arrays() if v is not model.head]))
                setting=selected['map'];model.settings['ridge']=setting['ridge'];model.refresh()
                torch.save({'backbone':model.model.state_dict(),'projection':model.projection,'head':model.head,'links':model.links,'history_analysis_only':history},out/f'{dataset}_{order}_{seed}.pt')
                _,result=fit(direct['arm'],seed,{'scale':direct['scale'],'ridge':direct['ridge']},x,y,torch.ones(len(y),dtype=torch.bool),pairs,tx,ty)
                rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'direct','preprocessing':direct['arm'],**result})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Coherent distributed branch-route decoding','','Fresh-seed comparison of exact ring communication, independent routing and the previous heuristic. No task IDs, example bank or saved configuration archive. Global analytic router protection remains.','',
           '| Dataset | Inference | Accuracy ± SD (six paired runs) | Oracle historical routes |','|---|---|---:|---:|']
    for dataset in protocol['datasets']:
        for arm in ['independent','heuristic','map','direct']:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm];a=[100*r['accuracy'] for r in rs]
            oracle=f'{100*st.mean(r["oracle_accuracy"] for r in rs):.2f}%' if arm!='direct' else '—'
            lines.append(f'| {dataset} | {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {oracle} |')
    lines+=['','MAP was validated against exhaustive enumeration on random2–4node toy rings. Every final branch backbone/statistic array except the analytically refitted routing head is asserted identical across inference arms. Oracle labels only score retained-route diagnostics; they never enter inference. Exact ring maximization is a known graphical-model algorithm, not new biological machinery. Passing accuracy controls would not alone establish novelty or conference readiness. Three seeds each, paired task-order repeats, no test tuning.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
