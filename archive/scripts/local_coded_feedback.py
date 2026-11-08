"""Local branch energy inference and stronger covariance-free controls."""
import argparse
import json
import math
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.local_coded_branch_pilot import LocalCoded,evaluate as old_evaluate


@torch.no_grad()
def evidence(model,x):
    q=model.cues(x)
    p=(model.cue_sum+1)/(model.cue_count[:,None]+2);p=p.clamp(1e-6,1-1e-6)
    scores=q@(p/(1-p)).log().T+(1-p).log().sum(1)[None]
    scores+=((model.cue_count+1)/(model.cue_count.sum()+model.units)).log()[None]
    return scores.masked_fill(model.cue_count[None]==0,-1e6)


@torch.no_grad()
def predict(model,x,beta):
    if beta=='independent':return model.predict(x,'bernoulli')
    response=torch.tanh(model.data_features(x)@model.weight)
    codes=model.codes.repeat_interleave(model.k,dim=1)
    cue=evidence(model,x)
    # Each possible output sends its fixed coded expectation to local branches.
    # No true label enters this function. Branch energy equals its training loss.
    value=-(response[:,None]-codes[None]).square()+beta*cue[:,None]
    if model.k==4:
        score=value.reshape(-1,10,model.n,4).max(-1).values.sum(-1)
        offset=beta*cue.reshape(-1,model.n,4).max(-1).values.sum(-1)
    else:
        score=value.topk(model.active_count,dim=-1).values.sum(-1)
        offset=beta*cue.topk(model.active_count,dim=-1).values.sum(-1)
    return (score-offset[:,None]).argmax(1)


@torch.no_grad()
def accuracy(model,x,y,beta):
    pred=torch.cat([predict(model,bx,beta) for bx in x.split(512)])
    return float((pred==y).float().mean())


def train(seed,arm,setting,x,y,mask,pairs):
    tau=math.inf if setting['tau']=='infinity' else setting['tau']
    model=LocalCoded(seed,arm,setting['rate'],tau,setting.get('neurons',24),setting.get('cue_dim',196))
    model.rho=setting.get('rho',.95);model.active_count=setting.get('active_count',24)
    history=[]
    for stage,pair in enumerate(pairs):
        ids=torch.nonzero(mask&((y==pair[0])|(y==pair[1])),as_tuple=True)[0]
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):model.learn(x[ix],y[ix])
        history.append(model.selection(torch.zeros(1,model.n,model.k)).clone())
    return model,history


class ClassAssociation:
    def __init__(self,dim,rule):self.dim=dim;self.rule=rule;self.sums=torch.zeros(10,dim);self.counts=torch.zeros(10)
    def features(self,x):
        if self.dim==196:x=x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196)
        return (x>x.mean(1,keepdim=True)).float() if self.rule=='bernoulli' else F.normalize(x,dim=1)
    def learn(self,x,y):self.sums.index_add_(0,y,self.features(x));self.counts+=torch.bincount(y,minlength=10)
    def predict(self,x):
        q=self.features(x)
        if self.rule=='cosine':score=q@F.normalize(self.sums,dim=1).T
        else:
            p=((self.sums+1)/(self.counts[:,None]+2)).clamp(1e-6,1-1e-6)
            score=q@(p/(1-p)).log().T+(1-p).log().sum(1)[None]
            score+=((self.counts+1)/(self.counts.sum()+10)).log()[None]
        return score.argmax(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'datasets':['mnist','fashion_mnist'],'development_seeds':[210,220],'confirmation_seeds':[[211,212,213],[221,222,223]],
              'branch':'Local coded teaching with recent-error EMA selection and local cue coactivity. No covariance/backprop/replay/taskID. Previous pilot motivated new inference rule, not selected using its officialtest scores.',
              'energy':'Each possible class sends its fixed coded expected response to all branches. Sum per-neuron best negative squared response error plus beta local cue likelihood; no true test label. beta selected on training validation, including independent retrieval option.',
              'short_memory_control':'Compare recent EMA.95 with cumulative all-history errors and previous-batch rho0. Independently tune rate/tau and feedback decoding. Same local feedback and data access.',
              'flat_control':'Same96data units/coactivity synapses, no neuron grouping. Tune active_count24/48/72/96 plus rate/tau and decoder; no artificial24unit bottleneck.',
              'strong_local_control':'Independently select raw784 or pooled196 class-conditioned associative means/cosine or Bernoulli counts. Direct supervised class synapses; no covariance/global gradient/archive. Much smaller memory than branch model; strong control, not memorymatched.',
              'selection':'Fresh50k/10k training validation selects all options. Confirmation fresh3seeds bothorders per dataset ONLY if selected localbranch development>=60%. Keep all failed candidates. Fixed coded outputs differ from original learned classifier; no novelty or biological implementation claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    # Meaningful local-update correctness check against the mathematical gradient.
    model=LocalCoded(999,'branch',.1,math.inf);x=torch.rand(7,784);y=torch.arange(7)
    z=model.data_features(x);w=model.weight.clone().requires_grad_();response=torch.tanh(z@w).reshape(-1,24,4)
    active=model.selection(response.detach());target=model.codes[y][:,:,None].expand_as(response).flatten(1)
    loss=.5*(response.flatten(1)[:,active]-target[:,active]).square().sum()/len(x)
    gradient=torch.autograd.grad(loss,w)[0];expected=w.detach()-.1*gradient;model.learn(x,y)
    assert torch.allclose(model.weight,expected,atol=1e-7,rtol=1e-6)
    (out/'validation.json').write_text(json.dumps({'local_update_matches_selected_unit_squared_error_gradient':True}))
    selections={};rows=[]
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['development_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;selected={};candidates=[]
        for arm,rhos,active_counts in [('branch',[.95,'cumulative',0.],[24]),('flat',[.95],[24,48,72,96])]:
            for rho in rhos:
                for count in active_counts:
                    for rate in [.1,1.,10.]:
                        for tau in [1000.,10000.,'infinity']:
                            setting={'rho':rho,'active_count':count,'rate':rate,'tau':tau}
                            model,_=train(seed,arm,setting,x,y,mask,list(SPLITS))
                            for beta in [0.,.001,.01,.1,1.,10.,'independent']:
                                a=accuracy(model,x[val],y[val],beta)
                                candidates.append({'arm':arm,**setting,'beta':beta,'accuracy':a,'numeric_bytes':model.bytes()})
                    print('dev',dataset,arm,rho,count,'best',round(max(r['accuracy'] for r in candidates if r['arm']==arm and r['rho']==rho and r['active_count']==count)*100,2),flush=True)
            if arm=='branch':
                for name,rho in [('branch',.95),('cumulative','cumulative'),('previous_batch',0.)]:selected[name]=max([r for r in candidates if r['arm']==arm and r['rho']==rho],key=lambda r:r['accuracy'])
            else:selected[arm]=max([r for r in candidates if r['arm']==arm],key=lambda r:r['accuracy'])
        direct=[]
        for dim in [196,784]:
            for rule in ['cosine','bernoulli']:
                model=ClassAssociation(dim,rule)
                # Cumulative statistics order-independent; two passes match exposure.
                for _ in range(2):
                    for ids in torch.nonzero(mask,as_tuple=True)[0].split(512):model.learn(x[ids],y[ids])
                pred=torch.cat([model.predict(bx) for bx in x[val].split(512)])
                direct.append({'dim':dim,'rule':rule,'accuracy':float((pred==y[val]).float().mean())})
        selected['direct']=max(direct,key=lambda r:r['accuracy'])
        selections[dataset]={'selected':selected,'candidates':candidates,'direct_candidates':direct};(out/'selection.json').write_text(json.dumps(selections,indent=2))
        print('selected',dataset,selected,flush=True)
        if selected['branch']['accuracy']<.6:print('gate fails',dataset,flush=True);continue
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                for arm in ['branch','cumulative','previous_batch','flat']:
                    setting=selected[arm];model,history=train(seed,setting['arm'],setting,x,y,torch.ones(len(y),dtype=torch.bool),pairs)
                    before=[v.clone() for v in [model.weight,model.error,model.usage,model.cue_sum,model.cue_count]]
                    a=accuracy(model,tx,ty,setting['beta']);ind=accuracy(model,tx,ty,'independent')
                    assert all(torch.equal(a,b) for a,b in zip(before,[model.weight,model.error,model.usage,model.cue_sum,model.cue_count]))
                    oracle=old_evaluate(model,tx,ty,pairs,history=history)['accuracy']
                    rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':arm,'accuracy':a,'independent_accuracy':ind,'oracle_accuracy':oracle,'numeric_bytes':model.bytes()})
                    print(dataset,order,seed,arm,round(a*100,2),flush=True)
                setting=selected['direct'];model=ClassAssociation(setting['dim'],setting['rule'])
                for pair in pairs:
                    ids=torch.nonzero((y==pair[0])|(y==pair[1]),as_tuple=True)[0]
                    for _ in range(2):
                        for batch in ids.split(512):model.learn(x[batch],y[batch])
                pred=torch.cat([model.predict(bx) for bx in tx.split(512)])
                rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'direct','accuracy':float((pred==ty).float().mean()),'numeric_bytes':model.sums.numel()*4+model.counts.numel()*4})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Local coded feedback and memory-duration controls','','No covariance, backpropagation, example/configuration bank, task ID or inference labels. Per-neuron TRAINING feedback comes from fixed class codes. Local input coactivity synapses are persistent associative memory.','', '| Dataset | Local learner | Validation-selected accuracy | Fresh officialtest mean |','|---|---|---:|---:|']
    for dataset,selection in selections.items():
        for arm in ['branch','cumulative','previous_batch','flat','direct']:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            value=f'{100*st.mean(r["accuracy"] for r in rs):.2f}%' if rs else 'Not run: branch-validation gate failed'
            lines.append(f'| {dataset} | {arm} | {100*selection["selected"][arm]["accuracy"]:.2f}% | {value} |')
    lines+=['','Controls independently select rates/metaplasticity/inference. Flat control can select24–96units; it is not forced to learn only24. Direct local classifier has far less memory; no total-memory advantage is claimed over it. Ordinary classifier versus latent coded inference is mathematically related to established local target-projection/latent-variable learning; no novelty claim. Local gradient implementation is checked, and test inference is asserted not to alter state.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
