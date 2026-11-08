"""Response-derived context: local branch coactivity, no input cue association.

Domain-incremental MNIST shares all ten labels across changing permutations.
Domain index appears in the stream harness/analysis, never the learner.
"""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_coded_branch_pilot import LocalCoded


class GraphLearner(LocalCoded):
    def __init__(self,seed,arm,setting):
        super().__init__(seed,'branch' if arm=='branch' else 'flat',setting['rate'],setting['tau'],64,196)
        self.arm=arm;self.active_count=64;self.rho=0.;self.gated=setting['gated']
        # These input-association tensors are absent, not merely unused caches.
        self.cue_sum=torch.empty(0);self.cue_count=torch.empty(0)
        self.links=torch.zeros(63,4,4) if arm=='branch' else torch.zeros(256,256)

    def select(self):
        if self.arm=='branch':return torch.arange(64)*4+self.error.argmin(-1)
        return self.error.flatten().topk(64,largest=False).indices

    @torch.no_grad()
    def learn(self,x,y):
        z=self.data_features(x);b=torch.tanh(z@self.weight)
        target=self.codes[y].repeat_interleave(self.k,dim=1);active=self.select()
        residual=target[:,active]-b[:,active]
        reliable=(residual.abs()<.2).float() if self.gated else torch.ones_like(residual)
        if self.arm=='branch':
            j=active%4;strength=(reliable[:,:-1]*reliable[:,1:]).sum(0)
            self.links[torch.arange(63),j[:-1],j[1:]]+=strength
        else:
            self.links[active[:,None],active[None,:]]+=reliable.T@reliable
        rate=self.rate/(1+self.usage[active]/self.tau)
        self.weight[:,active]+=(z.T@(residual*(1-b[:,active].square())))/len(x)*rate[None]
        self.usage[active]+=(residual.abs()<.2).float().sum(0)
        self.error.copy_((b-target).square().mean(0).reshape(self.n,self.k))
        self.presentations+=len(x)

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in [self.weight,self.codes,self.error,self.usage,self.links])


def permutations(seed,kind,mean):
    generator=torch.Generator().manual_seed(seed);perms=[torch.arange(784)]
    order=mean.argsort()
    for _ in range(3):
        if kind=='global':p=torch.randperm(784,generator=generator)
        else:
            p=torch.arange(784)
            for ids in order.split(49):p[ids]=ids[torch.randperm(len(ids),generator=generator)]
        perms.append(p)
    return perms


def train(seed,arm,setting,x,y,mask,perms):
    model=GraphLearner(seed,arm,setting);history=[]
    ids=torch.nonzero(mask,as_tuple=True)[0]
    for domain,p in enumerate(perms):
        for epoch in range(2):
            generator=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=generator)].split(128):model.learn(x[ix][:,p],y[ix])
        history.append(model.select().clone()) # Analysis only.
    return model,history


@torch.no_grad()
def branch_scores(model,response,gamma):
    codes=model.codes
    value=-(response[:,None].reshape(-1,1,64,4)-codes.T[None,:,:,None].transpose(1,2)).square()
    # shape B,10,64,4. Forward neighbor max-sum gives exact chain energy;
    # no full configuration templates or task hypotheses are stored.
    counts=model.links+1.
    edge=(counts/counts.sum(-1,keepdim=True)).log()
    dp=value[:,:,0]
    for n in range(1,64):dp=(dp[:,:,:,None]+gamma*edge[n-1][None,None]).max(2).values+value[:,:,n]
    return dp.max(-1).values


@torch.no_grad()
def flat_scores(model,response,gamma):
    value=-(response[:,None]-model.codes[None]).square()
    if gamma==0:return value.topk(64,dim=-1).values.sum(-1)
    # Strong dense-coactivity control: each neuron's learned neighbors seed an
    # assembly. Assemblies are reconstructed from synapses, not saved at boundaries.
    affinity=model.links.clone();affinity.fill_diagonal_(0)
    assemblies=affinity.topk(64,dim=-1).indices
    mask=torch.zeros(256,256).scatter_(1,assemblies,1.)
    unique=torch.unique(mask,dim=0)
    degree=model.links.diag().sqrt().clamp_min(1)
    norm=model.links/(degree[:,None]*degree[None,:]);norm.fill_diagonal_(0)
    coherence=(unique@norm*unique).sum(-1)/64
    return (value@unique.T+gamma*coherence[None,None]).max(-1).values


@torch.no_grad()
def evaluate(model,x,y,perms,gamma,history=None):
    result=[]
    for domain,p in enumerate(perms):
        pred=[]
        for ix in range(0,len(y),256):
            response=torch.tanh(model.data_features(x[ix:ix+256][:,p])@model.weight)
            if history is not None:
                active=history[domain];b=response[:,active]
                codes=model.codes.repeat_interleave(model.k,dim=1)[:,active]
                score=-(b[:,None]-codes[None]).square().sum(-1)
            else:score=branch_scores(model,response,gamma) if model.arm=='branch' else flat_scores(model,response,gamma)
            pred.append(score.argmax(-1))
        result.append(float((torch.cat(pred)==y).float().mean()))
    return {'accuracy':st.mean(result),'domains':result}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'dataset':'MNIST','domains':4,'stream_types':['global','mean_matched'],'dev_seed':270,'confirmation_seeds':[271,272,273],
              'stream':'Same ten classes per domain; identity then3pixel permutations. Mean-matched permutations shuffle within16source-TRAIN-mean quantile groups, reducing marginal context cues. No domain ID/boundary supplied to learner. Each current domain2passes; source images reused transformed in new domains (standard permuted-MNIST protocol),480kcurrent presentations in confirmation.',
              'memory':'Learned branch response weights, fixed coded readout, recent error, usage damping, and local neighbor coactivity ONLY. No input-conditioned cue sums, covariance, image bank or whole route archive. Teaching is label-derived per-neuron coded expected response, explicit supervised feedback.',
              'inference':'Each class hypothesis sends its coded response expectation. Local response squared errors plus neighbor coactivity identify a coherent assembly. Exact chain max-sum is a known graphical-model algorithm, not a biological dynamics/novel decoding claim. All10classes eligible; no true labels/IDs enter inference.',
              'flat_control':'Same256data units, same64active units and coded local learning; dense pair coactivity reconstructs assemblies from synaptic affinity rows. MORE persistent graph memory than branch chain; strong structural control, not equal-storage. No externally saved configurations.',
              'development':'Fresh50k/10k training split; rates.1/1, tau1000/10000, graphwrites all/gated, inference gamma0/.1/1/10 selected on validation. Both streamtypes predefined. Official confirmation only if branchvalidation>=60%. Oracle phaseend configs analysisonly; fresh3seeds. No test selection.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    generator=torch.Generator().manual_seed(270)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=generator)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;mean=x[mask].mean(0);selections={};rows=[]
    for kind in protocol['stream_types']:
        perms=permutations(270,kind,mean);candidates=[];selected={}
        for arm in ['branch','flat']:
            for rate in [.1,1.]:
                for tau in [1000.,10000.]:
                    for gated in [False,True]:
                        setting={'rate':rate,'tau':tau,'gated':gated}
                        model,history=train(270,arm,setting,x,y,mask,perms)
                        oracle=evaluate(model,x[val],y[val],perms,0.,history)
                        for gamma in [0.,.1,1.,10.]:
                            result=evaluate(model,x[val],y[val],perms,gamma)
                            candidates.append({'arm':arm,**setting,'gamma':gamma,**result,'oracle_accuracy':oracle['accuracy'],'numeric_bytes':model.bytes()})
                        print('dev',kind,arm,rate,tau,gated,'best',round(max(r['accuracy'] for r in candidates if r['arm']==arm)*100,2),'oracle',round(oracle['accuracy']*100,2),flush=True)
                        (out/'development_progress.json').write_text(json.dumps({'kind':kind,'candidates':candidates},indent=2))
            selected[arm]=max([r for r in candidates if r['arm']==arm],key=lambda r:r['accuracy'])
        selections[kind]={'selected':selected,'candidates':candidates};(out/'selection.json').write_text(json.dumps(selections,indent=2))
        if selected['branch']['accuracy']<.6:print('validationgate failed',kind,flush=True);continue
        for seed in protocol['confirmation_seeds']:
            perms=permutations(seed,kind,x.mean(0))
            for arm in ['branch','flat']:
                setting=selected[arm];model,history=train(seed,arm,setting,x,y,torch.ones(len(y),dtype=torch.bool),perms)
                result=evaluate(model,tx,ty,perms,setting['gamma']);oracle=evaluate(model,tx,ty,perms,0.,history)
                rows.append({'kind':kind,'seed':seed,'arm':arm,**result,'oracle_accuracy':oracle['accuracy'],'numeric_bytes':model.bytes(),'settings':setting})
                print(kind,seed,arm,round(result['accuracy']*100,2),'oracle',round(oracle['accuracy']*100,2),flush=True)
                torch.save({'weight':model.weight,'codes':model.codes,'links':model.links,'error':model.error,'usage':model.usage,'settings':setting,'permutations':perms},out/f'{kind}_{seed}_{arm}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Context from branch responses and coactivity alone','','Domain-incremental4permutation MNIST, same10labels reused. No input cue association, task ID, global covariance, backpropagation, replay or saved configuration archive. Local coded teaching remains explicit feedback.','', '| Stream | Architecture | Validation best | Fresh test mean |','|---|---|---:|---:|']
    for kind,selection in selections.items():
        for arm in ['branch','flat']:
            rs=[r for r in rows if r['kind']==kind and r['arm']==arm]
            result=f'{100*st.mean(r["accuracy"] for r in rs):.2f}%' if rs else 'Not run: branch validationgate failed'
            lines.append(f'| {kind} | {arm} | {100*selection["selected"][arm]["accuracy"]:.2f}% | {result} |')
    lines+=['','Flat control uses more graph memory and fixed64active units, so this is not a fully optimized ordinary baseline or a matched budget claim. Mean-matched permutations are approximate, not proof of identical marginal distributions. Exact graphical decoding itself is established; any success needs mechanism ablations and stronger analytic/associative/deep controls. No top-conference-ready claim.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
