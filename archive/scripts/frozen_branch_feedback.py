"""Can actual-outcome feedback learn routing over already retained skills?

Diagnostic only: branch backbone has already learned all five tasks. Router
training revisits historical TRAINING images. No saved configurations or task
IDs are given to the router. This is not an end-to-end CL result.
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
from bdh_cl.scripts.split_mnist_usefulness import make


@torch.no_grad()
def cache(model,x):
    q=x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196)
    q=F.normalize(q,dim=1)
    b=torch.cat([torch.tanh(model.first(z).reshape(-1,24,4)) for z in x.split(512)])
    return q,b


def initialize(seed):
    g=torch.Generator().manual_seed(seed)
    return torch.randn(197,24,4,generator=g)*.01


def scores(q,u):
    return torch.cat([q,torch.ones(len(q),1)],1)@u.reshape(197,96)


@torch.no_grad()
def evaluate(q,b,y,u,model):
    j=scores(q,u).reshape(-1,24,4).argmax(-1)
    h=b.gather(2,j[:,:,None]).squeeze(-1)
    pred=model.head(h).argmax(1)
    return {'accuracy':float((pred==y).float().mean()),'pairs':[
        float((pred[(y==a)|(y==c)]==y[(y==a)|(y==c)]).float().mean()) for a,c in SPLITS],
        'dominant_branch_fraction':float(F.one_hot(j,4).float().mean(0).max(1).values.mean())}


def train(q,b,y,model,seed,arm,rate,groups,epochs):
    u=initialize(seed);initial=u.clone();baseline=-math.log(10) if arm!='binary' else 0.
    stages=[];start=time.perf_counter()
    if arm=='gradient_control':
        u.requires_grad_();opt=torch.optim.Adam([u],lr=.01)
    for stage,ix in enumerate(groups):
        for epoch in range(epochs):
            g=torch.Generator().manual_seed(seed*10000+stage*100+epoch)
            choices=torch.Generator().manual_seed(seed*10000+stage*100+epoch+1000000)
            noise=torch.Generator().manual_seed(seed*10000+stage*100+epoch+2000000)
            order=ix[torch.randperm(len(ix),generator=g)]
            for ids in order.split(128):
                bx=q[ids];bb=b[ids];by=y[ids]
                aug=torch.cat([bx,torch.ones(len(bx),1)],1)
                p=(aug@u.reshape(197,96)).reshape(-1,24,4).softmax(-1)
                if arm=='gradient_control':
                    z=model.head((bb*p).sum(-1));loss=F.cross_entropy(z,by)
                    opt.zero_grad();loss.backward();opt.step();continue
                with torch.no_grad():
                    j=torch.multinomial(p.reshape(-1,4),1,generator=choices).reshape(-1,24)
                    z=model.head(bb.gather(2,j[:,:,None]).squeeze(-1))
                    r=(z.argmax(-1)==by).float() if arm=='binary' else -F.cross_entropy(z,by,reduction='none').clamp(max=10)
                    adv=r-baseline;baseline=.95*baseline+.05*float(r.mean())
                    if arm=='shuffled':adv=adv[torch.randperm(len(ids),generator=noise)]
                    tag=F.one_hot(j,4).float()-p
                    credit=tag*adv[:,None,None]
                    # Local entropy support avoids premature collapse; fixed .01.
                    logp=p.clamp_min(1e-9).log();entropy=-(p*logp).sum(-1,keepdim=True)
                    credit+=.01*(-p*(logp+entropy))
                    if arm!='frozen':u.add_((aug.T@credit.reshape(-1,96)).reshape(197,24,4),alpha=rate/len(ids))
        stages.append(u.detach().clone())
    return u.detach(),stages,{'seconds':time.perf_counter()-start,'router_update_norm':float((u.detach()-initial).norm())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'dev_backbone_seed':34,'dev_router_seed':74,'evaluation_pairs':[[34,75],[35,76],[36,77]],
              'scope':'Frozen-backbone retrieval diagnostic, NOT a new strict CL result. Backbone checkpoints previously trained with supervised counterfactual objective on all five tasks. Router joint training revisits historical training examples; optional sequential router training uses same prelearned backbone.',
              'feedback':'Actual single sampled configuration only. Binary correctness or negative observed cross-entropy clipped at10, scalar EMA baseline. Shuffled-feedback control. Explicit local score-function routing update; fixed entropy coefficient .01. No alternative branch prediction scores, prototype tables, config cache or task ID.',
              'parameters':'18,912 routing synapses (197x24x4), plus frozen75,610-weight backbone. Fixed 2x2 pooling and L2-normalized inputs. Transient eligibility is factored input x (selected-minus-probability), immediate feedback.',
              'development':'50,000 fit/10,000 stratified training validation; all were previously seen by frozen backbone. Select rate from1/10/100 separately for binary and loss reward,20 epochs. No official test selection.',
              'controls':'Frozen random router, shuffled negative-loss rewards at selected loss rate, differentiable soft-gating CE router at Adam .01 (engineering control, not biological). Soft-gating control evaluated with argmax like other routers.',
              'sequential':'Run if selected loss development val>=70%; two epochs per pair. This tests router forgetting over an already trained backbone and must not be presented as single-pass continual learning.',
              'validation':'Backbone weights/head/memory buffers remain bitwise unchanged. No classifier refitting. Official test labels only score.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    g=torch.Generator().manual_seed(74)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=g)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;ix=torch.nonzero(mask,as_tuple=True)[0]
    model=make('local','hard');model.load_state_dict(torch.load('results/split_mnist_compute_comparison_20261013/34_local.pt',weights_only=True));model.requires_grad_(False)
    q,b=cache(model,x);dev=[];selected={}
    for arm in ['binary','loss']:
        for rate in [1.,10.,100.]:
            u,_,stats=train(q,b,y,model,74,arm,rate,[ix],20)
            row={'arm':arm,'rate':rate,**evaluate(q[val],b[val],y[val],u,model),**stats};dev.append(row)
            print('development',arm,rate,round(row['accuracy']*100,2),flush=True)
        selected[arm]=max([r for r in dev if r['arm']==arm],key=lambda r:r['accuracy'])
    (out/'selection.json').write_text(json.dumps({'candidates':dev,'selected':selected},indent=2))
    records=[]
    for backbone_seed,seed in protocol['evaluation_pairs']:
        model.load_state_dict(torch.load(f'results/split_mnist_compute_comparison_20261013/{backbone_seed}_local.pt',weights_only=True))
        before={k:v.clone() for k,v in model.state_dict().items()}
        q,b=cache(model,x);tq,tb=cache(model,tx)
        for arm in ['frozen','binary','loss','shuffled','gradient_control']:
            rate=selected['binary' if arm=='binary' else 'loss']['rate']
            u,_,stats=train(q,b,y,model,seed,arm,rate,[torch.arange(len(y))],20)
            row={'backbone_seed':backbone_seed,'router_seed':seed,'mode':'joint_diagnostic','arm':arm,**evaluate(tq,tb,ty,u,model),**stats}
            records.append(row);torch.save(u,out/f'{seed}_joint_{arm}.pt')
            print(seed,arm,round(row['accuracy']*100,2),flush=True)
        if selected['loss']['accuracy']>=.7:
            for arm in ['loss','shuffled','gradient_control']:
                groups=[torch.nonzero((y==a)|(y==c),as_tuple=True)[0] for a,c in SPLITS]
                u,stages,stats=train(q,b,y,model,seed,arm,selected['loss']['rate'],groups,2)
                scores_=[evaluate(tq,tb,ty,v,model) for v in stages]
                records.append({'backbone_seed':backbone_seed,'router_seed':seed,'mode':'sequential_router_diagnostic','arm':arm,**scores_[-1],**stats,'stages':scores_})
                print(seed,'sequential router',arm,round(scores_[-1]['accuracy']*100,2),flush=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        (out/'results.json').write_text(json.dumps(records,indent=2))
    lines=['# Frozen branch computations: learning selection from scalar feedback','',
           '**Retrieval diagnostic, not autonomous continual learning.** The backbone already learned all tasks '
           'through the previous supervised method. This asks whether scalar feedback can learn input-dependent '
           'selection without a configuration cache or prototypes, keeping all final branch weights/head fixed.', '',
           '| Mode | Router | Test accuracy mean ± SD |', '|---|---|---:|']
    for mode in ['joint_diagnostic','sequential_router_diagnostic']:
        for arm in ['frozen','binary','loss','shuffled','gradient_control']:
            a=[100*r['accuracy'] for r in records if r['mode']==mode and r['arm']==arm]
            if a:lines.append(f'| {mode} | {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% |')
    lines+=['','Negative-loss reward is more informative than binary correctness, but still a label-derived '
            'global teaching signal. The local trace/update receives that scalar and the actual sampled choice, '
            'not every alternative branch error. An entropy term discourages early collapse. All routing '
            'associations reside in18,912 additional synaptic weights, not an external association table.', '',
            'Joint router fitting accesses historical training data again. Sequential router fitting starts '
            'from the same backbone that already saw future tasks. Neither condition is a strict single-stream '
            'continual-learning solution. A success here only establishes that fixed computations make the '
            'routing-credit problem tractable. All official test outcomes were evaluated after validation '
            'selection, and final backbone state was asserted unchanged for every condition.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:19]),flush=True)


if __name__=='__main__':main()
