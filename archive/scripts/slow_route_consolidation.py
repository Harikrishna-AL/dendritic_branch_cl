"""Online distillation of fast usefulness routes into input-dependent synapses.

No stored input/configuration archive. Covariance protection is an explicitly
global engineering control, not a new local biological learning rule.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make


def cue(x):
    return F.normalize(x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196),dim=1)


class Memory:
    def __init__(self,width,seed,filtered):
        self.width=width;self.filtered=filtered
        rng=torch.Generator().manual_seed(seed+10000)
        self.projection=torch.randn(196,width,generator=rng) if width!=196 else None
        self.gram=torch.zeros(width+1,width+1,dtype=torch.float64)
        self.cross=torch.zeros(width+1,106,dtype=torch.float64)
        self.links=torch.zeros(24,4,4,dtype=torch.float64)
        self.sgd=torch.zeros(width+1,96)
        self.accepted=0

    def features(self,q):
        z=q if self.projection is None else torch.tanh(q@self.projection)
        return torch.cat([z,torch.ones(len(q),1)],1)

    @torch.no_grad()
    def observe(self,q,y,route,reliable,target=None):
        ids=reliable if self.filtered else torch.ones(len(q),dtype=torch.bool)
        z=self.features(q[ids]);yy=y[ids]
        if not len(z):return
        tag=F.one_hot(route,4).flatten().float().expand(len(q),96) if target is None else target.flatten(1)
        tag=tag[ids]
        targets=torch.cat([tag,F.one_hot(yy,10).float()],1).double()
        self.gram.add_(z.double().T@z.double());self.cross.add_(z.double().T@targets)
        choices=tag.reshape(-1,24,4).argmax(-1)
        coactivity=F.one_hot(choices,4).double()
        self.links.add_(torch.einsum('bni,bnj->nij',coactivity,torch.roll(coactivity,-1,1)))
        p=(z@self.sgd).reshape(-1,24,4).softmax(-1)
        self.sgd.add_(z.T@(tag-p.flatten(1)),alpha=.1/len(z))
        self.accepted+=len(z)

    def solve(self,ridge):
        penalty=torch.eye(self.width+1,dtype=torch.float64)*ridge
        penalty[-1,-1]=0
        return torch.linalg.solve(self.gram+penalty,self.cross)

    def bytes(self):
        # Deployed analytic router; SGD is a separately measured control.
        return sum(t.numel()*t.element_size() for t in
                   [self.gram,self.cross[:,:96].contiguous(),self.links])+((self.width+1)*96*8)+(
                   0 if self.projection is None else self.projection.numel()*4)


@torch.no_grad()
def recruit(scores,links,gamma):
    j=scores.argmax(-1)
    if not gamma:return j
    # Ring-neighbor conditional coactivity; no saved whole configurations.
    right=links/links.sum(-1,keepdim=True).clamp_min(1)
    left=links/links.sum(-2,keepdim=True).clamp_min(1)
    ids=torch.arange(24)
    for _ in range(3):
        outgoing=right[ids[None,:,None],torch.arange(4)[None,None,:],torch.roll(j,-1,1)[:,:,None]]
        incoming=left[torch.roll(ids,1)[None,:,None],torch.roll(j,1,1)[:,:,None],torch.arange(4)[None,None,:]]
        j=(scores+gamma*(outgoing+incoming)).argmax(-1)
    return j


@torch.no_grad()
def evaluate(model,mem,x,y,head,gamma=0.,sgd=False,wrong_cue=False):
    before={k:v.clone() for k,v in model.state_dict().items()}
    accuracy=0;direct=0;routes=[]
    for start in range(0,len(y),512):
        bx=x[start:start+512];by=y[start:start+512]
        q=cue(bx)
        if wrong_cue:q=cue(x.roll(137,0)[start:start+512])
        z=mem.features(q)
        logits=z@(mem.sgd if sgd else head[:,:96].float())
        j=recruit(logits.reshape(-1,24,4),mem.links.float(),gamma)
        branches=torch.tanh(model.first(bx).reshape(-1,24,4))
        pred=model.head(branches.gather(2,j[:,:,None]).squeeze(-1)).argmax(1)
        accuracy+=int((pred==by).sum());routes.append(j)
        if not sgd:direct+=int(((z@head[:,96:].float()).argmax(1)==by).sum())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    return {'accuracy':accuracy/len(y),'direct_accuracy':direct/len(y),'routes':torch.cat(routes)}


def run(seed,x,y,fit,pairs,widths,teacher='actual',memory_factory=Memory):
    torch.manual_seed(seed);model=make('local','hard');opt=torch.optim.Adam(model.parameters(),lr=.001)
    memories={(w,f):memory_factory(w,seed,f) for w in widths for f in [False,True]}
    routes=[];start=time.perf_counter()
    for stage,pair in enumerate(pairs):
        ix=torch.nonzero(fit&((y==pair[0])|(y==pair[1])),as_tuple=True)[0]
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            order=ix[torch.randperm(len(ix),generator=rng)]
            for ids in order.split(128):
                bx=x[ids];by=y[ids]
                route=model.gates().argmax(-1)
                opt.zero_grad(set_to_none=True)
                logits,candidates=model(bx,counterfactual=True)
                errors=F.cross_entropy(candidates.reshape(-1,10),by[:,None,None].expand(candidates.shape[:-1]).reshape(-1),reduction='none').reshape(candidates.shape[:-1])
                # Route is associated only with CURRENT examples, before update.
                with torch.no_grad():
                    probability=logits.softmax(-1)
                    reliable=(probability.argmax(-1)==by)&(probability.max(-1).values>=.8)
                    q=cue(bx)
                    target=None
                    if teacher=='counterfactual':target=F.one_hot(errors.argmin(-1),4).float()
                    if teacher=='soft_counterfactual':target=(-errors/.2).softmax(-1)
                    for memory in memories.values():memory.observe(q,by,route,reliable,target)
                loss=F.cross_entropy(logits,by)+.5*errors.mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                opt.step();model.observe(errors.detach().mean(0))
        routes.append(model.gates().argmax(-1).clone()) # Analysis only, not memory input.
    return model,memories,torch.stack(routes),time.perf_counter()-start


@torch.no_grad()
def ordinary(model,x,y):
    p=torch.cat([model(b).argmax(1) for b in x.split(512)])
    return float((p==y).float().mean())


@torch.no_grad()
def diagnostics(model,x,y,route,history,pairs):
    pair_id=torch.zeros_like(y)
    for i,pair in enumerate(pairs):pair_id[(y==pair[0])|(y==pair[1])]=i
    expected=history[pair_id]
    exact=(route==expected).all(1)
    neuron=(route==expected).float().mean()
    oldmask=(y!=pairs[-1][0])&(y!=pairs[-1][1])
    out={'historical_exact_route_match':float(exact.float().mean()),'historical_neuron_match':float(neuron)}
    accuracy=0;oldcorrect=0;oracle=0
    for start in range(0,len(y),512):
        bx=x[start:start+512];by=y[start:start+512];rr=route[start:start+512];ee=expected[start:start+512]
        b=torch.tanh(model.first(bx).reshape(-1,24,4))
        p=model.head(b.gather(2,rr[:,:,None]).squeeze(-1)).argmax(1)
        op=model.head(b.gather(2,ee[:,:,None]).squeeze(-1)).argmax(1)
        accuracy+=int((p==by).sum());oracle+=int((op==by).sum())
        oldcorrect+=int(((p==by)&oldmask[start:start+512]).sum())
    out.update(accuracy=accuracy/len(y),old_accuracy=oldcorrect/int(oldmask.sum()),oracle_accuracy=oracle/len(y))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol={'development_seed':100,'confirmation_seeds':[101,102,103],
              'data':'50k/10k stratified training development; fresh confirmation full60k, official10k test. Two within-pair epochs, no old-task revisit.',
              'widths':[196,512],'ridges':[.1,1.,10.],'communication':[0.,.1,.5],
              'training':'Original learned24x4 usefulness branch model unchanged; shared classifier trained throughout. Pre-update active route is teacher; no task identity or boundary passed to memory.',
              'filter':'All current samples vs correct predictions with confidence>=.8; fixed threshold before observing results. Filtering uses current training feedback only.',
              'router':'Fixed pooled normalized pixels or random tanh512 cue. Analytic input-to96branch-choice output mapping accumulated online with full Gram/cross statistics. No images, prototypes or whole configurations archived. Persistent statistics are EXTRA CENTRAL MEMORY, not short-term traces or biological local learning.',
              'local communication':'Three synchronous ring-neighbor conditional coactivity loops.24x4x4statistics accumulated from actual active branches; no whole route bank.',
              'controls':'Ordinary online SGD route learning on identical cues/writes; direct10class analytic classifier on same cues/writes; wrong input cue; ordinary final gate; oracle historical route only for analysis.',
              'selection':'Select all router settings on development validation only. Also choose direct classifier settings separately on validation, so direct control is not handicapped. Fixed fresh seeds and both task orders. No official test-driven follow-ups in this script.',
              'scope':'Teacher/router distillation pilot. Automatic input-based retrieval acquired during backbone training, but router does not control backbone training. Original expensive counterfactual objective retained. Not a biological or dendrite-specific superiority claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(100)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    fit=torch.ones(len(y),dtype=torch.bool);fit[val]=False
    model,memories,history,seconds=run(100,x,y,fit,list(SPLITS),protocol['widths'])
    candidates=[]
    for (width,filtered),memory in memories.items():
        for ridge in protocol['ridges']:
            head=memory.solve(ridge)
            for gamma in protocol['communication']:
                row=evaluate(model,memory,x[val],y[val],head,gamma);row.pop('routes')
                row.update(width=width,filtered=filtered,ridge=ridge,gamma=gamma,accepted=memory.accepted)
                candidates.append(row)
        print('dev',width,filtered,'best',round(max(r['accuracy'] for r in candidates if r['width']==width and r['filtered']==filtered)*100,2),flush=True)
    selected=max(candidates,key=lambda r:r['accuracy'])
    direct=max(candidates,key=lambda r:r['direct_accuracy'])
    (out/'selection.json').write_text(json.dumps({'candidates':candidates,'router':selected,'direct':direct,'seconds':seconds},indent=2))
    print('selected',selected,'direct',direct,flush=True)
    records=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        for seed in protocol['confirmation_seeds']:
            model,memories,history,seconds=run(seed,x,y,torch.ones(len(y),dtype=torch.bool),pairs,sorted(set([selected['width'],direct['width']])))
            mem=memories[selected['width'],selected['filtered']];head=mem.solve(selected['ridge'])
            for arm,gamma,sgd,wrong in [('protected_route',selected['gamma'],False,False),('no_communication',0.,False,False),('sgd_route',0.,True,False),('wrong_cue',selected['gamma'],False,True)]:
                result=evaluate(model,mem,tx,ty,head,gamma,sgd,wrong)
                row={'seed':seed,'order':order,'arm':arm,**diagnostics(model,tx,ty,result['routes'],history,pairs),'seconds_all_controls':seconds,'router_numeric_bytes':mem.bytes(),'accepted':mem.accepted}
                records.append(row);print(order,seed,arm,round(row['accuracy']*100,2),flush=True)
            dm=memories[direct['width'],direct['filtered']];dh=dm.solve(direct['ridge'])
            dd=evaluate(model,dm,tx,ty,dh)
            records.append({'seed':seed,'order':order,'arm':'direct_classifier','accuracy':dd['direct_accuracy']})
            records.append({'seed':seed,'order':order,'arm':'ordinary_gate','accuracy':ordinary(model,tx,ty)})
            torch.save({'backbone':model.state_dict(),'history_analysis_only':history,'projection':mem.projection,'gram':mem.gram,'cross':mem.cross[:,:96].clone(),'links':mem.links,'head':head[:,:96].clone(),'sgd_control':mem.sgd},out/f'{order}_{seed}.pt')
            (out/'results.json').write_text(json.dumps(records,indent=2))
    lines=['# Fast usefulness teacher, slow input-to-branch routing','','Input-to-branch mappings are acquired during sequential backbone learning. No task ID, test feedback, replay, prototypes or configuration archive is used. Extra full covariance statistics protect the router; this is an engineering solution, not local biological plasticity.','', '| Method | Accuracy mean ± SD (six runs) |','|---|---:|']
    for arm in ['ordinary_gate','sgd_route','protected_route','no_communication','wrong_cue','direct_classifier']:
        a=[100*r['accuracy'] for r in records if r['arm']==arm]
        lines.append(f'| {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% |')
    for order in ['forward','reverse']:
        lines+=['',f'## {order} order','']
        for arm in ['protected_route','direct_classifier']:
            a=[100*r['accuracy'] for r in records if r['arm']==arm and r['order']==order]
            lines.append(f'{arm}: {st.mean(a):.2f} ± {st.stdev(a):.2f}%')
    lines+=['','## Interpretation boundaries','','The original learned backbone and shared classifier continue changing. The router learns from contemporaneous active branch choices, not retrospectively selected phase-end routes. A correct-confidence filter is a training feedback gate. All settings were selected on held-out training validation before test scoring. Historical routes and true digit pairs are used only after predictions for oracle and route-match diagnostics.','','A direct analytic classifier can solve the same dataset without retrieving branch skills. Compare its separately validation-selected result before attributing superiority to branches. These controls share cue features and sample exposure, but are not parameter/storage matched: the branch model and96routing outputs add storage and compute. Joint training of the retained backbone still uses counterfactual supervision; router-guided acquisition and biological locality remain unsolved.','','The current head can be solved from accumulated statistics on demand at any batch; the pilot solves at evaluation for speed. No boundary event changes memory. Covariance statistics and branch-conditioned coactivity persist across the stream. Deployable inference uses random cue projection, learned routing synapses, optional local links, and the final learned backbone; sufficient statistics are needed to continue learning.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:14]),flush=True)


if __name__=='__main__':main()
