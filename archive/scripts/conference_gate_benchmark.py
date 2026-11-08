"""Matched ongoing numeric storage, fresh seeds, two class-incremental datasets.

This is a decision experiment, not a promise of conference readiness. Timings
are measured rather than claimed equal; replay has extra old-sample exposure.
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.slow_route_consolidation import cue,recruit


BUDGET=4_205_520


def tensor_bytes(tensors):return sum(t.numel()*t.element_size() for t in tensors)


def optimizer_bytes(opt):
    return tensor_bytes([v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)])


class BranchRouter:
    def __init__(self,seed,settings):
        torch.manual_seed(seed);self.model=make('local','hard')
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.001)
        rng=torch.Generator().manual_seed(seed+10000)
        self.projection=torch.randn(196,512,generator=rng)
        self.gram=torch.zeros(513,513,dtype=torch.float64)
        self.cross=torch.zeros(513,96,dtype=torch.float64)
        self.head=torch.zeros(513,96,dtype=torch.float64)
        self.links=torch.zeros(24,4,4)
        self.settings=settings;self.presentations=0;self.replays=0

    def features(self,x):
        z=torch.tanh(cue(x)@self.projection)
        return torch.cat([z,torch.ones(len(z),1)],1)

    def learn(self,x,y):
        route=self.model.gates().argmax(-1)
        self.opt.zero_grad(set_to_none=True)
        logits,candidates=self.model(x,counterfactual=True)
        errors=F.cross_entropy(candidates.reshape(-1,10),y[:,None,None].expand(candidates.shape[:-1]).reshape(-1),reduction='none').reshape(candidates.shape[:-1])
        with torch.no_grad():
            z=self.features(x).double();tag=F.one_hot(route,4).flatten().double().expand(len(x),96)
            self.gram.add_(z.T@z);self.cross.add_(z.T@tag)
            self.links[torch.arange(24),route,torch.roll(route,-1)]+=len(x)
        loss=F.cross_entropy(logits,y)+.5*errors.mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),5);self.opt.step()
        self.model.observe(errors.detach().mean(0));self.presentations+=len(x)

    @torch.no_grad()
    def refresh(self):
        penalty=torch.eye(513,dtype=torch.float64)*self.settings['ridge'];penalty[-1,-1]=0
        self.head.copy_(torch.linalg.solve(self.gram+penalty,self.cross))

    @torch.no_grad()
    def predict(self,x):
        j=recruit((self.features(x)@self.head.float()).reshape(-1,24,4),self.links,self.settings['gamma'])
        b=torch.tanh(self.model.first(x).reshape(-1,24,4))
        return self.model.head(b.gather(2,j[:,:,None]).squeeze(-1)).argmax(1)

    def arrays(self):return list(self.model.state_dict().values())+[self.projection,self.gram,self.cross,self.head,self.links]
    def bytes(self):return tensor_bytes(self.arrays())+optimizer_bytes(self.opt)


def direct_bytes(width):return 784*width*4+8*(width+1)**2+16*(width+1)*10
def fly_bytes(width):return 784*width*4+width*10*12+10*8


def width_for(fn):
    width=1
    while fn(width+1)<=BUDGET:width+=1
    return width


class Direct:
    def __init__(self,seed,settings):
        self.width=width_for(direct_bytes);rng=torch.Generator().manual_seed(seed+10000)
        self.projection=torch.randn(784,self.width,generator=rng)
        self.gram=torch.zeros(self.width+1,self.width+1,dtype=torch.float64)
        self.cross=torch.zeros(self.width+1,10,dtype=torch.float64)
        self.head=torch.zeros(self.width+1,10,dtype=torch.float64)
        self.settings=settings;self.presentations=0;self.replays=0

    def features(self,x):
        z=torch.tanh(self.settings['scale']*(F.normalize(x,dim=1)@self.projection))
        return torch.cat([z,torch.ones(len(z),1)],1)

    @torch.no_grad()
    def learn(self,x,y):
        z=self.features(x).double();self.gram.add_(z.T@z);self.cross.add_(z.T@F.one_hot(y,10).double())
        self.presentations+=len(x)

    @torch.no_grad()
    def refresh(self):
        penalty=torch.eye(self.width+1,dtype=torch.float64)*self.settings['ridge'];penalty[-1,-1]=0
        self.head.copy_(torch.linalg.solve(self.gram+penalty,self.cross))

    @torch.no_grad()
    def predict(self,x):return (self.features(x)@self.head.float()).argmax(1)
    def arrays(self):return [self.projection,self.gram,self.cross,self.head]
    def bytes(self):return tensor_bytes(self.arrays())


class Replay:
    def __init__(self,seed,settings):
        torch.manual_seed(seed);self.model=make('mlp')
        self.opt=torch.optim.Adam(self.model.parameters(),lr=settings['lr'])
        # Adam first/second moments and four scalar step tensors included.
        self.capacity=(BUDGET-3*75535*4-16)//789
        self.images=np.zeros((self.capacity,784),dtype='uint8')
        self.labels=np.zeros(self.capacity,dtype='uint8')
        self.priorities=np.full(self.capacity,-1,dtype='float32')
        self.rng=np.random.default_rng(seed+10000);self.settings=settings
        self.presentations=0;self.replays=0

    def learn(self,x,y):
        used=np.flatnonzero(self.priorities>=0)
        if len(used):
            ix=self.rng.choice(used,128,replace=True)
            rx=torch.from_numpy(self.images[ix].astype('float32'))/255;ry=torch.from_numpy(self.labels[ix].astype('int64'))
            bx=torch.cat([x,rx]);by=torch.cat([y,ry]);self.replays+=len(rx)
        else:bx=x;by=y
        self.opt.zero_grad(set_to_none=True);loss=F.cross_entropy(self.model(bx),by)
        loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),5);self.opt.step()
        priorities=np.concatenate([self.priorities,self.rng.random(len(x)).astype('float32')])
        keep=np.argpartition(priorities,-self.capacity)[-self.capacity:]
        self.images=np.concatenate([self.images,(x.numpy()*255).round().astype('uint8')])[keep]
        self.labels=np.concatenate([self.labels,y.numpy().astype('uint8')])[keep]
        self.priorities=priorities[keep];self.presentations+=len(x)

    def refresh(self):pass
    @torch.no_grad()
    def predict(self,x):return self.model(x).argmax(1)
    def arrays(self):return list(self.model.state_dict().values())
    def bytes(self):return tensor_bytes(self.arrays())+optimizer_bytes(self.opt)+self.images.nbytes+self.labels.nbytes+self.priorities.nbytes


class SparseAssociative:
    def __init__(self,seed,settings):
        self.width=width_for(fly_bytes);rng=torch.Generator().manual_seed(seed+10000)
        self.projection=torch.randn(784,self.width,generator=rng)
        self.sums=torch.zeros(10,self.width,dtype=torch.float64)
        self.head=torch.zeros(10,self.width)
        self.counts=torch.zeros(10,dtype=torch.float64)
        self.settings=settings;self.presentations=0;self.replays=0

    def features(self,x):
        z=F.relu(F.normalize(x,dim=1)@self.projection)
        k=max(1,int(self.width*self.settings['sparsity']))
        values,indices=z.topk(k,dim=1)
        sparse=torch.zeros_like(z).scatter_(1,indices,values)
        return F.normalize(sparse,dim=1)

    @torch.no_grad()
    def learn(self,x,y):
        z=self.features(x);self.sums.index_add_(0,y,z.double());self.counts.add_(torch.bincount(y,minlength=10))
        self.presentations+=len(x)

    @torch.no_grad()
    def refresh(self):self.head.copy_(F.normalize(self.sums,dim=1).float())
    @torch.no_grad()
    def predict(self,x):return (self.features(x)@self.head.T).argmax(1)
    def arrays(self):return [self.projection,self.sums,self.head,self.counts]
    def bytes(self):return tensor_bytes(self.arrays())


BUILD={'branch':BranchRouter,'direct':Direct,'replay':Replay,'sparse_associative':SparseAssociative}


@torch.no_grad()
def score(model,x,y,pairs):
    snapshots=[v.clone() for v in model.arrays()]
    pred=torch.cat([model.predict(bx) for bx in x.split(512)])
    assert all(torch.equal(a,b) for a,b in zip(snapshots,model.arrays()))
    perpair=[float((pred[(y==a)|(y==b)]==y[(y==a)|(y==b)]).float().mean()) for a,b in pairs]
    return {'accuracy':float((pred==y).float().mean()),'pairs':perpair}


def fit(arm,seed,settings,x,y,mask,pairs,ex,ey,stages=True):
    model=BUILD[arm](seed,settings);results=[];elapsed=0.;cpu=0.;refresh=0.;updates=0;history=[]
    for stage,pair in enumerate(pairs):
        ix=torch.nonzero(mask&((y==pair[0])|(y==pair[1])),as_tuple=True)[0]
        start=time.perf_counter();cstart=time.process_time()
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            order=ix[torch.randperm(len(ix),generator=rng)]
            for ids in order.split(128):model.learn(x[ids],y[ids]);updates+=1
        elapsed+=time.perf_counter()-start;cpu+=time.process_time()-cstart
        if arm=='branch':history.append(model.model.gates().argmax(-1).tolist())
        if stages or stage==len(pairs)-1:
            start=time.perf_counter();model.refresh();refresh+=time.perf_counter()-start
            results.append(score(model,ex,ey,pairs))
    assert model.bytes()<=BUDGET
    final=results[-1]
    out={'accuracy':final['accuracy'],'old_accuracy':st.mean(final['pairs'][:-1]),'new_accuracy':final['pairs'][-1],
         'stages':results,'training_seconds':elapsed,'training_cpu_seconds':cpu,'head_refresh_seconds':refresh,
         'numeric_bytes':model.bytes(),'current_presentations':model.presentations,'replay_presentations':model.replays,'updates':updates}
    if stages:out['forgetting']=st.mean(max(results[s]['pairs'][i] for s in range(i,len(results)))-final['pairs'][i] for i in range(4))
    if arm=='branch':out['historical_routes_analysis_only']=history
    return model,out


def latency(model,x):
    bx=x[:128]
    for _ in range(10):model.predict(bx)
    times=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(30):model.predict(bx)
        times.append((time.perf_counter()-start)*1e6/30)
    return st.median(times)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'budget_bytes':BUDGET,'datasets':['mnist','fashion_mnist'],'dev_seeds':[130,140],'confirmation_seeds':[[131,132,133],[141,142,143]],
              'arms':list(BUILD),'orders':['forward','reverse'],
              'matching':'All continuing-learning numeric weights, buffers, optimizer tensors, stored examples, and cached heads count. Four augmented methods within1% of4.206MB. Excludes Python/RNG scalar state, allocator/transient activations, datasets and validation apparatus; not peak RAM. Input access raw784pixels for all, branch router also fixed pooled196cue. SAME current stream and two current-pair epochs. No task IDs/boundary writes/testfeedback/output masking. Replay has extra old presentations, explicitly reported.',
              'branch':'Learned24x4 model, original expensive counterfactual supervision, online actual-route analytic consolidation512cues. All writes inherited, select ridge/gamma per dataset on training validation only. Optimizer included.',
              'direct':'Fixed normalized raw784input random tanh features. Maximal width fitting budget; analytic10class head. Select scale and ridge on validation. No pretraining, old image archive or learned routing.',
              'replay':'Same75,535-parameter MLP with maximal-budget uint8image/label/priority random reservoir.128current plus128sampled old examples each update; Adam clipping5. LR selected on validation.',
              'sparse_associative':'Simplified control inspired by FlyModel, NOT an exact reproduction of published FlyModel: Gaussian fixed expansion, top-k positive activations, per-class accumulated normalized codes and cosine head. Select sparsity on validation. No covariance/replay/pretraining/taskID. Class-conditioned synaptic sums are persistent memory.',
              'development':'Separate50k/10k stratified training split, one devseed each; all settings selected before corresponding officialtest evaluation. All candidates saved.3freshseeds x2orders eachdataset. Order repetitions paired within seeds.',
              'compute':'CPU2threads; training acquisition, readout refresh and inference latency separately reported. Equal current examples and storage, NOT equal walltime/FLOPs. No claim compute-matched until dedicated time-budget follow-up.',
              'decision_gate':'Branch accuracy must exceed separately tuned equally budgeted strong controls on BOTH datasets before scaling as a performance-advantage contribution. Passing this engineering gate alone would still not establish novelty or top-conference readiness.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    grids={'branch':[{'ridge':r,'gamma':g} for r in [.1,1.,10.] for g in [0.,.5]],
           'direct':[{'ridge':r,'scale':s} for r in [.1,1.,10.] for s in [.5,1.,2.]],
           'replay':[{'lr':r} for r in [.0003,.001,.003]],
           'sparse_associative':[{'sparsity':s} for s in [.01,.05,.1]]}
    rows=[];selections={}
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['dev_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;selected={};candidates=[]
        for arm in BUILD:
            for settings in grids[arm]:
                _,result=fit(arm,seed,settings,x,y,mask,list(SPLITS),x[val],y[val],False)
                candidates.append({'arm':arm,'settings':settings,'accuracy':result['accuracy']})
            best=max([r for r in candidates if r['arm']==arm],key=lambda r:r['accuracy']);selected[arm]=best['settings']
            print('dev',dataset,arm,round(best['accuracy']*100,2),best['settings'],flush=True)
        selections[dataset]={'selected':selected,'candidates':candidates};(out/'selection.json').write_text(json.dumps(selections,indent=2))
        for order in protocol['orders']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for si,seed in enumerate(protocol['confirmation_seeds'][di]):
                arms=list(BUILD);offset=(si+(order=='reverse'))%len(arms);arms=arms[offset:]+arms[:offset]
                for arm in arms:
                    model,result=fit(arm,seed,selected[arm],x,y,torch.ones(len(y),dtype=torch.bool),pairs,tx,ty)
                    row={'dataset':dataset,'seed':seed,'order':order,'arm':arm,'settings':selected[arm],**result,'inference128_us':latency(model,tx)}
                    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
                    print(dataset,order,seed,arm,round(row['accuracy']*100,2),'bytes',row['numeric_bytes'],flush=True)
                    if si==0 and order=='forward' and arm=='branch':
                        torch.save({'backbone':model.model.state_dict(),'projection':model.projection,'gram':model.gram,'cross':model.cross,'head':model.head,'links':model.links},out/f'{dataset}_branch.pt')
    lines=['# Strong-control gate for branch retrieval','','Matched ongoing numeric storage, including optimizer state. Same current data stream; measured unequal compute. Neither image dataset is sufficient alone for a major-conference claim.','',
           '| Dataset | Method | Accuracy ± SD across six paired runs | Old pairs | Forgetting | Numeric MB | Train + refresh seconds | Inference128 µs |','|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in protocol['datasets']:
        for arm in BUILD:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm];a=[100*r['accuracy'] for r in rs]
            lines.append(f'| {dataset} | {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {100*st.mean(r["old_accuracy"] for r in rs):.2f}% | {100*st.mean(r["forgetting"] for r in rs):.2f} pp | {rs[0]["numeric_bytes"]/1e6:.3f} | {st.mean(r["training_seconds"]+r["head_refresh_seconds"] for r in rs):.3f} | {st.mean(r["inference128_us"] for r in rs):.1f} |')
    passed=all(st.mean(r['accuracy'] for r in rows if r['dataset']==d and r['arm']=='branch')>max(st.mean(r['accuracy'] for r in rows if r['dataset']==d and r['arm']==a) for a in BUILD if a!='branch') for d in protocol['datasets'])
    lines+=['',f'**Predeclared performance-advantage gate passed: {passed}.**', '',
            'Three independent seeds per dataset, each tested with forward and reverse orders. Pooled six-run SD is descriptive; paired order repeats are not independent seeds. All current models see120,000training presentations; replay sees additional buffered examples. Direct feature maps are fixed and analytic heads are order-independent apart from finite precision, so their repeated final scores across orders are not six independent confirmations.', '',
            'No total-FLOP/time match is claimed. Stronger controls receive independently selected hyperparameters on the same held-out training split. Sparse associative control is explicitly simplified and not a reproduction of published FlyModel. Branch model uses a global covariance router and expensive counterfactual backbone loss; retention cannot be assigned solely to branch memory.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
