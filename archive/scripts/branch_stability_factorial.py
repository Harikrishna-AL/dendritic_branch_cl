"""Factorial feature/router drift and streaming-only prototype timing test.

All 8 slots available from the start to remove slot activation as a confound.
Fixed router is random input-based selection; learned router predicts local
reconstruction errors and freezes at global batch 100, never a task boundary.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.predictive_branches_mnist import MASK
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS

class Model:
    def __init__(self,seed,learn_features,learn_router):
        rng=np.random.default_rng(seed+20)
        self.w=rng.normal(0,.1,(8,99,8));self.d=np.zeros((8,8,98))
        self.router=rng.normal(0,.02,(99,8));self.age=np.zeros(8,int);self.step=0
        self.learn_features=learn_features;self.learn_router=learn_router
    def response(self,images):
        x=np.column_stack([images[:,MASK],np.ones(len(images))])
        z=np.tanh(np.einsum('ni,bik->nbk',x,self.w))
        pred=np.einsum('nbk,bkj->nbj',z,self.d);t=images[:,~MASK];weight=1+4*t
        errors=((pred-t[:,None])**2*weight[:,None]).sum(2)/weight.sum(1)[:,None]
        return x,z,pred,t,weight,errors
    def route(self,x):return (x@self.router).argmin(1)
    def embed(self,images):
        x,z,*_=self.response(images);route=self.route(x)
        h=np.zeros_like(z);h[np.arange(len(images)),route]=z[np.arange(len(images)),route]
        return h.reshape(len(images),64),route
    def update(self,images):
        x,z,pred,t,weight,errors=self.response(images);route=self.route(x)
        beforew=self.w.copy();beforer=self.router.copy();frozen=self.age>=20
        for b in np.unique(route):
            ix=route==b
            if self.age[b]>=20:continue
            err=2*(pred[ix,b]-t[ix])*weight[ix]/weight[ix].sum(1)[:,None]
            de=z[ix,b].T@err/ix.sum();dz=(err@self.d[b].T)*(1-z[ix,b]**2)
            if self.learn_features:self.w[b]-=.2*x[ix].T@dz/ix.sum()
            self.d[b]-=.2*de;self.age[b]+=1
        if self.learn_router and self.step<100:
            self.router-=.005*x.T@(x@self.router-errors)/len(x)
        self.step+=1
        assert np.array_equal(beforew[frozen],self.w[frozen])
        if not self.learn_features:assert np.array_equal(beforew,self.w)
        if not self.learn_router or self.step>100:assert np.array_equal(beforer,self.router)
    def stable(self,route):
        router_stable=not self.learn_router or self.step>=100
        return (self.age[route]>=20 if self.learn_features else np.ones(len(route),bool))&router_stable

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        for lf in [False,True]:
            for lr in [False,True]:
                m=Model(seed,lf,lr);p={a:np.zeros((10,64)) for a in ['immediate','stable_only']}
                count={a:np.zeros(10,int) for a in p};seen=np.zeros(10,bool)
                logs={a:{'seed':seed,'learn_features':lf,'learn_router':lr,'prototype_timing':a,
                         'accuracy':[],'missing_classes':[],'counts':[],'feature_drift':[],'route_change':[]} for a in p}
                savedh={};savedroute={}
                for stage,ix in enumerate(data):
                    oldclasses=np.flatnonzero(seen);oldp={a:p[a].copy() for a in p}
                    for epoch in range(10):
                        order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                        for batch in np.array_split(order,16):
                            images=x[batch];labels=raw['train_labels'][batch]
                            m.update(images);h,route=m.embed(images);seen[np.unique(labels)]=True
                            for a in p:
                                eligible=np.ones(len(batch),bool) if a=='immediate' else m.stable(route)
                                for c in np.unique(labels):
                                    selected=(labels==c)&eligible;n=int(selected.sum())
                                    if not n:continue
                                    total=count[a][c]+n;p[a][c]=(count[a][c]*p[a][c]+h[selected].sum(0))/total;count[a][c]=total
                    for a in p:assert np.array_equal(p[a][oldclasses],oldp[a][oldclasses])
                    h,route=m.embed(tx);testseen=seen[raw['test_labels']]
                    change=[];drift=[]
                    for c in np.flatnonzero(seen):
                        selected=raw['test_labels']==c
                        if c in savedh:
                            change.append(float(np.mean(route[selected]!=savedroute[c])))
                            drift.append(float(np.mean((h[selected]-savedh[c])**2)))
                        else:savedh[c]=h[selected].copy();savedroute[c]=route[selected].copy()
                    for a in p:
                        classes=np.flatnonzero(seen)
                        distance=((h[:,None]-p[a][classes])**2).sum(2)
                        distance[:,count[a][classes]==0]=np.inf
                        prediction=classes[distance.argmin(1)] if np.any(count[a]) else np.full(len(tx),-1)
                        logs[a]['accuracy'].append(float(np.mean(prediction[testseen]==raw['test_labels'][testseen])))
                        logs[a]['missing_classes'].append(classes[count[a][classes]==0].tolist())
                        logs[a]['counts'].append(count[a].tolist())
                        logs[a]['feature_drift'].append(float(np.mean(drift)) if drift else 0)
                        logs[a]['route_change'].append(float(np.mean(change)) if change else 0)
                for a in p:runs.append(logs[a])
                print(seed,'features',lf,'router',lr,{a:logs[a]['accuracy'][-1] for a in p},flush=True)
    summary=[]
    for lf in [False,True]:
        for lr in [False,True]:
            for timing in ['immediate','stable_only']:
                group=[r for r in runs if r['learn_features']==lf and r['learn_router']==lr and r['prototype_timing']==timing]
                summary.append({'learn_features':lf,'learn_router':lr,'timing':timing,
                                'accuracy':float(np.mean([r['accuracy'][-1] for r in group])),
                                'sd':float(np.std([r['accuracy'][-1] for r in group],ddof=1)),
                                'old_feature_drift':float(np.mean([r['feature_drift'][-1] for r in group])),
                                'old_route_change':float(np.mean([r['route_change'][-1] for r in group])),
                                'missing':sum(len(r['missing_classes'][-1]) for r in group)})
    out=Path('results/branch_stability_factorial');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summary},indent=2))
    lines=['# Feature/router stability and prototype timing factorial','','Three paired seeds, same sequential five pairs, 500 training images per digit, ten epochs, all 10000 test images. No task ID, replay or old-prototype rebuilding. Prototype readout competes across all seen digits. Missing prototypes cannot predict their class; missing classes remain in the accuracy denominator. Eight 8-dimensional branch slots are available from the start in every condition, removing recruitment/activation as a confound. This differs from the earlier growing model.','','2x2 factorial: feature encoders either remain random/fixed or train via local pixel prediction and freeze after 20 assigned batches. Decoders train for the same 20 batches in both cases. Router either remains a fixed random input-based selector or learns to predict local errors for the first 100 global batches, then freezes. Freeze does not use class/task boundaries. Router governs predictor assignments in this test. No digit labels in predictor/router learning. Fixed router is a diagnostic control, not an error-trained solution.','','Two prototype policies share the same model stream: immediate accumulation, or accumulation only when the selected encoder and entire router are frozen. No reset or refreshing of old prototypes; delayed examples are discarded, not cached. Branch maturity is engineered. Held-out feature/route drift measured relative to when each class was introduced is diagnostic only, never feedback.','','| Train features | Train router | Prototype policy | Final accuracy | Seed SD | Old feature drift | Old route changes | Missing final class instances |','|---|---|---|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['learn_features']} | {s['learn_router']} | {s['timing']} | {100*s['accuracy']:.2f}% | {100*s['sd']:.2f} | {s['old_feature_drift']:.6f} | {100*s['old_route_change']:.2f}% | {s['missing']} |")
    lines+=['','Assertions check encoder/router freezing and no old-class prototype edits. Immediate/stable-only comparisons are paired within each stream. These are diagnostic ablations, not direct replacements for the prior recruitment experiment. Class introduction happened after the learned router freeze for most later pairs, so later old-class drift can be zero even when first-pair routing changed. Final drift averages over the first eight classes. Full counts, missing-class history and stage curves in results.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
