"""Local pixel-predictive branch prototype; task-free class-incremental readout.

Eight nonlinear branches predict checkerboard pixels from complementary pixels.
Input-derived prediction errors route and recruit; labels update prototypes only.
Warmup, novelty threshold and freezing are explicit engineered rules.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.reuse_split_mnist import load_raw, downsample, SPLITS

MASK=(np.indices((14,14)).sum(0)%2==0).reshape(-1)

class Branches:
    def __init__(self,seed,arm,threshold):
        rng=np.random.default_rng(seed+20)
        self.encoder=rng.normal(0,.1,(8,99,8));self.decoder=np.zeros((8,8,98))
        self.active=np.zeros(8,bool);self.active[0]=True
        self.age=np.zeros(8,int);self.frozen=np.zeros(8,bool)
        self.arm=arm;self.threshold=threshold;self.events=[];self.step=0
        if arm=='random_features':self.active[:]=True;self.frozen[:]=True
        self.prototypes=np.zeros((10,64));self.count=np.zeros(10,int)

    def responses(self,images):
        x=np.column_stack([images[:,MASK],np.ones(len(images))]);target=images[:,~MASK]
        z=np.tanh(np.einsum('ni,bik->nbk',x,self.encoder))
        predicted=np.einsum('nbk,bkj->nbj',z,self.decoder)
        # More weight on foreground pixels; no digit labels enter this error.
        weight=1+4*target
        errors=np.sum((predicted-target[:,None])**2*weight[:,None],axis=2)/weight.sum(1)[:,None]
        errors[:,~self.active]=np.inf
        return x,target,z,predicted,weight,errors

    def embed(self,images):
        _,_,z,_,_,errors=self.responses(images)
        if self.arm in ['ungated_frozen','random_features']:return (z*self.active[None,:,None]).reshape(len(images),64)
        chosen=errors.argmin(1)
        out=np.zeros_like(z);out[np.arange(len(images)),chosen]=z[np.arange(len(images)),chosen]
        return out.reshape(len(images),64)

    def learn_pixels(self,images):
        if self.arm=='random_features':self.step+=1;return
        x,target,z,predicted,weight,errors=self.responses(images)
        chosen=errors.argmin(1);best=errors.min(1)
        # Recruit only when all existing branches are mature, avoiding duplicates
        # during warmup. No stream-boundary or class signal is available here.
        if self.frozen[self.active].all() and np.median(best)>self.threshold and not self.active.all():
            b=int(np.flatnonzero(~self.active)[0]);self.active[b]=True
            self.events.append({'step':self.step,'branch':b,'median_error':float(np.median(best))})
            chosen[best>self.threshold]=b
        for b in np.unique(chosen):
            ix=chosen==b
            if self.frozen[b] and self.arm!='plastic':continue
            err=2*(predicted[ix,b]-target[ix])*weight[ix]/weight[ix].sum(1)[:,None]
            de=z[ix,b].T@err/ix.sum()
            dz=(err@self.decoder[b].T)*(1-z[ix,b]**2)
            self.encoder[b]-=.2*x[ix].T@dz/ix.sum()
            self.decoder[b]-=.2*de
            self.age[b]+=1
            if self.age[b]>=20:self.frozen[b]=True
        self.step+=1

    def observe(self,images,labels):
        frozen_before=self.frozen.copy();olde=self.encoder[frozen_before].copy();oldd=self.decoder[frozen_before].copy()
        self.learn_pixels(images)
        if self.arm!='plastic':
            assert np.array_equal(olde,self.encoder[frozen_before]) and np.array_equal(oldd,self.decoder[frozen_before])
        h=self.embed(images)
        for c in np.unique(labels):
            ix=labels==c;n=int(ix.sum());total=self.count[c]+n
            self.prototypes[c]=(self.prototypes[c]*self.count[c]+h[ix].sum(0))/total;self.count[c]=total

    def predict(self,images):
        h=self.embed(images);classes=np.flatnonzero(self.count)
        distance=np.sum((h[:,None]-self.prototypes[classes])**2,axis=2)
        return classes[distance.argmin(1)]

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        for arm in ['frozen_reuse','plastic','ungated_frozen','random_features']:
            for threshold in [.03,.06]:
                m=Branches(seed,arm,threshold);accuracy=[];pairs=[];route_changes=[];routes={};pixel_errors=[]
                for stage,ix in enumerate(data):
                    oldp=m.prototypes.copy();oldclasses=np.flatnonzero(m.count)
                    for epoch in range(10):
                        order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                        for batch in np.array_split(order,16):m.observe(x[batch],raw['train_labels'][batch])
                    assert np.array_equal(oldp[oldclasses],m.prototypes[oldclasses])
                    pred=m.predict(tx);seen=raw['test_labels']<2*(stage+1)
                    accuracy.append(float(np.mean(pred[seen]==raw['test_labels'][seen])))
                    pairs.append([float(np.mean(pred[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
                    errors=m.responses(tx)[-1];chosen=errors.argmin(1)
                    change=[]
                    for c in range(2*(stage+1)):
                        current=chosen[raw['test_labels']==c]
                        if c in routes:change.append(float(np.mean(current!=routes[c])))
                        else:routes[c]=current.copy()
                    route_changes.append(float(np.mean(change)) if change else 0)
                    pixel_errors.append(float(errors[seen].min(1).mean()))
                runs.append({'seed':seed,'arm':arm,'threshold':threshold,'seen_accuracy':accuracy,'pairs':pairs,
                             'route_changes':route_changes,'pixel_mse':pixel_errors,'active':int(m.active.sum()),'events':m.events})
                print(seed,arm,threshold,accuracy,'branches',int(m.active.sum()),flush=True)
    summary=[]
    for arm in ['frozen_reuse','plastic','ungated_frozen','random_features']:
        for threshold in [.03,.06]:
            group=[r for r in runs if r['arm']==arm and r['threshold']==threshold]
            summary.append({'arm':arm,'threshold':threshold,'final_accuracy':float(np.mean([r['seen_accuracy'][-1] for r in group])),
                            'sd':float(np.std([r['seen_accuracy'][-1] for r in group],ddof=1)),
                            'branches':float(np.mean([r['active'] for r in group])),
                            'old_route_change':float(np.mean([r['route_changes'][-1] for r in group]))})
    out=Path('results/predictive_branches_mnist');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summary},indent=2))
    lines=['# Local predictive branches on class-incremental MNIST','','Three seeds, five sequential digit pairs, 500 training images per digit, ten epochs per pair, all official test images. No task IDs, pair output masks or replay. Labels update class prototypes only. Eight branch slots, each an 8-dimensional trainable tanh encoder from 98 checkerboard pixels plus bias and linear decoder to the other 98 pixels. Fixed 64-dimensional concatenated representation.','','Branches minimize foreground-weighted pixel prediction MSE. Lowest-error branch routes each image; error uses only pixels of the same observed image, not a class label. Mature branches freeze after 20 assigned minibatches. Median best error exceeding 0.03 or 0.06 recruits a spare branch when existing branches are mature. Rules are engineered; no task-boundary input. At inference all image pixels are observed, permitting evaluation of held-out prediction errors. Unused slots have zero features. Newly active branches can steal old inputs; no claim of stable routing.','','Frozen reuse updates immature branches only. Plastic uses the same recruitment/routing but keeps updating mature branches. Ungated frozen uses all active branch latents, retaining the same predictor training and recruitment; it removes winner-based feature gating. Nearest cumulative class prototype competes across all seen digits. Prototypes are not refreshed on old classes; warmup embeddings contribute to their means.','','| Arm | Threshold | Final accuracy | Seed SD | Branches | Old route changes |','|---|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['arm']} | {s['threshold']} | {100*s['final_accuracy']:.2f}% | {100*s['sd']:.2f} | {s['branches']:.2f} | {100*s['old_route_change']:.2f}% |")
    lines+=['','This is the first pixel-predictive branch implementation, not biological evidence. Pixel reconstruction need not identify digit-discriminative contexts. Fixed slots do not by themselves guarantee fixed embeddings: weights, route selection and slot activation all matter. Frozen weight and old prototype invariance assertions run throughout. Full stage curves, recruitment events and per-pair accuracy are in results.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
