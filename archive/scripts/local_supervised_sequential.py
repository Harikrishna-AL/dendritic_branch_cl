"""Minimal sequential Fashion-MNIST: local vs shared supervision vs random.

No routing, recruitment, protection, task IDs or replay. Training heads persist
across all classes and are discarded for prototype-based evaluation.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS
from bdh_cl.scripts.feature_learning_clean import softmax_gradient

class Learner:
    def __init__(self,seed,arm):
        self.arm=arm;self.w=np.random.default_rng(seed+20).normal(0,.1,(197,64))
        self.head=np.zeros((65,10));self.local_heads=np.zeros((8,9,10))
        self.proto=np.zeros((10,64));self.count=np.zeros(10,int)
    def features(self,x):return np.tanh(x@self.w)
    def observe(self,x,y):
        h=self.features(x)
        if self.arm=='shared':
            hb=np.column_stack([h,np.ones(len(h))]);error=softmax_gradient(hb@self.head,y)
            dw=x.T@((error@self.head[:-1].T)*(1-h*h));dh=hb.T@error
            self.w-=.1*dw;self.head-=.1*dh
        elif self.arm=='local':
            groups=h.reshape(len(h),8,8)
            gb=np.concatenate([groups,np.ones((len(h),8,1))],axis=2)
            logits=np.einsum('nbk,bkc->nbc',gb,self.local_heads)
            p=np.exp(logits-logits.max(2,keepdims=True));p/=p.sum(2,keepdims=True)
            p[np.arange(len(y))[:,None],np.arange(8)[None],y[:,None]]-=1;p/=len(y)
            local=np.einsum('nbc,bkc->nbk',p,self.local_heads[:,:8])*(1-groups*groups)
            dw=x.T@local.reshape(len(y),64);dh=np.einsum('nbk,nbc->bkc',gb,p)
            self.w-=.1*dw;self.local_heads-=.1*dh
        h=self.features(x)
        for c in np.unique(y):
            selected=y==c;n=int(selected.sum());total=self.count[c]+n
            self.proto[c]=(self.proto[c]*self.count[c]+h[selected].sum(0))/total;self.count[c]=total
    def predict(self,x):
        h=self.features(x);classes=np.flatnonzero(self.count)
        distance=((h[:,None]-self.proto[classes])**2).sum(2)
        return classes[distance.argmin(1)]

def main():
    raw=load_raw('data_cache/fashion_mnist')
    # Constant centering avoids estimating normalization from future classes.
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255-.5
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255-.5
    x=np.column_stack([x,np.ones(len(x))]);tx=np.column_stack([tx,np.ones(len(tx))])
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        ix=np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==c),1000,replace=False) for c in range(10)])
        data=[ix[np.isin(raw['train_labels'][ix],pair)] for pair in SPLITS]
        for arm in ['random','shared','local']:
            m=Learner(seed,arm);initial=m.w.copy();accuracy=[];pairs=[];saved={};drift=[];confusions=[]
            for stage,indices in enumerate(data):
                oldclasses=np.flatnonzero(m.count);oldp=m.proto.copy()
                for epoch in range(10):
                    order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(indices)
                    for batch in np.array_split(order,32):m.observe(x[batch],raw['train_labels'][batch])
                assert np.array_equal(oldp[oldclasses],m.proto[oldclasses])
                if arm=='random':assert np.array_equal(initial,m.w)
                pred=m.predict(tx);seen=raw['test_labels']<2*(stage+1)
                accuracy.append(float(np.mean(pred[seen]==raw['test_labels'][seen])))
                pairs.append([float(np.mean(pred[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
                h=m.features(tx);current={c:h[raw['test_labels']==c].mean(0) for c in range(2*(stage+1))}
                for c in SPLITS[stage]:saved[c]=current[c].copy()
                drift.append({str(c):float(np.mean((current[c]-saved[c])**2)) for c in current})
                cm=np.zeros((10,10),int);np.add.at(cm,(raw['test_labels'],pred),1);confusions.append(cm.tolist())
            runs.append({'seed':seed,'arm':arm,'accuracy':accuracy,'pairs':pairs,'feature_drift':drift,'confusions':confusions})
            print(seed,arm,accuracy,'pairs',pairs[-1],flush=True)
    summary=[]
    for arm in ['random','shared','local']:
        group=[r for r in runs if r['arm']==arm]
        summary.append({'arm':arm,'accuracy':float(np.mean([r['accuracy'][-1] for r in group])),
                        'sd':float(np.std([r['accuracy'][-1] for r in group],ddof=1)),
                        'old_pair_accuracy':float(np.mean([r['pairs'][-1][:-1] for r in group])),
                        'new_pair_accuracy':float(np.mean([r['pairs'][-1][-1] for r in group])),
                        'forgetting_pp':100*float(np.mean([max(r['pairs'][k][j] for k in range(j,5))-r['pairs'][-1][j] for r in group for j in range(4)])),
                        'feature_drift':float(np.mean([v for r in group for c,v in r['feature_drift'][-1].items() if int(c)<8]))})
    out=Path('results/local_supervised_sequential');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'protocol':{'dataset':'Fashion-MNIST','seeds':3,'train_per_class':1000,'epochs_per_pair':10,'feature_dimension':64,'normalization':'constant subtract 0.5','task_id':False,'replay':False,'protection':False,'routing':False},'runs':runs,'summary':summary},indent=2))
    lines=['# Minimal locally supervised branches: sequential Fashion-MNIST','','Five pairs 0/1→2/3→4/5→6/7→8/9; three paired seeds; 1000 training examples per class, ten epochs per pair, all 10000 official test images. Constant pixel centering (subtract 0.5), no future-class normalization statistics. Same 14x14 input, initial encoder, 64 features, learning rate 0.1, paired minibatches and prototype readout across arms. No task-ID input, routing, recruitment, freezing, protection or replay. Training heads compete across all ten classes; no masking or resetting at stage boundaries. Prototype inference competes across all seen classes.','','Random features never update. Shared supervision uses one 64-feature ten-class training head. Local supervision uses eight independent 8-feature ten-class heads, each updating its own encoder group. Training heads persist but are never used at evaluation; all features contribute to one prototype vector. Prototype means accumulate online after each encoder update, and old classes are never refreshed. This differs from mixed-class evaluation, which formed prototypes after the representation had finished training. Constant centering also differs from that earlier experiment. Thus absolute mixed/sequential differences are not a pure forgetting estimate; stage retention and feature drift are the direct sequential evidence.','','| Arm | Final accuracy | Seed SD | Old-pair accuracy | Last-pair accuracy | Forgetting pp | Old feature drift |','|---|---|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['arm']} | {100*s['accuracy']:.2f}% | {100*s['sd']:.2f} | {100*s['old_pair_accuracy']:.2f}% | {100*s['new_pair_accuracy']:.2f}% | {s['forgetting_pp']:.2f} | {s['feature_drift']:.6f} |")
    lines+=['','Prototype values remain unchanged for old classes, but embeddings may drift. Forgetting is mean peak-to-final pair accuracy, averaging over first four pairs; more competing classes can lower accuracy even for fixed features. Held-out feature mean drift is diagnostic only, never feedback. Frozen random and old-prototype invariance assertions passed. All class confusions, stage scores and drift values in results.json. This is a pilot; branch-local supervision alone need not create task specialization.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
