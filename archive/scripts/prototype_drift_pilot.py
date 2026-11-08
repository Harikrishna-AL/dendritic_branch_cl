"""Class-incremental prototype readout: frozen/shared/subspace-protected features.

Protected is an engineered input-direction constraint, not biological dendrites.
No task ID, replay, classifier parameters, feature growth, or boundary updates.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.reuse_split_mnist import load_raw, downsample, SPLITS

class Learner:
    def __init__(self,arm,seed):
        self.arm=arm;self.w=np.random.default_rng(seed+20).normal(0,.1,(197,64))
        self.prototypes=np.zeros((10,64));self.count=np.zeros(10,dtype=int)
        self.basis=np.zeros((197,24));self.rank=0

    def embed(self,x):return np.tanh(x@self.w)

    def predict(self,x):
        classes=np.flatnonzero(self.count)
        h=self.embed(x)
        scores=-np.sum((h[:,None]-self.prototypes[classes])**2,axis=2)
        return classes[scores.argmax(1)]

    def observe(self,x,y):
        # Bootstrap a new class only from its labeled observations.
        h=self.embed(x)
        for c in np.unique(y):
            if self.count[c]==0:self.prototypes[c]=h[y==c].mean(0)
        classes=np.union1d(np.flatnonzero(self.count),np.unique(y))
        p=self.prototypes[classes]
        scores=-np.sum((h[:,None]-p)**2,axis=2)/4
        prob=np.exp(scores-scores.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        target=np.searchsorted(classes,y)
        predicted=classes[prob.argmax(1)]
        error=2*(prob@p-p[target])/4
        gradient=x.T@(error*(1-h*h))/len(x)
        if self.arm!='frozen':
            if self.arm=='protected':
                q=self.basis[:,:self.rank]
                gradient-=q@(q.T@gradient)
            self.w-=.03*gradient
        # Historical prototypes are never refreshed using old examples.
        after=self.embed(x)
        for c in np.unique(y):
            chosen=y==c;n=int(chosen.sum());total=self.count[c]+n
            self.prototypes[c]=(self.count[c]*self.prototypes[c]+after[chosen].sum(0))/total
            self.count[c]=total
        if self.arm=='protected' and self.rank<24:
            # Protect one correctly classified mean input direction per class/batch.
            for c in np.unique(y):
                selected=(y==c)&(predicted==y)
                if not selected.any() or self.rank==24:continue
                v=x[selected].mean(0);norm=np.linalg.norm(v)
                q=self.basis[:,:self.rank]
                for _ in range(2):v-=q@(q.T@v)
                if np.linalg.norm(v)>.05*norm:
                    self.basis[:,self.rank]=v/np.linalg.norm(v);self.rank+=1

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    x=np.column_stack([x,np.ones(len(x))]);tx=np.column_stack([tx,np.ones(len(tx))])
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        indices=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        for arm in ['frozen','shared','protected']:
            model=Learner(arm,seed);initial=model.w.copy();matrix=[];accuracies=[];drift=[];saved={};snapshots=[]
            for stage,ix in enumerate(indices):
                oldp=model.prototypes.copy();oldclasses=np.flatnonzero(model.count)
                for epoch in range(10):
                    order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                    for batch in np.array_split(order,16):model.observe(x[batch],raw['train_labels'][batch])
                assert np.array_equal(model.prototypes[oldclasses],oldp[oldclasses]),'Old prototype edited'
                if arm=='frozen':assert np.array_equal(initial,model.w)
                if model.rank:
                    q=model.basis[:,:model.rank];assert np.allclose(q.T@q,np.eye(model.rank),atol=1e-9)
                pred=model.predict(tx);seen=raw['test_labels']<2*(stage+1)
                matrix.append([float(np.mean(pred[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
                accuracies.append(float(np.mean(pred[seen]==raw['test_labels'][seen])))
                embedding=model.embed(tx)
                means={c:embedding[raw['test_labels']==c].mean(0) for c in range(2*(stage+1))}
                for c in SPLITS[stage]:saved[c]=means[c].copy()
                drift.append({str(c):float(np.mean((means[c]-saved[c])**2)) for c in means})
                snapshots.append(model.rank)
            runs.append({'seed':seed,'arm':arm,'accuracy_matrix':matrix,'seen_accuracy':accuracies,'test_mean_feature_drift':drift,'protection_rank':snapshots})
            print(seed,arm,accuracies,'rank',model.rank,flush=True)
    summaries=[]
    for arm in ['frozen','shared','protected']:
        group=[r for r in runs if r['arm']==arm]
        summaries.append({'arm':arm,'final_accuracy':float(np.mean([r['seen_accuracy'][-1] for r in group])),
            'sd':float(np.std([r['seen_accuracy'][-1] for r in group],ddof=1)),
            'old_feature_drift':float(np.mean([v for r in group for c,v in r['test_mean_feature_drift'][-1].items() if int(c)<8])),
            'final_pairs':np.mean([r['accuracy_matrix'][-1] for r in group],axis=0).tolist()})
    out=Path('results/prototype_drift_pilot');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summaries},indent=2))
    lines=['# Class-incremental prototype drift pilot','','Same five sequential pairs, three seeds, 500 train images per digit, ten epochs and all official test images. 14x14 input, float64. Single nearest-prototype prediction among ALL seen digit classes; no pair selection or task-ID input. No learned classifier weights, replay, feature growth or task-boundary updates. Fixed 64-dimensional tanh features.','','Prototypes are cumulative means of features observed during training (including repeated epochs). Old classes receive no updates after their examples leave the stream. Shared/protected features train using distance-softmax cross entropy, treating prototypes as detached targets. Temperature 4, learning rate 0.03 fixed before run. Frozen uses initial random features. Protected projects feature-weight updates away from up to 24 online input-mean directions; directions are from correctly classified examples. This is a generic subspace-protection baseline, not an implementation of branch gating. Basis and prototypes are summary memory, not stored replay examples.','','| Arm | Final accuracy | Seed SD | Old feature mean drift |','|---|---|---|---|']
    for s in summaries:lines.append(f"| {s['arm']} | {100*s['final_accuracy']:.2f}% | {100*s['sd']:.2f} | {s['old_feature_drift']:.6f} |")
    lines+=['','Checks: old prototype values unchanged between stages; frozen feature weights unchanged; protection basis orthonormal. Feature drift uses test class means only as a reporting diagnostic; test data never updates learner. New prototypes can compete with old ones even if all embeddings are fixed, so declining accuracy need not mean representation forgetting. Prototypes of trainable features average across changing coordinate systems; old means can be stale. Protection stores extra basis memory (197x24 doubles) and can saturate early. Three seeds and subsampled training are a pilot, not a full benchmark.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
