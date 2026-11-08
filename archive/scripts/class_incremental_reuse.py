"""Single ten-class prediction, task-free learner API; no replay or pair masks.

The runner arranges digit-pair streams, but the learner gets only (x,y).
Growth is driven by loss persistence, with fixed age/capacity constraints.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.reuse_split_mnist import load_raw, downsample, SPLITS

class Learner:
    def __init__(self,arm,seed):
        self.arm=arm;self.rng=np.random.default_rng(seed+20)
        self.w=self.rng.normal(0,.1,(197,64));self.head=np.zeros((64,10))
        self.step=0;self.age=0;self.ema=None;self.bad=0;self.events=[];self.start=0

    def logits(self,x):return np.tanh(x@self.w)@self.head

    def observe(self,x,y):
        h=np.tanh(x@self.w);logits=h@self.head
        logp=logits-logits.max(1,keepdims=True)
        logp-=np.log(np.exp(logp).sum(1,keepdims=True))
        loss=float(-logp[np.arange(len(y)),y].mean())
        p=np.exp(logp);p[np.arange(len(y)),y]-=1;p/=len(y)
        dw=x.T@((p@self.head.T)*(1-h*h));dh=h.T@p
        if self.arm=='shared':self.w-=.15*dw;self.head-=.15*dh
        elif self.arm=='frozen_features':self.head-=.15*dh
        else:
            oldw=self.w[:,:self.start].copy();oldh=self.head[:self.start].copy()
            self.w[:,self.start:]-=.15*dw[:,self.start:]
            self.head[self.start:]-=.15*dh[self.start:]
            assert np.array_equal(oldw,self.w[:,:self.start]) and np.array_equal(oldh,self.head[:self.start])
        self.step+=1;self.age+=1
        self.ema=loss if self.ema is None else .95*self.ema+.05*loss
        loss_threshold=.3 if self.arm=='growth_sensitive' else 1.0
        self.bad=self.bad+1 if self.ema>loss_threshold else 0
        patience=5 if self.arm=='growth_sensitive' else 20
        if self.arm in ['frozen_growth','growth_sensitive'] and self.age>=100 and self.bad>=patience and self.w.shape[1]<192:
            self.start=self.w.shape[1]
            self.w=np.column_stack([self.w,self.rng.normal(0,.1,(197,32))])
            self.head=np.vstack([self.head,np.zeros((32,10))])
            self.events.append({'step':self.step,'loss_ema':self.ema,'width':self.w.shape[1]})
            self.age=0;self.bad=0
        return loss

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    x=np.column_stack([x,np.ones(len(x))]);tx=np.column_stack([tx,np.ones(len(tx))])
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        indices=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        for arm in ['shared','frozen_features','frozen_growth','growth_sensitive']:
            model=Learner(arm,seed);matrix=[];seen_accuracy=[];confusions=[];boundaries=[]
            for stage,ix in enumerate(indices):
                for epoch in range(10):
                    order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                    for batch in np.array_split(order,int(np.ceil(len(order)/64))):model.observe(x[batch],raw['train_labels'][batch])
                # Ten output options always, including unseen digits; never mask logits.
                logits=model.logits(tx)
                assert logits.shape==(10000,10) and np.isfinite(logits).all()
                pred=logits.argmax(1)
                values=[float(np.mean(pred[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]]
                matrix.append(values)
                seen=np.isin(raw['test_labels'],np.arange(2*(stage+1)))
                seen_accuracy.append(float(np.mean(pred[seen]==raw['test_labels'][seen])))
                cm=np.zeros((10,10),dtype=int);np.add.at(cm,(raw['test_labels'],pred),1);confusions.append(cm.tolist())
                boundaries.append(model.step)
            runs.append({'seed':seed,'arm':arm,'accuracy_matrix':matrix,'seen_accuracy':seen_accuracy,
                         'confusion_matrices':confusions,'growth_events':model.events,'runner_boundaries':boundaries,'width':model.w.shape[1]})
            print(seed,arm,'accuracy',seen_accuracy,'growth',model.events,flush=True)
    summary=[]
    for arm in ['shared','frozen_features','frozen_growth','growth_sensitive']:
        group=[r for r in runs if r['arm']==arm]
        finals=[r['seen_accuracy'][-1] for r in group]
        summary.append({'arm':arm,'final_accuracy':float(np.mean(finals)),'final_sd':float(np.std(finals,ddof=1)),
                        'forgetting':float(np.mean([max(r['accuracy_matrix'][k][j] for k in range(j,5))-r['accuracy_matrix'][-1][j] for r in group for j in range(4)])),
                        'final_per_pair':np.mean([r['accuracy_matrix'][-1] for r in group],axis=0).tolist(),
                        'mean_width':float(np.mean([r['width'] for r in group]))})
    out=Path('results/class_incremental_reuse');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'protocol':{'task_id':False,'output_masking':False,'outputs':10,'replay':False,'seeds':3,'train_per_digit':500,'epochs':10,'resolution':'14x14','growth':'loss EMA >1.0 for 20 batches, block age >=100, width cap 192'},'runs':runs,'summary':summary},indent=2))
    lines=['# Class-incremental Split MNIST','','Learner receives only images and digit labels. One ten-class output throughout. No task ID, binary heads, output masks, replay, or task-boundary signal to learner. Runner orders digit pairs; stage labels are used only for reporting. All official test images; seen-class metric still allows all ten predicted labels. Three paired seeds, 500 train images per digit, ten epochs, 14x14 input, float64.','','Shared trains all features and readout. Frozen features keeps its initial random 64 features, trains a shared readout (no first-task privilege). Frozen growth freezes old feature/readout blocks when an error-driven new block is recruited; new block adds scores to the same ten outputs. All branches participate: this does not yet implement learned selective gating. Initial 64, recruitment size 32, cap 192. Growth criterion fixed before running; no test tuning.','','| Arm | Final accuracy | Across-seed SD | Forgetting (pp) | Mean width |','|---|---|---|---|---|']
    for s in summary:
        lines.append(f"| {s['arm']} | {100*s['final_accuracy']:.2f}% | {100*s['final_sd']:.2f} | {100*s['forgetting']:.2f} | {s['mean_width']:.0f} |")
    lines+=['','Exploratory growth_sensitive arm uses loss EMA threshold 0.3 and five high-loss batches instead of 1.0 and twenty. Added after the initial run showed no recruitment; shortening persistence alone also did not recruit. This was not a preregistered threshold; all other settings are identical.','','Frozen old blocks need not preserve predictions: new blocks can change summed scores on old inputs. Readout updates can also overwrite class discrimination even if input features are frozen. Growth events, boundaries, per-pair accuracy matrices and confusion matrices are in results.json. This is a pilot with unequal growth capacity, not a full benchmark or proof about dendrites.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
