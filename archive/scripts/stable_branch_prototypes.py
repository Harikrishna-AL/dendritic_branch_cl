"""Stable branch-coordinate prototype readout; no hard classification winner.

Same learning stream as learn-features/learn-router factorial. Each class stores
means only for mature branches, with a validity mask. Label-free router still
assigns predictive training, but classification compares all valid coordinates.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.branch_stability_factorial import Model
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS

def main(random_features=False):
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        m=Model(seed,True,True);proto=np.zeros((10,8,8));count=np.zeros((10,8),int);seen=np.zeros(10,bool)
        random_weights=m.w.copy()
        curves=[];missing=[];valid_history=[]
        for stage,ix in enumerate(data):
            old=np.flatnonzero(seen);before=proto.copy()
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                for batch in np.array_split(order,16):
                    m.update(x[batch]);response=m.response(x[batch]);z=response[1];mature=m.age>=20
                    if random_features:z=np.tanh(np.einsum('ni,bik->nbk',response[0],random_weights))
                    labels=raw['train_labels'][batch];seen[np.unique(labels)]=True
                    for c in np.unique(labels):
                        selected=labels==c;n=int(selected.sum())
                        for b in np.flatnonzero(mature):
                            total=count[c,b]+n
                            proto[c,b]=(count[c,b]*proto[c,b]+z[selected,b].sum(0))/total;count[c,b]=total
            assert np.array_equal(before[old],proto[old])
            classes=np.flatnonzero(seen);valid=count[classes]>0;z=m.response(tx)[1]
            if random_features:z=np.tanh(np.einsum('ni,bik->nbk',m.response(tx)[0],random_weights))
            dist=((z[:,None]-proto[classes])**2).sum(3)
            denominator=8*valid.sum(1)
            score=(dist*valid[None]).sum(2)/np.maximum(denominator,1)[None]
            score[:,denominator==0]=np.inf
            pred=classes[score.argmin(1)] if denominator.any() else np.full(len(tx),-1)
            testseen=seen[raw['test_labels']]
            curves.append(float(np.mean(pred[testseen]==raw['test_labels'][testseen])))
            missing.append(classes[denominator==0].tolist());valid_history.append(valid.sum(1).tolist())
        runs.append({'seed':seed,'accuracy':curves,'missing':missing,'valid_branch_counts':valid_history})
        print(seed,curves,'valid',valid_history[-1],flush=True)
    accuracy=float(np.mean([r['accuracy'][-1] for r in runs]));sd=float(np.std([r['accuracy'][-1] for r in runs],ddof=1))
    out=Path('results/stable_branch_prototypes_random' if random_features else 'results/stable_branch_prototypes');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'random_feature_control':random_features,'runs':runs,'mean_accuracy':accuracy,'sd':sd},indent=2))
    lines=['# Stable branch-coordinate prototypes','','Same trainable-feature/trainable-router model stream as branch_stability_factorial: eight slots available initially, local pixel prediction, each encoder/decoder freezes at 20 assigned updates, router freezes at global batch 100. Three seeds, 500 train images per digit, ten epochs per pair, all test images, no task IDs or replay.','','Class prototypes are maintained independently per branch only after that branch matures. Each class has a validity mask. Classification uses average squared distance over all valid branch feature coordinates for each seen class; no image winner is used. Missing classes remain in the accuracy denominator. Old prototypes never refreshed. Frozen branch coordinates are stable by construction, unlike winner-gated vectors. This abandons selective classification gating and may compute more encoders. Valid coordinate counts may differ across classes, making score calibration imperfect.','','Final accuracy: '+f'{100*accuracy:.2f}% (seed SD {100*sd:.2f} pp).','','This is a readout remedy/diagnostic, not a demonstration that local predictive routing improves classification. It requires comparison to frozen random branch features with identical maturity masks. Summary coordinate memory and class labels are used, but no old training examples retained.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if random_features:
        with (out/'REPORT.md').open('a') as f:f.write('\nCONTROL: Features in this run are fixed random initial encoder outputs; learned predictors/router run solely to match maturity masks and data eligibility.\n')

if __name__=='__main__':main()
