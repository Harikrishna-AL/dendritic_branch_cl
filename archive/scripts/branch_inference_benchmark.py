"""Exact lazy-inference equivalence and CPU latency, not reduced-feedback training."""
import json
import platform
import time
from pathlib import Path
import numpy as np
from bdh_cl.scripts.predictive_branches_mnist import Branches,MASK
from bdh_cl.scripts.predictive_router_ablation import embed
from bdh_cl.scripts.predictive_quality_ablation import Deep
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS

ARMS=['frozen_router','ungated','random_active']

def lazy_embed(model,images,arm,router,random_encoder):
    x=np.column_stack([images[:,MASK],np.ones(len(images))])
    output=np.zeros((len(images),8,8))
    weights=random_encoder if arm=='random_active' else model.encoder
    active=np.flatnonzero(model.active)
    if arm=='frozen_router':
        scores=x@router[:,active];chosen=active[scores.argmin(1)]
        for b in np.unique(chosen):
            selected=chosen==b
            output[selected,b]=np.tanh(x[selected]@weights[b])
    else:
        # One dense multiplication for all active encoders; no decoders.
        packed=weights[active].transpose(1,0,2).reshape(99,-1)
        output[:,active]=np.tanh(x@packed).reshape(len(images),len(active),8)
    return output.reshape(len(images),64)

def classify(features,prototypes,classes):
    # Same readout in both timing paths, isolating encoder/decoder savings.
    distances=np.sum((features[:,None]-prototypes[classes])**2,axis=2)
    return classes[distances.argmin(1)]

def timed(call,repeats):
    for _ in range(3):call()
    samples=[]
    for _ in range(7):
        start=time.perf_counter_ns()
        for _ in range(repeats):call()
        samples.append((time.perf_counter_ns()-start)/repeats/1000)
    return {'median_us_per_batch':float(np.median(samples)),
            'min_us_per_batch':float(min(samples)),'max_us_per_batch':float(max(samples))}

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[];bench=[]
    out=Path('results/branch_inference_benchmark');out.mkdir(parents=True,exist_ok=True)
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        m=Branches(seed,'frozen_reuse',.03);random_encoder=m.encoder.copy()
        router=np.zeros((99,8));router[-1]=.15
        proto={a:np.zeros((10,64)) for a in ARMS};count=np.zeros(10,int)
        deep=Deep(seed);curves={a:[] for a in ARMS};curves['deep']=[];maxdiff=0
        for stage,ix in enumerate(data):
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                for batch in np.array_split(order,16):
                    images=x[batch];labels=raw['train_labels'][batch]
                    frozen=m.frozen.copy();beforee=m.encoder[frozen].copy();befored=m.decoder[frozen].copy()
                    m.learn_pixels(images)
                    assert np.array_equal(beforee,m.encoder[frozen]) and np.array_equal(befored,m.decoder[frozen])
                    visible,_,_,_,_,errors=m.responses(images);eligible=m.active&~frozen
                    router[:,eligible]-=.005*visible.T@(visible@router[:,eligible]-errors[:,eligible])/len(images)
                    features={a:lazy_embed(m,images,a,router,random_encoder) for a in ARMS}
                    for c in np.unique(labels):
                        chosen=labels==c;n=int(chosen.sum());total=count[c]+n
                        for a in ARMS:proto[a][c]=(count[c]*proto[a][c]+features[a][chosen].sum(0))/total
                        count[c]=total
                    deep.observe(images,labels)
            classes=np.flatnonzero(count);seen=raw['test_labels']<2*(stage+1)
            for a in ARMS:
                reference=embed(m,tx,a,router,random_encoder)[0]
                optimized=lazy_embed(m,tx,a,router,random_encoder)
                difference=float(np.max(np.abs(reference-optimized)));maxdiff=max(maxdiff,difference)
                assert np.allclose(reference,optimized,atol=1e-12,rtol=1e-12)
                original=classify(reference,proto[a],classes);new=classify(optimized,proto[a],classes)
                assert np.array_equal(original,new),'Classification changed'
                curves[a].append(float(np.mean(new[seen]==raw['test_labels'][seen])))
            curves['deep'].append(float(np.mean(deep.predict(tx)[seen]==raw['test_labels'][seen])))
        runs.append({'seed':seed,'curves':curves,'active':int(m.active.sum()),'max_feature_difference':maxdiff})
        np.savez(out/f'checkpoint_seed{seed}.npz',encoder=m.encoder,decoder=m.decoder,active=m.active,
                 router=router,random_encoder=random_encoder,**{f'prototype_{a}':proto[a] for a in ARMS})
        for size,repeats in [(1,100),(64,20),(256,10)]:
            images=tx[:size]
            for a in ARMS:
                original=lambda:classify(embed(m,images,a,router,random_encoder)[0],proto[a],classes)
                optimized=lambda:classify(lazy_embed(m,images,a,router,random_encoder),proto[a],classes)
                old=timed(original,repeats);new=timed(optimized,repeats)
                bench.append({'seed':seed,'batch':size,'arm':a,'original':old,'lazy':new,
                              'speedup':old['median_us_per_batch']/new['median_us_per_batch']})
            bench.append({'seed':seed,'batch':size,'arm':'deep','timing':timed(lambda:deep.predict(images),repeats)})
        print(seed,'active',int(m.active.sum()),'accuracy',{a:curves[a][-1] for a in curves},'maxdiff',maxdiff,flush=True)
    (out/'results.json').write_text(json.dumps({'environment':{'platform':platform.platform(),'numpy':np.__version__,'dtype':'float64','timing':'7 rounds; median; CPU; first test batch, no data loading'},'runs':runs,'latency':bench},indent=2))
    lines=['# Lazy branch inference benchmark','','Same class-incremental stream as previous pilots: three seeds, 500 training images per digit, ten epochs per pair, all official test images. No task ID or replay. Full reconstruction errors still run during training/recruitment. Only feature extraction for classification is optimized.','','Frozen router computes scores for active branches from visible pixels, then encodes only the chosen branch. Ungated/random-active compute only active encoders. No decoder is called by lazy inference. Prototype readout is identical in original/lazy timing paths. Zero slots and frozen router choices are retained. All five stage predictions match on every official test image for all three seeds; features match to 1e-12.','','| Version | Final mean accuracy |','|---|---|']
    for a in ARMS+['deep']:lines.append(f"| {a} | {100*np.mean([r['curves'][a][-1] for r in runs]):.2f}% |")
    lines+=['','| Batch | Version | Original us/batch | Lazy us/batch | Median speedup across seeds |','|---|---|---|---|---|']
    for size in [1,64,256]:
        for a in ARMS:
            group=[b for b in bench if b['batch']==size and b['arm']==a]
            lines.append(f"| {size} | {a} | {np.median([b['original']['median_us_per_batch'] for b in group]):.2f} | {np.median([b['lazy']['median_us_per_batch'] for b in group]):.2f} | {np.median([b['speedup'] for b in group]):.2f}x |")
        group=[b for b in bench if b['batch']==size and b['arm']=='deep']
        lines.append(f"| {size} | deep | {np.median([b['timing']['median_us_per_batch'] for b in group]):.2f} | — | — |")
    lines+=['','Timing is local Python/NumPy CPU latency, not a hardware-independent efficiency claim; no GPU, energy, training-time or feedback-frequency savings measured. Packed encoder weights are assembled on each call (could be cached after freezing). Batch timings include routing, activations, feature assembly and prototype comparisons, exclude image preprocessing/loading. Architecture and classification quality differ from deep baseline. No claim of faster end-to-end learning. Learned branches use 2–3 active slots. Full per-seed ranges in results.json; predictor/router/prototype snapshots saved as checkpoint_seed*.npz.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-14:]),flush=True)

if __name__=='__main__':main()
