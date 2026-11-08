"""Matched-predictor routing ablation; routers never receive digit labels.

One label-free predictor stream is shared across arms to isolate feature routing.
Image-based linear routers predict branch reconstruction errors. Frozen router
rows stop learning when their corresponding predictor matures.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.predictive_branches_mnist import Branches, MASK
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS

ARMS=['reconstruction','soft_reconstruction','learned_router','frozen_router','ungated','random_active']

def embed(model,images,arm,router,random_encoder):
    x,_,z,_,_,errors=model.responses(images)
    if arm=='random_active':
        z=np.tanh(np.einsum('ni,bik->nbk',x,random_encoder))
        return (z*model.active[None,:,None]).reshape(len(images),64),np.zeros(len(images),int)
    if arm=='ungated':return (z*model.active[None,:,None]).reshape(len(images),64),errors.argmin(1)
    if arm in ['learned_router','frozen_router']:
        errors=x@router;errors[:,~model.active]=np.inf
    chosen=errors.argmin(1)
    if arm=='soft_reconstruction':
        gate=np.exp(-(errors-errors.min(1,keepdims=True))/.03);gate/=gate.sum(1,keepdims=True)
        return (z*gate[:,:,None]).reshape(len(images),64),chosen
    h=np.zeros_like(z);h[np.arange(len(images)),chosen]=z[np.arange(len(images)),chosen]
    return h.reshape(len(images),64),chosen

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        model=Branches(seed,'frozen_reuse',.03);random_encoder=model.encoder.copy()
        routers={arm:np.zeros((99,8)) for arm in ['learned_router','frozen_router']}
        for router in routers.values():router[-1,:]=.15
        protos={a:np.zeros((10,64)) for a in ARMS};count=np.zeros(10,int)
        logs={a:{'seed':seed,'arm':a,'seen_accuracy':[],'pairs':[],'route_change':[],'teacher_agreement':[]} for a in ARMS}
        saved={a:{} for a in ARMS}
        for stage,ix in enumerate(data):
            old={a:protos[a].copy() for a in ARMS};oldclasses=np.flatnonzero(count)
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                for batch in np.array_split(order,16):
                    images=x[batch];labels=raw['train_labels'][batch]
                    frozen_before=model.frozen.copy()
                    frozen_encoder=model.encoder[frozen_before].copy()
                    frozen_decoder=model.decoder[frozen_before].copy()
                    model.learn_pixels(images)
                    assert np.array_equal(frozen_encoder,model.encoder[frozen_before])
                    assert np.array_equal(frozen_decoder,model.decoder[frozen_before])
                    visible,_,_,_,_,errors=model.responses(images)
                    for arm,router in routers.items():
                        eligible=model.active.copy()
                        if arm=='frozen_router':eligible&=~frozen_before
                        before=router[:,~eligible].copy()
                        residual=visible@router[:,eligible]-errors[:,eligible]
                        router[:,eligible]-=.005*visible.T@residual/len(images)
                        assert np.array_equal(before,router[:,~eligible])
                    representations={a:embed(model,images,a,routers.get(a),random_encoder)[0] for a in ARMS}
                    for c in np.unique(labels):
                        selected=labels==c;n=int(selected.sum());total=count[c]+n
                        for a in ARMS:protos[a][c]=(count[c]*protos[a][c]+representations[a][selected].sum(0))/total
                        count[c]=total
            teacher=model.responses(tx)[-1].argmin(1)
            seen=raw['test_labels']<2*(stage+1);classes=np.flatnonzero(count)
            for arm in ARMS:
                assert np.array_equal(old[arm][oldclasses],protos[arm][oldclasses])
                h,route=embed(model,tx,arm,routers.get(arm),random_encoder)
                distance=np.sum((h[:,None]-protos[arm][classes])**2,axis=2)
                pred=classes[distance.argmin(1)]
                log=logs[arm];log['seen_accuracy'].append(float(np.mean(pred[seen]==raw['test_labels'][seen])))
                log['pairs'].append([float(np.mean(pred[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
                changes=[]
                for c in classes:
                    current=route[raw['test_labels']==c]
                    if int(c) in saved[arm]:changes.append(float(np.mean(current!=saved[arm][int(c)])))
                    else:saved[arm][int(c)]=current.copy()
                log['route_change'].append(float(np.mean(changes)) if changes else 0)
                log['teacher_agreement'].append(float(np.mean(route[seen]==teacher[seen])))
        for arm in ARMS:
            logs[arm]['active']=int(model.active.sum());logs[arm]['events']=model.events
            runs.append(logs[arm]);print(seed,arm,logs[arm]['seen_accuracy'],flush=True)
    summary=[]
    for arm in ARMS:
        group=[r for r in runs if r['arm']==arm];values=[r['seen_accuracy'][-1] for r in group]
        summary.append({'arm':arm,'accuracy':float(np.mean(values)),'sd':float(np.std(values,ddof=1)),
                        'route_change':float(np.mean([r['route_change'][-1] for r in group])),
                        'teacher_agreement':float(np.mean([r['teacher_agreement'][-1] for r in group]))})
    out=Path('results/predictive_router_ablation');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summary},indent=2))
    lines=['# Matched predictive router ablation','','Three seeds, same five class-incremental pairs, 500 training images per digit, ten epochs, all 10000 test images. All seen class prototypes compete; no task IDs or replay. Same 8 branch slots, same predictor weights, same label-free recruitment events shared across arms. Every branch has 8 latent features; active capacity is matched per seed at every step. Frozen-predictive model and threshold 0.03 unchanged from previous run.','','Reconstruction uses lowest measured held-out pixel error. Soft reconstruction weights all active branches with error softmax, temperature 0.03. Learned routers predict each branch error using the 98 visible pixels plus bias; online squared-error update, rate 0.005. Learned router remains plastic; frozen router freezes each output column when that branch matures. Router receives no digit labels and inference needs only visible pixels. Neither router uses test data for updates. Ungated exposes all active learned latents; random-active exposes the same number of frozen random latents. Label stream is used exclusively for prototype means.','','| Arm | Final accuracy | Seed SD | Old route changes | Agreement with reconstruction winner |','|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['arm']} | {100*s['accuracy']:.2f}% | {100*s['sd']:.2f} | {100*s['route_change']:.2f}% | {100*s['teacher_agreement']:.2f}% |")
    lines+=['','Route statistics are not meaningful for ungated/random-active as classification gates; shown only as diagnostics. Frozen router columns do not ensure stable winners: a new column can win on old images. This isolates routing used for classification; learned routers do NOT change predictor training assignments, which remain reconstruction-based. A linear router may poorly approximate branch errors. Branches freeze after 20 updates, so representation quality is limited. All hyperparameters fixed before this ablation; no test-based selection. This remains an engineered pilot, not biological evidence.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
