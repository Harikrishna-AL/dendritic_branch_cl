"""Mixed-class representation test; no routing/recruitment/continual-learning claim.

Fixed 196->64 tanh representation (8 groups of 8 branch features). Pixel decoder
is additive across groups; group-local errors backpropagate through each group's
decoder weights. This is mathematically a one-hidden-layer autoencoder, not a
novel learning algorithm. Denoising changes only the input corruption.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample

def softmax_gradient(logits,y):
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    p[np.arange(len(y)),y]-=1
    return p/len(y)

def prototypes(h,y,test):
    means=np.stack([h[y==c].mean(0) for c in range(10)])
    return ((test[:,None]-means)**2).sum(2).argmin(1)

def main():
    raw=load_raw('data_cache/fashion_mnist')
    rawx=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    rawtest=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        ix=np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==c),1000,replace=False) for c in range(10)])
        x=rawx[ix];y=raw['train_labels'][ix];mean=x.mean(0);x=x-mean;test=rawtest-mean
        xb=np.column_stack([x,np.ones(len(x))]);tb=np.column_stack([test,np.ones(len(test))])
        initial=np.random.default_rng(seed+20).normal(0,.1,(197,64))
        for arm in ['random','reconstruction','denoising','supervised_reference','local_supervised']:
            w=initial.copy();decoder=np.zeros((65,196));head=np.zeros((65,10));curve=[]
            local_heads=np.zeros((8,9,10))
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+epoch).permutation(len(x))
                for batch in np.array_split(order,157):
                    inp=xb[batch].copy()
                    if arm=='denoising':
                        mask=np.random.default_rng(seed*100000+epoch*1000+int(batch[0])).random((len(batch),196))<.2
                        inp[:,:196][mask]=0
                    h=np.tanh(inp@w);hb=np.column_stack([h,np.ones(len(h))])
                    if arm in ['reconstruction','denoising']:
                        error=2*(hb@decoder-x[batch])/(len(batch)*196)
                        dw=inp.T@((error@decoder[:-1].T)*(1-h*h));dd=hb.T@error
                        w-=.2*dw;decoder-=.2*dd
                    elif arm=='supervised_reference':
                        error=softmax_gradient(hb@head,y[batch]);dw=inp.T@((error@head[:-1].T)*(1-h*h))
                        dh=hb.T@error;w-=.1*dw;head-=.1*dh
                    elif arm=='local_supervised':
                        groups=h.reshape(len(h),8,8)
                        group_bias=np.concatenate([groups,np.ones((len(h),8,1))],axis=2)
                        logits=np.einsum('nbk,bkc->nbc',group_bias,local_heads)
                        p=np.exp(logits-logits.max(2,keepdims=True));p/=p.sum(2,keepdims=True)
                        p[np.arange(len(batch))[:,None],np.arange(8)[None],y[batch][:,None]]-=1
                        p/=len(batch)
                        # Independent branch errors; no shared label readout.
                        local=np.einsum('nbc,bkc->nbk',p,local_heads[:,:8])*(1-groups*groups)
                        dw=inp.T@local.reshape(len(batch),64)
                        dh=np.einsum('nbk,nbc->bkc',group_bias,p)
                        w-=.1*dw;local_heads-=.1*dh
                if arm in ['reconstruction','denoising']:
                    h=np.tanh(xb@w);hb=np.column_stack([h,np.ones(len(h))])
                    curve.append(float(np.mean((hb@decoder-x)**2)))
            h=np.tanh(xb@w);th=np.tanh(tb@w)
            proto_accuracy=float(np.mean(prototypes(h,y,th)==raw['test_labels']))
            # Identical supervised linear probe on every frozen representation.
            ph=np.column_stack([h,np.ones(len(h))]);pth=np.column_stack([th,np.ones(len(th))])
            probe=np.zeros((65,10))
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+epoch+20000).permutation(len(x))
                for batch in np.array_split(order,157):
                    probe-=.1*ph[batch].T@softmax_gradient(ph[batch]@probe,y[batch])
            linear_accuracy=float(np.mean((pth@probe).argmax(1)==raw['test_labels']))
            if arm=='random':assert np.array_equal(w,initial)
            assert np.isfinite(h).all() and np.isfinite(th).all()
            runs.append({'seed':seed,'arm':arm,'prototype_accuracy':proto_accuracy,'linear_probe_accuracy':linear_accuracy,
                         'reconstruction_training_curve':curve,'mean_feature_variance':float(h.var(0).mean()),
                         'encoder_movement':float(np.linalg.norm(w-initial))})
            print(seed,arm,'prototype',proto_accuracy,'linear',linear_accuracy,'variance',h.var(0).mean(),flush=True)
    summary=[]
    for arm in ['random','reconstruction','denoising','supervised_reference','local_supervised']:
        group=[r for r in runs if r['arm']==arm]
        summary.append({'arm':arm,'prototype_accuracy':float(np.mean([r['prototype_accuracy'] for r in group])),
                        'prototype_sd':float(np.std([r['prototype_accuracy'] for r in group],ddof=1)),
                        'linear_probe_accuracy':float(np.mean([r['linear_probe_accuracy'] for r in group])),
                        'linear_sd':float(np.std([r['linear_probe_accuracy'] for r in group],ddof=1))})
    out=Path('results/feature_learning_clean');out.mkdir(parents=True,exist_ok=True)
    hashes={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in Path('data_cache/fashion_mnist').glob('*.gz')}
    (out/'results.json').write_text(json.dumps({'protocol':{'dataset':'Fashion-MNIST','training':'mixed classes; 1000/class; 10 epochs','test':'all official 10000','features':64,'image':'14x14','preprocessing':'subtract training-subset pixel means','linear_probe':'10 epochs, frozen features','continual_learning':False,'source':'https://github.com/zalandoresearch/fashion-mnist','dataset_md5':hashes},'runs':runs,'summary':summary},indent=2))
    lines=['# Minimal representation learning experiment','','Fashion-MNIST mixed-class training, 10000 balanced training examples, all 10000 official test images, three paired seeds, ten representation epochs. Fixed architecture and dimensionality: 14x14 image, 196 pixels plus bias -> 64 tanh features. Training-subset pixel means subtracted for every arm. No task routing, recruitment, protection, prototype timing or sequential training. Eight groups of eight features are computationally equivalent to one hidden layer; this test does not establish dendritic novelty.','','Random encoder frozen. Reconstruction learns clean pixels. Denoising learns the same pixel targets from inputs with 20% coordinates set to their mean (zero after centering). Both use the same encoder and additive linear decoder, learning rate 0.2. Supervised reference uses digit/category labels and ten-class cross entropy with learning rate 0.1. It is a reference, not label-free. All three learned representations train for ten epochs; random gets the same labeled readout data. No test tuning.','','After representation training features freeze. Two identical readouts per arm: class mean prototypes, and a supervised linear softmax probe trained ten epochs at rate 0.1. Probe labels do not update representation. All ten classes compete. Readouts assess usable feature quality, not retention.','','| Features | Prototype accuracy | Linear probe accuracy |','|---|---|---|']
    for s in summary:lines.append(f"| {s['arm']} | {100*s['prototype_accuracy']:.2f}% | {100*s['linear_probe_accuracy']:.2f}% |")
    lines+=['','Local-supervised follow-up: each of eight 8-feature branches predicts the class through its own 8+bias -> 10 head. Each local class error updates only that branch encoder/head. These training heads are discarded before readout evaluation. Same 64 total features, seed, examples and epochs. Uses labels and is not unsupervised; has 720 temporary head parameters rather than 650 in the shared supervised reference. Added after initial results showed weak pixel-learning benefits; exploratory comparison, not preregistered.','','Interpret prototype and linear probe results together: means may be inadequate even when class information is linearly accessible. Reconstruction loss reduction does not prove semantic improvement. Dataset hashes, seed scores, feature variance and encoder movement in results.json. This is subsampled and exploratory; reference and unsupervised training have different objectives.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
