"""Input-only retrieval pilot; retrospective, boundary-associated training cues.

No network retraining. No task ID or target enters router inference. This does
not demonstrate autonomous archive creation or online cue learning.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import make


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def describe(x):
    # Fixed local pooling, no learned encoder or pretraining.
    return normalize(x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196)).astype('float32')


def softmax(x):
    e = np.exp(x - x.max(-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)


def fit_local(q, target):
    keys = np.zeros((24,4,196),dtype='float32')
    valid = np.zeros((24,4),dtype=bool)
    links = np.zeros((24,4,4),dtype='float32')
    for i in range(24):
        for j in range(4):
            mask = target[:,i]==j
            if mask.any():
                keys[i,j]=q[mask].mean(0)
                valid[i,j]=True
        # Undirected ring edges: i -> (i+1)%24. Store only one copy.
        np.add.at(links[i], (target[:,i],target[:,(i+1)%24]),1)
    links /= np.maximum(links.sum((1,2),keepdims=True),1)
    return normalize(keys), valid, links


def unary(q, keys, valid):
    scores = (q @ keys.reshape(96,196).T).reshape(-1,24,4)
    return np.where(valid[None], scores, -1e4)


def communicate(p, links):
    # Conditional coactivity in either direction along a fixed local ring.
    right = links/np.maximum(links.sum(2,keepdims=True),1e-8)
    left = links/np.maximum(links.sum(1,keepdims=True),1e-8)
    return .5*(np.einsum('...nl,njl->...nj',np.roll(p,-1,axis=-2),right)
                + np.einsum('...nj,njl->...nl',np.roll(p,1,axis=-2),np.roll(left,1,axis=0)))


def infer_local(scores, links, gamma=0., trace=0.):
    if trace == 0:
        p = softmax(scores/.05)
        if gamma:
            for _ in range(3):
                p = softmax((scores+gamma*communicate(p,links))/.05)
        return p.argmax(-1)
    # Genuine chronological memory, not independent batch-lane histories.
    history = np.zeros((24,4),dtype='float32')
    winners=[]
    for s in scores:
        drive=s+trace*history
        p=softmax(drive/.05)
        if gamma:
            for _ in range(3):
                p=softmax((drive+gamma*communicate(p,links))/.05)
        winners.append(p.argmax(-1))
        history=.9*history+.1*p
    return np.array(winners)


def fit_shared(q, phase, slots, seed):
    # Per-configuration spherical k-means: input-only within each association.
    rng=np.random.default_rng(seed)
    centers=[]
    labels=[]
    for k in range(5):
        z=q[phase==k]
        c=z[rng.choice(len(z),slots,replace=False)].copy()
        for _ in range(8):
            assignments=(z@c.T).argmax(1)
            c=normalize(np.stack([z[assignments==j].mean(0) if (assignments==j).any() else c[j]
                                  for j in range(slots)]))
        centers.append(c)
        labels.extend([k]*slots)
    return np.concatenate(centers),np.array(labels,dtype='int64')


@torch.no_grad()
def responses(model,x):
    return torch.cat([torch.tanh(model.first(b).reshape(-1,24,4)) for b in x.split(512)]).numpy()


def predict(branches,winners,weight,bias):
    hidden=np.take_along_axis(branches,winners[:,:,None],axis=2)[:,:,0]
    return (hidden@weight.T+bias).argmax(1)


def metrics(pred,y,winners,routes):
    expected=routes[y//2]
    return {'accuracy':float((pred==y).mean()), 'old_accuracy':float((pred[y<8]==y[y<8]).mean()),
            'branch_agreement':float((winners==expected).mean()),
            'exact_route_match':float((winners==expected).all(1).mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_text(Path(__file__).read_text())
    protocol={'seeds':[34,35,36], 'training':'Retrospective association pass over 50,000 historical training images; pairs associate images with their externally saved end-of-phase configurations. Not autonomous storage or online learning.',
              'validation':'10,000 historical training images (1,000 per digit), held out from router fitting. Frozen backbone previously saw all 60,000; this is router validation, not a new held-out backbone validation set.',
              'test':'Official 10,000 images, randomly permuted, identical order across arms. Test targets used only for metrics and explicitly labeled oracle control. Temporal trace resets at start, never at task boundaries.',
              'features':'Fixed 2x2 average pooling to 196 pixels, L2 normalization; no pretraining.',
              'local':'96 normalized input centroids conditioned on branch activity; 24-neuron ring coactivity edges; masked unused branches; temperature .05; 3 synchronous loops; gamma .05/.2/.5 picked on router validation.',
              'trace':'EMA .9 of previous branch probabilities; coefficient .05/.2 selected on router validation, genuine sequential inference on randomly shuffled images.',
              'shared':'5 or 95 input prototypes (1 or 19 per historical configuration); 8 spherical k-means iterations. Same association targets and input features. 95-prototype router matches local communication storage within 1%; 5-prototype router is smaller.',
              'limits':'Research-inspired circuit abstraction, not a biophysical simulation. Original supervised counterfactual training retained. No claim of dendrite-specific advantage or self-organizing memory storage. All retrieval methods can fail under feature/task ambiguity.',
              'selection':'Fixed candidate grids before test evaluation; each nonzero communication/trace arm selected on validation classification accuracy, ties favor smaller coefficient. Zero-effect baselines reported separately.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=raw['train_labels'].astype('int64'); ty=raw['test_labels'].astype('int64')
    rng=np.random.default_rng(20261013)
    val=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:1000] for c in range(10)])
    val=rng.permutation(val)
    train=np.setdiff1d(np.arange(len(y)),val)
    test_order=rng.permutation(len(ty))
    np.savez(out/'split_indices.npz',train=train,validation=val,test_order=test_order)
    tq=describe(x[train].numpy()); vq=describe(x[val].numpy()); eq=describe(tx[test_order].numpy())
    runs=json.loads(Path('results/split_mnist_compute_comparison_20261013/results.json').read_text())
    records=[]
    for seed in protocol['seeds']:
        run=next(r for r in runs if r['seed']==seed and r['arm']=='local')
        routes=np.array(run['routes'],dtype='int64')
        model=make('local','hard')
        model.load_state_dict(torch.load(f'results/split_mnist_compute_comparison_20261013/{seed}_local.pt',weights_only=True))
        before={k:v.clone() for k,v in model.state_dict().items()}
        weight=model.head.weight.detach().numpy(); bias=model.head.bias.detach().numpy()
        vb=responses(model,x[val]); eb=responses(model,tx[test_order])
        keys,valid,links=fit_local(tq,routes[y[train]//2])
        vu=unary(vq,keys,valid); eu=unary(eq,keys,valid)
        choices={}; sweeps={}
        for name,candidates in [('local',[(0.,0.)]),('local_communication',[(g,0.) for g in [.05,.2,.5]]),
                                ('local_trace',[(0.,t) for t in [.05,.2]]),
                                ('local_communication_trace',[(g,t) for g in [.05,.2,.5] for t in [.05,.2]])]:
            trials=[]
            for gamma,trace in candidates:
                winners=infer_local(vu,links,gamma,trace)
                acc=float((predict(vb,winners,weight,bias)==y[val]).mean())
                trials.append({'gamma':gamma,'trace':trace,'validation_accuracy':acc})
            best=max(trials,key=lambda z:z['validation_accuracy'])
            choices[name]=best; sweeps[name]=trials
        # Complete selection before evaluating any test intervention.
        (out/f'{seed}_validation.json').write_text(json.dumps(sweeps,indent=2))
        state={'keys':keys,'valid':valid,'links':links,'routes':routes}
        for name,choice in choices.items():
            start=time.perf_counter()
            winners=infer_local(eu,links,choice['gamma'],choice['trace'])
            elapsed=time.perf_counter()-start
            result=metrics(predict(eb,winners,weight,bias),ty[test_order],winners,routes)
            row={'seed':seed,'arm':name,**choice,**result,'selection_seconds_excluding_input_projection':elapsed,
                 'router_bytes':keys.nbytes+valid.nbytes+(links.nbytes if choice['gamma'] else 0)+(384 if choice['trace'] else 0)}
            records.append(row)
            print(seed,name,round(result['accuracy']*100,2),choice,flush=True)
        for slots in [1,19]:
            centers,labels=fit_shared(tq,y[train]//2,slots,seed)
            state[f'shared_{slots}_keys']=centers; state[f'shared_{slots}_labels']=labels
            winners=routes[labels[(eq@centers.T).argmax(1)]]
            records.append({'seed':seed,'arm':f'shared_{slots*5}',
                            **metrics(predict(eb,winners,weight,bias),ty[test_order],winners,routes),
                            'router_bytes':centers.nbytes+labels.nbytes+routes.nbytes})
        for name,winners in [('current',np.broadcast_to(routes[-1],(len(ty),24))),
                             ('random_route',routes[np.random.default_rng(seed).integers(5,size=len(ty))]),
                             ('oracle',routes[ty[test_order]//2])]:
            records.append({'seed':seed,'arm':name,**metrics(predict(eb,winners,weight,bias),ty[test_order],winners,routes)})
        # Check vectorized local communication against single-example evaluation.
        assert np.array_equal(infer_local(eu[:4],links,.2),np.concatenate([infer_local(eu[i:i+1],links,.2) for i in range(4)]))
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert abs(records[-1]['accuracy']-run['oracle_accuracy'])<1e-7
        assert abs(records[-3]['accuracy']-run['final_accuracy'])<1e-7
        np.savez(out/f'{seed}_router_state.npz',**state)
        (out/'results.json').write_text(json.dumps(records,indent=2))
    names=['current','random_route','local','local_trace','local_communication','local_communication_trace','shared_5','shared_95','oracle']
    lines=['# Input-driven associative branch retrieval', '',
           '**Input-only inference; retrospective, training-boundary-associated memory construction.** '
           'Three seeds, frozen final networks, same official test images in random order. '
           'No test-label feedback or task ID for learned routers. Oracle is a separate reference.', '',
           '| Method | Accuracy mean ± SD | Old digits | Exact historical route | Router bytes |',
           '|---|---:|---:|---:|---:|']
    for name in names:
        rs=[r for r in records if r['arm']==name]
        acc=[100*r['accuracy'] for r in rs]
        lines.append(f'| {name} | {statistics.mean(acc):.2f} ± {statistics.stdev(acc):.2f}% | '
                     f'{100*statistics.mean(r["old_accuracy"] for r in rs):.2f}% | '
                     f'{100*statistics.mean(r["exact_route_match"] for r in rs):.2f}% | {rs[0].get("router_bytes","—")} |')
    lines+=['', '## Method and interpretation', '',
            'For each neuron/branch, store the normalized mean of training input descriptions associated '
            'with that historical branch selection. New inputs excite branches by cosine similarity; '
            'softmax competition forms branch preferences. Nearest-neighbor ring edges store coactivity '
            'frequencies, converted to conditional support in both directions during three synchronous '
            'communication loops. Optional EMA activity traces bias the next image. These are heuristic '
            'circuit abstractions, not measured biological parameters.', '',
            'The shared control stores whole configurations and retrieves one using nearest input prototypes. '
            'Both local and shared methods receive the same training-image/configuration associations. '
            'The 95-prototype control has 76,200 router bytes versus 76,896 for local communication '
            '(excluding shared frozen backbone); the local temporal trace adds 384 bytes. '
            'Prototype/association counts, dtype and route indices are included. Inference temporaries '
            'and training workspace are excluded.', '',
            'Training images are retrospectively associated with configurations saved at the end of their '
            'training phase. This uses historical phase membership and access to historical inputs; it is '
            'not online self-organizing storage. All 50,000 router-fit images are used; 10,000 other training '
            'images choose the coefficients before test evaluation. Those validation images were previously '
            'seen by the frozen classifier. No parameter is selected using the official test set.', '',
            'The primary stream mixes all test images randomly, so recent history is not a reliable task cue. '
            'State resets once at the beginning, not using task boundaries. A benefit here establishes a '
            'retrieval proof of concept conditional on stored configurations, not end-to-end continual learning. '
            'Shared-router superiority would rule out a dendrite-specific advantage for these tested rules.', '',
            'Validation: final weights/buffers unchanged; current and oracle accuracies match saved reference '
            'runs; batched and single-image communication agree. All split indices, states, validation sweeps '
            'and source are saved. Selection timings exclude cue projection and classification and must not '
            'be interpreted as end-to-end latency.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:16]),flush=True)


if __name__=='__main__':
    main()
