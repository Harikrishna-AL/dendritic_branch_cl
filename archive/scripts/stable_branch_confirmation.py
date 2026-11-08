"""One-pass prototype-free class-incremental learning; matched numeric storage.

Random branch computations and response-max selection stay fixed. Cumulative
readout sufficient statistics retain data constraints. This is an engineering
baseline, not three-factor biological learning or proof of unique dendrites.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.continual_stability_pilot import stable,StableFeatures
from bdh_cl.scripts.eligibility_router_pilot import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'dev_seed':80,'final_seeds':[84,85,86],
              'selection':'Increasing fixed branch widths95/192/384/768; stop at first>=90% final training-validation accuracy, otherwise keep best. Fixed ridge1 inherited from previous training validation. Official test evaluated only after locking width.',
              'stream':'One pass through all60,000 training images in disjoint digit-pair sequence, minibatch512. No task boundary input, no task IDs, no replay/examples/prototype/configuration archive, no pretraining. Original train labels are supervised learning targets.',
              'learning':'Frozen random tanh branches; each neuron chooses largest current branch response. One shared10-class readout accumulates feature Gram and feature/onehot-target cross-products; ridge solve after every minibatch. This standard sufficient-statistic learner is NOT local three-factor learning.',
              'memory':'Includes all frozen first-layer weights/biases and FP64 Gram, cross and head arrays. Covariance is additional persistent memory, not a short-term trace. No claims of zero extra memory, biologically plausible synapses, or uniquely dendritic retention.',
              'matching':'Choose MLP frozen-random-feature width minimizing numeric-state byte difference. Different number of parameters and features, explicitly counted. Same data, one pass, updates and ridge1; similar storage, unequal operations.',
              'orders':'Forward and reverse. Exact same final data multiset leads same least-squares optimum; not independent evidence of an adaptive protection rule.',
              'validation':'Compare final streaming predictions with an offline sufficient-statistic reference on training data ONLY after completion. Verify per-batch cumulative counts, first-layer invariance, no target/ID inputs to inference. Test data are official10,000 held-out images.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for file in [Path(__file__),Path('bdh_cl/scripts/continual_stability_pilot.py')]:
        (out/file.name).write_bytes(file.read_bytes())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(80)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False
    groups=[torch.nonzero(mask&((y==a)|(y==b)),as_tuple=True)[0] for a,b in SPLITS]
    dev=[]
    for width in [95,192,384,768]:
        model,_,stats=stable('branch',80,x,y,groups,1.,width)
        score=evaluate(model,x[val],y[val])
        dev.append({'width':width,**score,**stats});print('validation width',width,round(score['accuracy']*100,2),flush=True)
        if score['accuracy']>=.90:break
    best=dev[-1] if dev[-1]['accuracy']>=.9 else max(dev,key=lambda r:r['accuracy'])
    width=best['width'];probe=StableFeatures('branch',0,width)
    target=sum(v.numel()*v.element_size() for v in probe.state_dict().values())
    # MLP bytes=8h^2+3316h+168 with this exact representation.
    mlp_width=min(range(1,3000),key=lambda h:abs(8*h*h+3316*h+168-target))
    (out/'selection.json').write_text(json.dumps({'candidates':dev,'selected':best,'matched_mlp_width':mlp_width},indent=2))
    rows=[];checks=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        groups=[torch.nonzero((y==a)|(y==b),as_tuple=True)[0] for a,b in pairs]
        for seed in [84,85,86]:
            for kind,w in [('branch',width),('mlp',mlp_width)]:
                model,states,stats=stable(kind,seed,x,y,groups,1.,w)
                first=StableFeatures(kind,seed,w).first.state_dict()
                assert all(torch.equal(v,model.first.state_dict()[k]) for k,v in first.items())
                seen=0
                for ix,state in zip(groups,states):
                    seen+=len(ix)
                    assert float(state['gram'][-1,-1])==seen
                stage_scores=[]
                for state in states:model.load_state_dict(state);stage_scores.append(evaluate(model,tx,ty))
                old=[i for i,pair in enumerate(SPLITS) if pair!=pairs[-1]]
                # Independent full-history reference used solely as post-run audit.
                refgram=torch.zeros_like(model.gram);refcross=torch.zeros_like(model.cross)
                with torch.no_grad():
                    for ids in torch.arange(len(y)).split(512):
                        z=model.features(x[ids]);refgram+=z.T@z
                        refcross+=z.T@torch.nn.functional.one_hot(y[ids],10).double()
                    reg=torch.eye(w+1,dtype=torch.float64);reg[-1,-1]=0
                    reference=torch.linalg.solve(refgram+reg,refcross)
                    diff=float((reference-model.head).abs().max())
                    refpred=torch.cat([(model.features(b)@reference).argmax(1) for b in tx.split(512)])
                    streampred=torch.cat([model(b).argmax(1) for b in tx.split(512)])
                    assert torch.equal(refpred,streampred)
                checks.append({'order':order,'seed':seed,'kind':kind,'offline_head_max_error':diff,'test_predictions_exact_match':True,'examples_seen':seen,'first_weights_unchanged':True})
                row={'order':order,'seed':seed,'kind':kind,'width':w,'accuracy':stage_scores[-1]['accuracy'],
                     'old_accuracy':st.mean(stage_scores[-1]['pairs'][i] for i in old),'stages':stage_scores,**stats,
                     'numeric_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                     'first_layer_parameters':sum(p.numel() for p in model.first.parameters())}
                rows.append(row);torch.save(model.state_dict(),out/f'{order}_{seed}_{kind}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2));(out/'validation.json').write_text(json.dumps(checks,indent=2))
                print(order,seed,kind,w,round(row['accuracy']*100,2),'bytes',row['numeric_bytes'],flush=True)
    lines=['# One-pass class-incremental learning with stable branch computations','',
           '**Working engineering model on Split MNIST; not the proposed short-term-memory biological mechanism.** '
           'Random computations stay stable; an accumulating ridge readout preserves old training constraints. '
           'All models begin before task1, learn each image once, and predict among all ten digits from the image alone.', '',
           '| Model | Width | Final accuracy mean ± SD | Old pairs | Numeric state bytes |', '|---|---:|---:|---:|---:|']
    for kind in ['branch','mlp']:
        rs=[r for r in rows if r['kind']==kind];a=[100*r['accuracy'] for r in rs]
        lines.append(f'| {kind} | {rs[0]["width"]} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {100*st.mean(r["old_accuracy"] for r in rs):.2f}% | {rs[0]["numeric_bytes"]:,} |')
    lines+=['','Each branch neuron selects the largest of four tanh responses, computed from the current image. '
            'There is no saved branch configuration to retrieve. Learned long-term readout statistics store '
            'past training constraints. Full covariance is extra persistent memory, not free routing or an '
            'eligibility trace. The matched MLP receives a similar total numeric storage budget and the same '
            'readout learning rule.', '',
            'The classifier is updated after every minibatch, not only at supplied task boundaries. Single-pass '
            'supervised class-incremental learning is operationally satisfied. However, the representation is '
            'random/frozen and the learning mechanism is standard streaming ridge regression. This establishes '
            'a functional engineering baseline, not novelty, dendritic superiority, or fully biological learning.', '',
            'Branch width was selected only using held-out training validation. Fresh evaluation seeds84–86, '
            'forward and reversed orders. No official test tuning. Final streaming test predictions exactly '
            'match an independently calculated full-training-data reference; that reference is a post-run '
            'audit and never updates the streaming learner. This confirms the absence of order-induced '
            'information loss for the fixed representation, within numeric precision.', '',
            'The original reward-routed model with synaptic protection still failed. The covariance-based '
            'control solves a different mechanism: preserving sufficient statistics. It must not be '
            'reported as evidence that short-term branch traces alone prevented forgetting.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:11]),flush=True)


if __name__=='__main__':main()
