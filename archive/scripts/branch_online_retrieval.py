"""Boundary-free associative memory acquisition alongside original training.

Observe accepts only the current image descriptions and active branch route.
No historical image replay, phase identity, labels, or end-of-phase writes.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.branch_associative_retrieval import describe, normalize, unary, infer_local, responses, predict, metrics


class OnlineMemory:
    def __init__(self, stable_batches, seed):
        self.stable_batches=stable_batches
        self.previous=None
        self.streak=0
        self.accepted=0
        self.seen=0
        self.local_sum=np.zeros((24,4,196),dtype='float32')
        self.local_count=np.zeros((24,4),dtype='float32')
        self.links=np.zeros((24,4,4),dtype='float32')
        self.centers=np.zeros((95,196),dtype='float32')
        self.routes=np.zeros((95,24),dtype='uint8')
        self.counts=np.zeros(95,dtype='float32')
        self.replacements=0
        self.rng=np.random.default_rng(seed)
        self.reservoir=np.zeros((95,196),dtype='float32')
        self.reservoir_routes=np.zeros((95,24),dtype='uint8')
        self.priorities=np.full(95,-1,dtype='float32')

    def observe(self,q,route):
        self.seen+=len(q)
        self.streak=self.streak+1 if self.previous is not None and np.array_equal(route,self.previous) else 1
        self.previous=route.copy()
        if self.streak<self.stable_batches:
            return
        self.accepted+=len(q)
        # Uniform random-priority reservoir, independent of labels and phases.
        priorities=np.concatenate([self.priorities,self.rng.random(len(q)).astype('float32')])
        keep=np.argsort(priorities)[-95:]
        self.reservoir=np.concatenate([self.reservoir,q])[keep]
        self.reservoir_routes=np.concatenate([self.reservoir_routes,np.broadcast_to(route,(len(q),24))])[keep]
        self.priorities=priorities[keep]
        self.local_sum[np.arange(24),route]+=q.sum(0)
        self.local_count[np.arange(24),route]+=len(q)
        self.links[np.arange(24),route,np.roll(route,-1)]+=len(q)
        for item in q:
            used=self.counts>0
            compatible=used & (self.routes==route).all(1)
            eligible=np.flatnonzero(compatible)
            best=-1
            if len(eligible):
                similarity=normalize(self.centers[eligible])@item
                best=int(eligible[similarity.argmax()])
            # Fixed novelty threshold; never tuned on test results.
            allocate=best<0 or float(normalize(self.centers[best:best+1])[0]@item)<.85
            if allocate and (~used).any():
                best=int(np.flatnonzero(~used)[0])
                self.centers[best]=item;self.routes[best]=route;self.counts[best]=1
            elif best<0:
                # Full bank and a new route: evict least-supported prototype.
                best=int(self.counts.argmin())
                self.centers[best]=item;self.routes[best]=route;self.counts[best]=1
                self.replacements+=1
            else:
                self.counts[best]+=1
                self.centers[best]+=(item-self.centers[best])/self.counts[best]

    def state(self):
        return {k:v.copy() for k,v in vars(self).items() if isinstance(v,np.ndarray)}

    def size(self):
        return {'local_communication_bytes':self.local_sum.nbytes+self.local_count.nbytes+self.links.nbytes,
                'shared_bytes':self.centers.nbytes+self.routes.nbytes+self.counts.nbytes,
                'reservoir_bytes':self.reservoir.nbytes+self.reservoir_routes.nbytes+self.priorities.nbytes,
                'common_write_state':'24-byte previous route plus streak counter; Python overhead excluded'}


@torch.no_grad()
def evaluate(model,mem,tx,ty,routes):
    before=mem.state()
    q=describe(tx.numpy())
    b=responses(model,tx)
    weight=model.head.weight.numpy();bias=model.head.bias.numpy()
    scores=unary(q,normalize(mem.local_sum),mem.local_count>0)
    links=mem.links/np.maximum(mem.links.sum((1,2),keepdims=True),1)
    results=[]
    for name,gamma,trace in [('local',0.,0.),('local_communication',.05,0.),('local_trace',0.,.05),('local_communication_trace',.05,.05)]:
        winners=infer_local(scores,links,gamma,trace)
        results.append({'arm':name,**metrics(predict(b,winners,weight,bias),ty,winners,routes)})
    used=mem.counts>0
    if used.any():
        winners=mem.routes[used][(q@normalize(mem.centers[used]).T).argmax(1)].astype('int64')
    else:
        winners=np.broadcast_to(model.gates().argmax(-1).numpy(),(len(tx),24))
    results.append({'arm':'shared_95',**metrics(predict(b,winners,weight,bias),ty,winners,routes)})
    used=mem.priorities>=0
    if used.any():
        winners=mem.reservoir_routes[used][(q@mem.reservoir[used].T).argmax(1)].astype('int64')
    else:
        winners=np.broadcast_to(model.gates().argmax(-1).numpy(),(len(tx),24))
    results.append({'arm':'shared_reservoir_95',**metrics(predict(b,winners,weight,bias),ty,winners,routes)})
    assert all(np.array_equal(v,mem.state()[k]) for k,v in before.items())
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    protocol={'seeds':[34,35,36], 'stable_batches':[1,10], 'training':'Original five-pair supervised stream; two epochs per pair, batch 128, Adam .001, clip 5, unchanged original objective. Two epochs are within-task passes; no revisiting old tasks.',
              'memory':'Writes happen on current training minibatch using its input and pre-update active route. No labels, task IDs, phase-end events, or future batches supplied to memory. Each image may be observed during each of the original two epochs; no extra historical pass.',
              'write_control':'Observe every batch, or only after 10 consecutive minibatches with exactly the same route. Global route stability signal is an engineering control, not claimed biologically local.',
              'local':'Running branch-conditioned input sums and coactivity on 24-neuron ring. Fixed gamma .05, three communication loops, trace .05/rho .9 inherited from prior pilot. No new tuning.',
              'shared':'95 online normalized-pixel prototypes tagged with observed routes. Update nearest compatible-route prototype; allocate if cosine below .85 and free slot exists. Full bank merges within route; new route evicts minimum-support entry. Fixed .85 threshold before running; no test tuning.',
              'reservoir_followup':'Added after observing first pilot bank saturation and performance. Fixed 95-entry random-priority reservoir of pooled inputs plus contemporaneous routes; nearest cosine entry at test time. No classifier replay, labels or phase inputs. Exploratory follow-up, not preregistered confirmation. No hyperparameter sweep.',
              'testing':'Randomly mixed official 10,000 images; same order for all arms. No test labels/IDs fed to routing; all 10 output classes. Weights and associative memory frozen. Trace is transient, starts empty, no task resets.',
              'scope':'Associative sidecar acquired concurrently with training; original usefulness gates still control gradient training. Input-based retrieval used at evaluation. Not a fully local biological training algorithm.',
              'diagnostics':'Phase-end routes saved only for oracle/route-match scoring; never supplied to memory. All final backbone tensors must exactly reproduce prior checkpoints.',
              'budget':'Local+communication 77,184 bytes; shared 77,140 bytes. Shared and local are separate experimental mechanisms stored together only for paired measurement; common previous-route/streak bookkeeping extra.',
              'limitations':'Three reused classifier seeds, one task order, one fixed shared threshold, two within-task epochs. No autonomous data curriculum, joint retrieval-driven training, pure local credit assignment, or task-overlap benchmark.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for file in [Path(__file__),Path('bdh_cl/scripts/branch_associative_retrieval.py'),Path('bdh_cl/scripts/split_mnist_usefulness.py')]:
        (out/file.name).write_bytes(file.read_bytes())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    ty=raw['test_labels'].astype('int64')
    order=np.random.default_rng(20261013).permutation(len(tx));tx=tx[order];ty=ty[order]
    np.save(out/'test_order.npy',order)
    records=[];audits=[]
    for seed in protocol['seeds']:
        torch.manual_seed(seed);model=make('local','hard');opt=torch.optim.Adam(model.parameters(),lr=.001)
        memories={n:OnlineMemory(n,seed) for n in protocol['stable_batches']}
        routes=[];stage_counts=[];memory_seconds={n:0. for n in memories};learn_seconds=0.
        for stage,pair in enumerate(SPLITS):
            ix=torch.nonzero((y==pair[0])|(y==pair[1]),as_tuple=True)[0]
            for epoch in range(2):
                gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                indices=ix[torch.randperm(len(ix),generator=gen)]
                for ids in indices.split(128):
                    bx,by=x[ids],y[ids]
                    route=model.gates().argmax(-1).numpy().astype('uint8')
                    q=describe(bx.numpy())
                    for n,mem in memories.items():
                        start=time.perf_counter();mem.observe(q,route);memory_seconds[n]+=time.perf_counter()-start
                    start=time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    logits,candidates=model(bx,counterfactual=True)
                    targets=by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                    errors=F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                    loss=F.cross_entropy(logits,by)+.5*errors.mean()
                    feedback=errors.detach().mean(0)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.observe(feedback)
                    learn_seconds+=time.perf_counter()-start
            # Only diagnostic logging at a boundary; never influences memory.
            routes.append(model.gates().argmax(-1).numpy())
            stage_counts.append({n:{'accepted':m.accepted,'occupied':int((m.counts>0).sum()),'replacements':m.replacements}
                                 for n,m in memories.items()})
            print(seed,'stage',stage+1,stage_counts[-1],flush=True)
        reference=torch.load(f'results/split_mnist_compute_comparison_20261013/{seed}_local.pt',weights_only=True)
        assert all(torch.equal(v,reference[k]) for k,v in model.state_dict().items())
        routes=np.stack(routes)
        before={k:v.clone() for k,v in model.state_dict().items()}
        for n,mem in memories.items():
            results=evaluate(model,mem,tx,ty,routes)
            for result in results:
                records.append({'seed':seed,'stable_batches':n,**result})
            np.savez(out/f'{seed}_stability{n}_memory.npz',**mem.state())
            audits.append({'seed':seed,'stable_batches':n,'accepted_presentations':mem.accepted,'seen_presentations':mem.seen,
                           'occupied_slots':int((mem.counts>0).sum()),'distinct_routes':len(np.unique(mem.routes[mem.counts>0],axis=0)),
                           'replacements':mem.replacements,'update_wall_seconds_both_memories':memory_seconds[n],
                           'backbone_training_seconds_excluding_input_prep':learn_seconds,**mem.size(),'stage_counts':stage_counts,
                           'backbone_exact_match':True})
            print(seed,'stability',n,[(r['arm'],round(r['accuracy']*100,2)) for r in results],flush=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        (out/'results.json').write_text(json.dumps(records,indent=2))
        (out/'audit.json').write_text(json.dumps(audits,indent=2))
    lines=['# Online acquisition of input-to-branch retrieval memory','',
           '**Memory is acquired during the original training stream; no historical revisit or task-boundary signal is used by either router.** '
           'Backbone uses supervised labels as before. Router sees only current input descriptions and active branches. '
           'All three backbone checkpoints exactly reproduce the previous runs.', '',
           '| Writes allowed | Retrieval | Final accuracy mean ± SD | Old-digit accuracy | Exact historical route match |',
           '|---|---|---:|---:|---:|']
    for n in [1,10]:
        for name in ['local','local_communication','local_trace','local_communication_trace','shared_95','shared_reservoir_95']:
            rs=[r for r in records if r['stable_batches']==n and r['arm']==name]
            a=[r['accuracy']*100 for r in rs]
            lines.append(f'| {"Every batch" if n==1 else "After 10 stable batches"} | {name} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | '
                         f'{100*st.mean(r["old_accuracy"] for r in rs):.2f}% | {100*st.mean(r["exact_route_match"] for r in rs):.2f}% |')
    lines+=['','Reference: ordinary final backbone 19.40%; externally boundary-saved oracle configurations 98.02%. '
            'Previous retrospective shared retrieval was 91.97%, but used historical training inputs associated '
            'with end-of-phase routes; it is a different, privileged acquisition protocol.', '',
            '## Interpretation and limits','',
            'The online learner associates an image with the route active before its training update. This can '
            'be a transitional or poorly trained route. Stability-gated writes suppress some transitions without '
            'using task boundaries; this uses a global route-stability signal and is not claimed biologically local. '
            'The shared bank can forget associations through capacity limits and replacement. Unlike the retrospective '
            'pilot, no memory is fitted using future end-of-phase configurations.', '',
            'Local cue descriptions are fixed 2×2 pooled, normalized pixels; each branch accumulates its coactive '
            'input descriptions. Ring-edge coactivity supports three communication loops. Shared prototypes are '
            'updated online, with fixed novelty threshold and bounded slots. Local+communication storage is '
            '77,184 bytes, shared storage 77,140 bytes, excluding common write bookkeeping and frozen backbone. '
            'A follow-up random-priority reservoir also uses 77,140 bytes, excluding small RNG state. It stores '
            '95 pooled input descriptions and their observed routes, performs no classifier replay, and retrieves '
            'the nearest entry at inference. This exploratory control was added after the first pilot revealed '
            'early bank saturation; no settings were swept. '
            'No tuning on official test data; all planned conditions are reported. Trace experiments process '
            'the same shuffled image sequence and have no task-boundary resets.', '',
            'These are concurrent sidecar retrieval memories: their choices do not yet control gradient training. '
            'Original expensive counterfactual learning and backpropagation remain. Two epochs within each current '
            'task are retained, but no additional pass through old tasks is made. Oracle route-match statistics '
            'use labels only after predictions and do not feed into the router. No claim of a fully biological '
            'learning rule, dendrite-specific advantage, or robustness to overlapping tasks.', '',
            '## Reproducibility','',
            'Run `python -m bdh_cl.scripts.branch_online_retrieval --out NEW_DIRECTORY`. '
            'Protocol, copied source, test ordering, learned memories, per-seed results and acquisition counts '
            'are saved alongside this report. Each inference call asserts that associative memory stays fixed '
            'and final backbone state matches the original saved run exactly.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:17]),flush=True)


if __name__=='__main__':
    main()
