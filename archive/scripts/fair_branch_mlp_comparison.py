"""Fixed-protocol, new-seed comparison under a common numeric-state budget.

Five seeds x two orders. No test tuning. Replay and input-based output gating
are controls for the extra memory added to a branch network.
"""
import argparse
import json
import platform
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.branch_associative_retrieval import describe, normalize, unary


class Reservoir:
    def __init__(self,capacity,shape,dtype,value_shape,value_dtype,seed):
        self.data=np.zeros((capacity,*shape),dtype=dtype)
        self.values=np.zeros((capacity,*value_shape),dtype=value_dtype)
        self.priorities=np.full(capacity,-1,dtype='float32')
        self.rng=np.random.default_rng(seed)

    def write(self,data,values):
        priorities=np.concatenate([self.priorities,self.rng.random(len(data)).astype('float32')])
        keep=np.argsort(priorities)[-len(self.priorities):]
        self.data=np.concatenate([self.data,data])[keep]
        self.values=np.concatenate([self.values,values])[keep]
        self.priorities=priorities[keep]

    def nbytes(self):
        return self.data.nbytes+self.values.nbytes+self.priorities.nbytes

    def state(self):
        return {'data':self.data,'values':self.values,'priorities':self.priorities}


class Memory:
    def __init__(self,arm,seed):
        self.arm=arm;self.previous=None;self.streak=0
        self.sample_rng=np.random.default_rng(seed+10000)
        if arm=='branch_local':
            self.sums=np.zeros((24,4,196),dtype='float32')
            self.counts=np.zeros((24,4),dtype='float32')
        elif arm=='branch_reservoir':
            self.bank=Reservoir(95,(196,),'float32',(24,),'uint8',seed)
        elif arm=='mlp_replay':
            self.bank=Reservoir(98,(784,),'uint8',(),'uint8',seed)
        elif arm=='mlp_retrieval':
            self.bank=Reservoir(97,(196,),'float32',(10,),'uint8',seed)

    def stable(self,value):
        self.streak=self.streak+1 if self.previous is not None and np.array_equal(value,self.previous) else 1
        self.previous=value.copy()
        return self.streak>=10

    def observe(self,bx,by,route):
        if self.arm=='plain_mlp':return
        if self.arm=='mlp_replay':
            self.bank.write((bx.numpy()*255).round().astype('uint8'),by.numpy().astype('uint8'))
        elif self.arm=='mlp_retrieval':
            # Context inferred from observed supervised batch labels, not task ID.
            active=np.zeros(10,dtype='uint8');active[np.unique(by.numpy())]=1
            if self.stable(active):
                self.bank.write(describe(bx.numpy()),np.broadcast_to(active,(len(bx),10)))
        elif self.stable(route):
            q=describe(bx.numpy())
            if self.arm=='branch_local':
                self.sums[np.arange(24),route]+=q.sum(0)
                self.counts[np.arange(24),route]+=len(q)
            else:self.bank.write(q,np.broadcast_to(route,(len(bx),24)))

    def replay(self,n):
        used=np.flatnonzero(self.bank.priorities>=0)
        if not len(used):return None
        ix=self.sample_rng.choice(used,n,replace=True)
        return torch.tensor(self.bank.data[ix],dtype=torch.float32)/255,torch.tensor(self.bank.values[ix],dtype=torch.long)

    def state(self):
        arrays={}
        if hasattr(self,'bank'):arrays.update(self.bank.state())
        if hasattr(self,'sums'):arrays.update(sums=self.sums,counts=self.counts)
        if self.previous is not None:arrays['previous']=self.previous
        return arrays

    def bytes(self):
        return sum(v.nbytes for v in self.state().values())


@torch.no_grad()
def predictor(model,mem):
    """Build frozen inference cache; cache footprint separately reported."""
    arm=mem.arm
    if arm=='branch_local':
        keys=normalize(mem.sums);valid=mem.counts>0
        cache=keys.nbytes+valid.nbytes
    elif arm in ('branch_reservoir','mlp_retrieval'):
        used=mem.bank.priorities>=0
        keys=mem.bank.data[used];values=mem.bank.values[used]
        cache=keys.nbytes+values.nbytes
    else:cache=0
    def run(bx):
        with torch.no_grad():
            if arm in ('plain_mlp','mlp_replay'):return model(bx).argmax(-1)
            q=describe(bx.numpy())
            if arm=='mlp_retrieval':
                if not len(keys):return model(bx).argmax(-1)
                mask=torch.from_numpy(values[(q@keys.T).argmax(1)]).bool()
                return model(bx).masked_fill(~mask,-1e9).argmax(-1)
            if arm=='branch_local':winners=unary(q,keys,valid).argmax(-1)
            elif len(keys):winners=values[(q@keys.T).argmax(1)].astype('int64')
            else:winners=np.broadcast_to(model.gates().argmax(-1).numpy(),(len(bx),24)).copy()
            b=torch.tanh(model.first(bx).reshape(-1,24,4))
            h=b.gather(2,torch.from_numpy(winners).long()[:,:,None]).squeeze(-1)
            return model.head(h).argmax(-1)
    return run,cache


def accuracy(pred,y,pairs):
    values=[]
    for a,b in pairs:
        ix=(y==a)|(y==b);values.append(float((pred[ix]==y[ix]).float().mean()))
    return {'all_accuracy':float((pred==y).float().mean()),'pair_accuracy':values}


def latency(run,x):
    results={}
    for size,repeats in [(1,200),(128,50)]:
        bx=x[:size]
        for _ in range(10):run(bx)
        blocks=[]
        for _ in range(5):
            start=time.perf_counter()
            for _ in range(repeats):run(bx)
            blocks.append((time.perf_counter()-start)*1e6/repeats)
        results[str(size)]={'median_us':st.median(blocks),'blocks_us':blocks}
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    arms=['plain_mlp','branch_local','branch_reservoir','mlp_replay','mlp_retrieval']
    protocol={'seeds':[41,42,43,44,45],'orders':['forward','reverse'],'arms':arms,
              'budget':'380,000 bytes maximum for FP32 weights, persistent model buffers and allocated numeric memory arrays. All augmented arms within 1% of cap. Excludes small Python/RNG/counter state, optimizer, transient activations, and inference cache copies; those are not claimed peak RAM. Optimizer tensor storage and inference cache copies reported separately.',
              'stream':'All 60,000 MNIST train images, raw pixels; five disjoint class pairs, two passes per current pair, batch128, same shuffle per seed/order across methods. No past-task revisit beyond explicit replay arm. Full official 10,000 test images after each stage; no test labels or task ID enter inference.',
              'training':'Adam .001, clip5, one update per current minibatch. Branch factual CE + .5 counterfactual CE as before. MLP ordinary CE. Replay concatenates up to128 buffered examples with current batch, mean CE, same number of optimizer updates. Replay sampled before adding current data.',
              'retrieval':'Fixed .05-free hard nearest-centroid local selection or nearest reservoir input. Stable writes after10 identical observed configurations. Branch route comes from existing training feedback gates. MLP output-set configuration is inferred from classes present in current supervised minibatch; no phase ID supplied. At inference nearest input description selects an output set, which masks MLP logits. All ten classes are eligible through input-based retrieval.',
              'memory':'Branch reservoir95 entries:196 float32 pixels,24 uint8 branch indices,float32 priority. MLP retrieval97 entries:196 float32 pixels,10 uint8 class-set indicators,float32 priority. MLP replay98 entries:784 uint8 pixels,uint8 label,float32 priority. Stored pooled descriptions are compressed exemplars, not memory-free learning.',
              'no_tuning':'Fresh seeds and reversed order added; all settings frozen before these tests. No test selection or per-arm hyperparameter searches. Prior pilot informed branch settings, so not an exhaustive optimized-baseline comparison.',
              'compute':'CPU two torch threads. Train wall and process CPU includes shuffling, indexing, model updates and memory acquisition, excludes data loading, evaluation and serialization. Same update count but unequal operations; report runtime and replay presentations. Not equal-wall-time or equal-FLOP training.',
              'scope':'Matched trainable parameter counts and approximately matched total storage, including all retrieval memory. Branch learning still uses counterfactual global feedback. No claim of biological learning or universal superiority.',
              'environment':f'{platform.platform()}, torch {torch.__version__}, numpy {np.__version__}'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for file in [Path(__file__),Path('bdh_cl/scripts/split_mnist_usefulness.py'),Path('bdh_cl/scripts/branch_associative_retrieval.py')]:
        (out/file.name).write_bytes(file.read_bytes())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    records=[]
    for oi,order in enumerate(protocol['orders']):
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        for si,seed in enumerate(protocol['seeds']):
            offset=(si+oi)%len(arms)
            for arm in arms[offset:]+arms[:offset]:
                torch.manual_seed(seed);branch=arm.startswith('branch')
                model=make('local' if branch else 'mlp','hard')
                opt=torch.optim.Adam(model.parameters(),lr=.001);mem=Memory(arm,seed)
                row={'seed':seed,'order':order,'arm':arm,'stages':[],'train_wall_s':0.,'train_cpu_s':0.,
                     'train_presentations':0,'replay_presentations':0,'updates':0}
                for stage,pair in enumerate(pairs):
                    indices=torch.nonzero((y==pair[0])|(y==pair[1]),as_tuple=True)[0]
                    start=time.perf_counter();cpu=time.process_time()
                    for epoch in range(2):
                        gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                        indices_epoch=indices[torch.randperm(len(indices),generator=gen)]
                        for ids in indices_epoch.split(128):
                            bx,by=x[ids],y[ids]
                            route=model.gates().argmax(-1).numpy().astype('uint8') if branch else None
                            if arm=='mlp_replay':
                                replay=mem.replay(len(bx))
                                if replay is not None:
                                    bx_loss=torch.cat([bx,replay[0]]);by_loss=torch.cat([by,replay[1]])
                                    row['replay_presentations']+=len(replay[0])
                                else:bx_loss,by_loss=bx,by
                            else:bx_loss,by_loss=bx,by
                            opt.zero_grad(set_to_none=True)
                            if branch:
                                logits,candidates=model(bx,counterfactual=True)
                                targets=by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                                errors=F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                                loss=F.cross_entropy(logits,by)+.5*errors.mean();feedback=errors.detach().mean(0)
                            else:loss=F.cross_entropy(model(bx_loss),by_loss)
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                            if branch:model.observe(feedback)
                            # route is the PRE-update choice, matching previous online protocol.
                            mem.observe(bx,by,route)
                            row['train_presentations']+=len(bx);row['updates']+=1
                    row['train_wall_s']+=time.perf_counter()-start;row['train_cpu_s']+=time.process_time()-cpu
                    run,cache=predictor(model,mem)
                    pred=torch.cat([run(b) for b in tx.split(512)])
                    row['stages'].append(accuracy(pred,ty,pairs))
                before={k:v.clone() for k,v in model.state_dict().items()}
                before_mem={k:v.copy() for k,v in mem.state().items()}
                run,cache=predictor(model,mem)
                row['latency']=latency(run,tx)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
                assert all(np.array_equal(v,mem.state()[k]) for k,v in before_mem.items())
                row['parameters']=sum(p.numel() for p in model.parameters())
                row['model_bytes']=sum(v.numel()*v.element_size() for v in model.state_dict().values())
                row['memory_bytes']=mem.bytes();row['total_state_bytes']=row['model_bytes']+row['memory_bytes']
                assert row['total_state_bytes']<=380000
                if arm!='plain_mlp':assert row['total_state_bytes']>=.99*380000
                row['inference_cache_copy_bytes']=cache
                row['optimizer_tensor_bytes']=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
                row['final_accuracy']=row['stages'][-1]['all_accuracy']
                last=row['stages'][-1]['pair_accuracy'];oldmask=~((ty==pairs[-1][0])|(ty==pairs[-1][1]))
                pred=torch.cat([run(b) for b in tx.split(512)])
                row['old_accuracy']=float((pred[oldmask]==ty[oldmask]).float().mean())
                row['acquisition']=st.mean(row['stages'][i]['pair_accuracy'][i] for i in range(5))
                row['forgetting']=st.mean(max(row['stages'][s]['pair_accuracy'][i] for s in range(i,4))-last[i] for i in range(4))
                if arm=='mlp_retrieval':
                    with torch.no_grad():
                        original=torch.cat([model(b).argmax(-1) for b in tx.split(512)])
                    row['unmasked_accuracy']=float((original==ty).float().mean())
                prefix=f'{order}_{seed}_{arm}'
                torch.save(model.state_dict(),out/f'{prefix}.pt');np.savez(out/f'{prefix}_memory.npz',**mem.state())
                records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
                print(order,seed,arm,round(row['final_accuracy']*100,2),'train',round(row['train_wall_s'],2),'bytes',row['total_state_bytes'],flush=True)
    lines=['# Fairer branch–MLP comparison: total storage and measured runtime','',
           'Five fresh seeds (41–45), two task orders, ten runs per method. Settings frozen before tests. '
           'No task IDs or test labels at inference. Both orders use the same five digit pairs. '
           'All augmented methods have within 1% of 380,000 bytes of numeric model+memory state.', '',
           '| System | Final accuracy mean ± SD | Old digits | State bytes | Train seconds | Inference μs /128 |',
           '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        rs=[r for r in records if r['arm']==arm];a=[r['final_accuracy']*100 for r in rs]
        lines.append(f'| {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | '
                     f'{100*st.mean(r["old_accuracy"] for r in rs):.2f}% | {rs[0]["total_state_bytes"]:,} | '
                     f'{st.mean(r["train_wall_s"] for r in rs):.2f} | {st.mean(r["latency"]["128"]["median_us"] for r in rs):.2f} |')
    lines+=['','SD pools seeds and orders; do not treat these ten runs as ten independent task sequences. '
            'Old-digit accuracy excludes the last pair in each order (0–7 forward, 2–9 reversed).', '',
            '| System | Forward accuracy | Reverse accuracy |', '|---|---:|---:|']
    for arm in arms:
        vals=[100*st.mean(r['final_accuracy'] for r in records if r['arm']==arm and r['order']==o) for o in ['forward','reverse']]
        lines.append(f'| {arm} | {vals[0]:.2f}% | {vals[1]:.2f}% |')
    lines+=['','## What is matched','',
            'Same raw training data, current-stream sample order, two epochs per task, optimizer/lr/clipping, '
            'update count and parameter count to within0.1%. Additional memory is charged by bytes, including '
            'stored inputs, labels/configurations and reservoir priorities. Raw replay images are stored '
            'as uint8, not artificially inflated float32. Branch models retain their auxiliary objective; '
            'MLP replay uses additional stored examples. This is a system comparison, not an isolated '
            'architecture ablation.', '',
            'The plain MLP is a reference below the memory cap. Branch local stores branch-conditioned means; '
            'branch reservoir retrieves online branch configurations. MLP replay stores 98 labelled images '
            'and replays a minibatch per update. MLP retrieval stores 97 pooled images with the set of labels '
            'present in their training minibatch; image similarity selects a candidate output set at inference. '
            'This is inferred context, not an oracle task ID. It benefits from the disjoint-class stream and '
            'would need separate testing when classes overlap. No extra phase-boundary writes occur.', '',
            'All learned numeric model/memory arrays are included. Small Python object overhead, RNG state '
            'and scalar counters are excluded; a few bytes of bookkeeping do not explain the accuracy gap. '
            'Optimizer tensor bytes and derived inference cache copies are recorded separately in results.json. '
            'The caches can replace their source arrays at deployment but this implementation holds both. '
            'This is not peak RAM accounting.', '',
            '## Compute and scope','',
            'Runtime includes memory acquisition/replay and training, excluding evaluation/checkpoint I/O. '
            'Inference includes pooling, retrieval and final classification, with warmed caches. Five timing '
            'blocks at batches1/128. CPU local implementation, not GPU/energy measurements. Same optimizer '
            'update count is not equal compute; replay adds roughly120,000 stored-example presentations. '
            'All per-run timings and counts are saved. No equal-wall-time tuning was performed.', '',
            'Branch settings were developed in earlier pilots. These baseline settings are fixed, simple '
            'controls rather than tuned state-of-the-art methods. The new seeds and second order improve '
            'the comparison but do not establish generality beyond Split MNIST. No biological plausibility '
            'claim follows from these experiments.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:13]),flush=True)


if __name__=='__main__':main()
