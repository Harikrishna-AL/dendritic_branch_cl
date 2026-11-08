"""Rate-based allocation/clustering abstraction, not a spiking-paper reproduction.

Unsupervised update(x), fixed storage, no replay, labels or task boundaries.
Novelty and usage explicitly allocate slots; generic slots implement the same
algorithm to prevent mistaking generic memory packing for a dendritic result.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ARMS = ('independent','competitive','competitive_stable','generic_slots')


class BranchMemory:
    def __init__(self, mode, seed, inputs=320, branches=12, contacts=40, noise=.002):
        self.mode,self.contacts,self.noise=mode,contacts,noise
        self.rng=np.random.default_rng(seed)
        self.trace=self.rng.uniform(0,.01,(branches,inputs)).astype(np.float32)
        self.weights=np.zeros_like(self.trace)
        self.usage=np.zeros(branches,dtype=np.int64)
        self.threshold=.55
        self.rewire()

    def rewire(self):
        # Top-contact projection is a global branch operation, not literal STDP.
        selected=np.argpartition(self.trace,-self.contacts,axis=1)[:,-self.contacts:]
        mask=np.zeros_like(self.weights,dtype=bool)
        np.put_along_axis(mask,selected,True,axis=1)
        self.weights[:]=self.trace*mask
        self.weights/=np.maximum(np.linalg.norm(self.weights,axis=1,keepdims=True),1e-12)

    def scores(self,x):
        x=np.asarray(x,dtype=np.float32)
        return (x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12))@self.weights.T

    def update(self,x):
        target=np.asarray(x,dtype=np.float32)
        target=target/max(np.linalg.norm(target),1e-12)
        similarities=self.scores(target)
        if self.mode=='independent':
            active=np.arange(len(self.weights))
            allocate=self.usage[active]==0
        else:
            best=int(np.argmax(similarities))
            if similarities[best]>=self.threshold and self.usage[best]>0:
                chosen=best
                allocate=np.array([False])
            else:
                # Explicit least-used allocation: not claimed to emerge from spikes.
                chosen=int(np.argmin(self.usage))
                allocate=np.array([True])
            active=np.array([chosen])
        # Explicit allocation seeds a new receptive field. It is not an
        # emergent biological rewiring result; this is a mechanism abstraction.
        for branch in active[allocate]:
            self.trace[branch]=target
            self.usage[branch]=0
        stabilized=self.mode in ('competitive_stable','generic_slots')
        plasticity=1/(1+self.usage/20) if stabilized else np.ones(len(self.usage))
        # Background stochastic rewiring operates even on unused branches.
        self.trace+=self.rng.normal(0,self.noise,self.trace.shape).astype(np.float32)*plasticity[:,None]
        self.trace=np.maximum(self.trace,0)
        rate=.1*plasticity[active]
        self.trace[active]=(1-rate[:,None])*self.trace[active]+rate[:,None]*target
        self.usage[active]+=1
        self.rewire()

    def bytes(self):return self.weights.nbytes+self.trace.nbytes+self.usage.nbytes


def assemblies(patterns, inputs, overlap, seed):
    rng=np.random.default_rng(seed)
    shared=int(40*overlap)
    assert shared+patterns*(40-shared)<=inputs
    order=rng.permutation(inputs)
    result=np.zeros((patterns,inputs),dtype=np.float32)
    for p in range(patterns):
        result[p,order[:shared]]=1
        result[p,order[shared+p*(40-shared):shared+(p+1)*(40-shared)]]=1
    return result


def noisy(pattern,rng,n=1):
    result=np.broadcast_to(pattern,(n,len(pattern))).copy()
    drop=rng.random(result.shape)<.15
    result[drop]=0
    result[rng.random(result.shape)<.01]=1
    return result.astype(np.float32)


def evaluate(model,queries,novel):
    scores=np.stack([model.scores(q) for q in queries])
    recalls=(scores.max(-1)>=model.threshold).mean(-1)
    means=scores.mean(1)
    branch_preferences=means.argmax(0)
    # Occupation uses selectivity above threshold, not merely largest background score.
    occupied=means.max(0)>=model.threshold
    represented=[int(np.sum(occupied & (branch_preferences==p))) for p in range(len(queries))]
    return {'recall':recalls.tolist(),'represented_patterns':int((recalls>=.8).sum()),
            'branches_per_pattern':represented,'unused_selective_branches':int((~occupied).sum()),
            'novel_false_positive':float((model.scores(novel).max(-1)>=model.threshold).mean())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--noise',type=float,default=.002);ap.add_argument('--presentations',type=int,default=250)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.smoke:args.seeds,args.presentations='0',10
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol={**vars(args),'arms':ARMS,'scenarios':{'disjoint':0.,'shared30':.3,'capacity4':0.},
              'patterns':8,'inputs':320,'branches':12,'contacts_per_branch':40,'query_examples_per_pattern':100,
              'recognition':'fixed cosine threshold .55; nonlinear branch recognition; no trained classifier',
              'learning':'independent updates all branches; competition uses best recognized branch or least-used branch; novelty explicitly replaces that receptive field and resets its usage; stability reduces Hebbian rate and stochastic turnover by 1/(1+usage/20)',
              'structural_rule':'retain dense coactivity scores; publish strongest40 nonnegative contacts per branch with L2 normalization; explicit heuristic, not Langevin spiking equations',
              'data':'sequential pattern stream, 15% input dropout plus 1% background noise; true pattern identities accessible only to runner',
              'intervention':'after learning, average synaptic weights across branches then project each back to top40; learner state preserved outside evaluator copy',
              'generic_control':'identical allocation/update/storage without biological naming; expected exact equivalence',
              'limits':'engineered novelty allocator; 12 branches except capacity4 uses four; binary assemblies; stable sensory coordinates; no classifier, BDH, teacher or large-scale claim',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter();records=[]
    for scenario,overlap in protocol['scenarios'].items():
        for seed in [int(s) for s in args.seeds.split(',')]:
            patterns=assemblies(8,320,overlap,seed)
            rng=np.random.default_rng(seed+100)
            stream=np.concatenate([noisy(p,rng,args.presentations) for p in patterns])
            queryrng=np.random.default_rng(seed+200)
            queries=np.stack([noisy(p,queryrng,100) for p in patterns])
            novel=np.zeros((100,320),dtype=np.float32)
            for row in novel:row[queryrng.choice(320,40,False)]=1
            np.savez_compressed(out/f'{scenario}_seed{seed}_data.npz',patterns=patterns,stream=stream,queries=queries,novel=novel)
            report={'scenario':scenario,'seed':seed,'arms':{}}
            models={}
            for arm in ARMS:
                branches=4 if scenario=='capacity4' else 12
                model=BranchMemory(arm,seed,branches=branches,noise=args.noise);curve=[];references=[]
                for t,x in enumerate(stream):
                    model.update(x)
                    if (t+1)%args.presentations==0:
                        curve.append(evaluate(model,queries,novel));references.append(model.weights.copy())
                final=curve[-1]
                # Feature readout for previous patterns must remain the same after later learning.
                preservation=[]
                for p,weights in enumerate(references[:-1]):
                    key=patterns[p]/np.linalg.norm(patterns[p])
                    before=weights@key;now=model.weights@key
                    preservation.append(float(np.linalg.norm(now-before)/max(np.linalg.norm(before),1e-12)))
                mixed=BranchMemory(arm,seed,branches=branches,noise=args.noise)
                mixed.trace[:]=model.trace.mean(0)[None]
                mixed.usage[:]=model.usage
                mixed.rewire()
                disrupted=evaluate(mixed,queries,novel)
                report['arms'][arm]={'curve':curve,'final':final,'mean_old_recall':float(np.mean(final['recall'][:-1])),
                                     'new_recall':final['recall'][-1], 'old_response_drift':float(np.mean(preservation)),
                                     'after_synaptic_mixing':disrupted,'state_bytes':model.bytes(),
                                     'usage':model.usage.tolist()}
                np.savez(out/f'{scenario}_seed{seed}_{arm}.npz',weights=model.weights,trace=model.trace,usage=model.usage)
                models[arm]=model
                print(f'{scenario} seed={seed} {arm} represented={final["represented_patterns"]}/8 old={report["arms"][arm]["mean_old_recall"]:.3f}',flush=True)
            np.testing.assert_array_equal(models['competitive_stable'].weights,models['generic_slots'].weights)
            report['generic_control_exactly_equivalent']=True
            (out/f'{scenario}_seed{seed}.json').write_text(json.dumps(report,indent=2)+'\n');records.append(report)
    summary={}
    for scenario in protocol['scenarios']:
        summary[scenario]={}
        for arm in ARMS:
            rows=[r['arms'][arm] for r in records if r['scenario']==scenario]
            values={'represented_patterns':[r['final']['represented_patterns'] for r in rows],
                    'old_recall':[r['mean_old_recall'] for r in rows], 'new_recall':[r['new_recall'] for r in rows],
                    'false_positive':[r['final']['novel_false_positive'] for r in rows],
                    'mixed_old_recall':[float(np.mean(r['after_synaptic_mixing']['recall'][:-1])) for r in rows]}
            summary[scenario][arm]={k:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)) if len(v)>1 else None,'per_seed':v} for k,v in values.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(f'COMPLETE {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':main()
