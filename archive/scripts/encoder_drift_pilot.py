"""Task-free encoder drift experiment with acquisition-time latent replay.

Matched-size ordinary and dendritic encoders share the same classifier. Probe
snapshots belong exclusively to the evaluator and never enter learner updates.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bdh_cl.data.mnist import build_pools
from bdh_cl.scripts.taskfree_latent_pilot import LatentMemory

ARMS = [(architecture, frozen, memory) for architecture in ('mlp','sum','max')
        for frozen, memory in ((False,False),(False,True),(True,True))]


class Encoder(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'mlp':
            self.network = nn.Sequential(nn.Linear(196,193),nn.ReLU(),nn.Linear(193,64),nn.ReLU())
        else:
            self.weight = nn.Parameter(torch.randn(64,4,196)/math.sqrt(196))
            self.bias = nn.Parameter(torch.zeros(64,4))

    def branches(self,x):
        if self.kind == 'mlp':return None
        return F.relu(F.linear(x,self.weight.flatten(0,1),self.bias.flatten())).reshape(len(x),64,4)

    def forward(self,x):
        if self.kind == 'mlp':z=self.network(x)
        else:
            a=self.branches(x)
            z=a.sum(-1)/2 if self.kind=='sum' else a.max(-1).values
        # Fixed per-example normalization: no moving statistics or future data.
        return 3*F.normalize(z,dim=-1)


class Learner:
    def __init__(self, architecture, frozen, replay, seed, capacity=100):
        torch.manual_seed(seed)
        self.encoder=Encoder(architecture).requires_grad_(not frozen)
        # Head starts identically across encoder architectures.
        torch.manual_seed(seed+1000)
        self.head=nn.Sequential(nn.Linear(64,115),nn.ReLU(),nn.Linear(115,10))
        self.optimizer=torch.optim.SGD([
            {'params':list(self.encoder.parameters()),'lr':.05},
            {'params':list(self.head.parameters()),'lr':.05}])
        self.memory=LatentMemory('reservoir' if replay else 'none',capacity=capacity,seed=seed+2000)
        self.updates,self.examples,self.correct,self.replayed=0,0,0,0

    def update(self,x,y):
        self.optimizer.zero_grad(set_to_none=True)
        z=self.encoder(x)
        logits=self.head(z)
        self.correct+=int((logits.detach().argmax(-1)==y).sum())
        loss=F.cross_entropy(logits,y)
        old=self.memory.sample()
        if old is not None:
            oldz,oldy=old
            loss=loss+F.cross_entropy(self.head(oldz),oldy)
            self.replayed+=len(oldy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.head.parameters()),5)
        self.optimizer.step()
        # Save the actual prediction-time features, not a refresh after update.
        self.memory.observe(z.detach(),y)
        self.updates+=1;self.examples+=len(y)


@torch.no_grad()
def evaluate(learner,tests):
    return [float((learner.head(learner.encoder(x)).argmax(-1)==y).float().mean()) for x,y in tests]


def cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return float((a.T@b).square().sum()/((a.T@a).norm()*(b.T@b).norm()).clamp_min(1e-12))


def probes(test):
    x,y=test
    half=len(y)//2
    ix=torch.cat((torch.arange(32),torch.arange(half,half+32)))
    return x[ix],y[ix]


@torch.no_grad()
def drift(learner,tests,references):
    result=[]
    for i,(old,oldbranches) in enumerate(references):
        x,y=probes(tests[i])
        current=learner.encoder(x)
        row={'pair':i,'relative_l2':float((current-old).norm()/old.norm().clamp_min(1e-12)),
             'cosine':float(F.cosine_similarity(current,old).mean()),'linear_cka':cka(old,current),
             'stale_feature_accuracy':float((learner.head(old).argmax(-1)==y).float().mean()),
             'current_feature_accuracy':float((learner.head(current).argmax(-1)==y).float().mean())}
        if oldbranches is not None:
            a=learner.encoder.branches(x)
            active=(a.max(-1).values>0)&(oldbranches.max(-1).values>0)
            row['winner_agreement']=float((a.argmax(-1)[active]==oldbranches.argmax(-1)[active]).float().mean()) if active.any() else None
        result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--samples-per-domain',type=int,default=4000)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.smoke:args.samples_per_domain,args.seeds=64,'0'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    seeds=[int(s) for s in args.seeds.split(',')]
    protocol={**vars(args),'seeds':seeds,'arms':ARMS,
              'encoder':'MLP 196->193->64 versus 64 neurons x 4 branches; 50,437 versus 50,432 parameters; output L2 norm fixed to 3',
              'classifier':'shared architecture 64->115->10; same initialization across all arms',
              'stream':'one pass through 4000 examples per digit pair 01/23/45/67/89; neighboring domains interleaved in 512-example windows; then 512 new 01 examples',
              'learner':'update(x,y), no boundaries, one persistent SGD optimizer; constant encoder/head rates .05; batch32',
              'memory':'100 prediction-time latents, Algorithm R, FP16 plus int64 labels; 13,600 bytes; no stored pixels, feature refresh or decoder; 16 replayed samples/update',
              'probes':'64 fixed official-test examples per pair, snapshots only in evaluator at acquisition landmark; measure cosine, relative L2, linear CKA, stale/current head accuracy and branch winner agreement',
              'retention':'global ten-class accuracy at 20,000 presentations, before return learning; 200 test images/pair',
              'limits':'no pretraining; supervised labels; fixed ten-class output; dendrites use sum or max only, no rewiring/protection; frozen encoders are paired random controls',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    pools=build_pools(ROOT/'data_cache/mnist',side=14,task_tags=False);start=time.perf_counter()
    for seed in seeds:
        rng=np.random.default_rng(seed);generator=torch.Generator().manual_seed(seed+300)
        blocks,tests,indices=[],[],[]
        def tensor(images):return torch.tensor(images.copy(),dtype=torch.float32).flatten(1)/255
        # Consume the same generator draw as the earlier fixed-feature harness.
        torch.randn(196,64,generator=generator)
        for p in pools:
            ix=rng.choice(len(p.train_labels),args.samples_per_domain+512,False)
            te=np.concatenate([rng.choice(np.flatnonzero(p.test_labels==c),100,False) for c in p.classes])
            blocks.append((tensor(p.train_images[ix[:args.samples_per_domain]]),torch.tensor(p.train_labels[ix[:args.samples_per_domain]].astype(np.int64))))
            tests.append((tensor(p.test_images[te]),torch.tensor(p.test_labels[te].astype(np.int64))))
            indices.append({'train':ix.tolist(),'test':te.tolist()})
        x=torch.cat([v[0] for v in blocks]);y=torch.cat([v[1] for v in blocks])
        for i in range(1,5):
            middle=i*args.samples_per_domain;half=min(256,args.samples_per_domain//4)
            ix=torch.randperm(2*half,generator=generator)+middle-half
            x[middle-half:middle+half]=x[ix].clone();y[middle-half:middle+half]=y[ix].clone()
        ri=np.array(indices[0]['train'][-512:]);p=pools[0]
        x=torch.cat((x,tensor(p.train_images[ri])));y=torch.cat((y,torch.tensor(p.train_labels[ri].astype(np.int64))))
        (out/f'indices_seed{seed}.json').write_text(json.dumps(indices))
        report={'seed':seed,'arms':{}}
        for architecture,frozen,replay in ARMS:
            arm=f'{architecture}_{"frozen" if frozen else "moving"}_{"latent" if replay else "none"}'
            learner=Learner(architecture,frozen,replay,seed);references=[];curve=[];t0=time.perf_counter()
            for offset in range(0,len(y),32):
                learner.update(x[offset:offset+32],y[offset:offset+32])
                if learner.examples<=5*args.samples_per_domain and learner.examples%args.samples_per_domain==0:
                    # This runner-only landmark does not influence any update.
                    pair=learner.examples//args.samples_per_domain-1
                    with torch.no_grad():
                        px,_=probes(tests[pair]);features=learner.encoder(px).clone();branches=learner.encoder.branches(px)
                        references.append((features,None if branches is None else branches.clone()))
                if learner.updates%100==0 or learner.examples==5*args.samples_per_domain or offset+32>=len(y):
                    curve.append({'examples':learner.examples,'accuracy':evaluate(learner,tests),'drift':drift(learner,tests,references)})
            before=next(row for row in curve if row['examples']==5*args.samples_per_domain)
            olddrift=before['drift'][:4]
            report['arms'][arm]={'curve':curve,'before_return_accuracy':float(np.mean(before['accuracy'])),
                                 'old_accuracy_before_return':float(np.mean(before['accuracy'][:4])),
                                 'new_accuracy_before_return':before['accuracy'][4],
                                 'final_accuracy':float(np.mean(curve[-1]['accuracy'])),
                                 'old_feature_cosine':float(np.mean([d['cosine'] for d in olddrift])),
                                 'old_feature_cka':float(np.mean([d['linear_cka'] for d in olddrift])),
                                 'stale_minus_current_accuracy':float(np.mean([d['stale_feature_accuracy']-d['current_feature_accuracy'] for d in olddrift])),
                                 'parameters':sum(p.numel() for module in (learner.encoder,learner.head) for p in module.parameters()),
                                 'trainable_parameters':sum(p.numel() for module in (learner.encoder,learner.head) for p in module.parameters() if p.requires_grad),
                                 'memory_bytes':learner.memory.bytes(),'updates':learner.updates,'prequential_accuracy':learner.correct/learner.examples,'seconds':time.perf_counter()-t0}
            (out/f'seed{seed}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'seed={seed} {arm} retained={report["arms"][arm]["before_return_accuracy"]:.3f} cosine={report["arms"][arm]["old_feature_cosine"]:.3f}',flush=True)
    summary={}
    for arm in report['arms']:
        rows=[json.loads((out/f'seed{s}.json').read_text())['arms'][arm] for s in seeds]
        summary[arm]={k:{'mean':float(np.mean([r[k] for r in rows])),
                        'sd':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None,
                        'per_seed':[r[k] for r in rows]} for k in ('before_return_accuracy','old_accuracy_before_return','new_accuracy_before_return','old_feature_cosine','old_feature_cka','stale_minus_current_accuracy','final_accuracy','seconds')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(f'COMPLETE {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':main()
