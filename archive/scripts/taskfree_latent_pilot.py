"""Boundary-free learner with fixed random features and bounded latent memory.

Only the runner constructs distribution shifts. Learner.update receives z,y;
there are no stage IDs, resets, task-dependent schedules or teacher snapshots.
"""
import argparse
import copy
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
from bdh_cl.scripts.dendritic_cl_pilot import Student, distill

ARMS = {
    "mlp_none": ("mlp", "none", False),
    "mlp_latent": ("mlp", "reservoir", False),
    "mlp_centroids": ("mlp", "centroids", False),
    "mlp_ema_current": ("mlp", "none", True),
    "mlp_latent_ema": ("mlp", "reservoir", True),
    "sum_latent": ("sum", "reservoir", False),
    "max_none": ("max", "none", False),
    "max_latent": ("max", "reservoir", False),
    "max_centroids": ("max", "centroids", False),
    "max_latent_ema": ("max", "reservoir", True),
}


class LatentMemory:
    def __init__(self, mode, width=64, capacity=100, seed=0):
        self.mode, self.capacity, self.seen, self.size = mode, capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(capacity, width, dtype=torch.float16) if mode == "reservoir" else None
        self.y = torch.zeros(capacity, dtype=torch.int64) if mode == "reservoir" else None
        self.mean = torch.zeros(10, width) if mode == "centroids" else None
        self.count = torch.zeros(10, dtype=torch.int64) if mode == "centroids" else None

    def sample(self, n=16):
        if self.mode == "reservoir" and self.size:
            ix = self.rng.integers(self.size, size=n)
            return self.z[ix].float(), self.y[ix]
        if self.mode == "centroids":
            labels = (self.count > 0).nonzero().flatten()
            if len(labels):
                ix = labels[self.rng.integers(len(labels), size=n)]
                return self.mean[ix].clone(), ix
        return None

    @torch.no_grad()
    def observe(self, z, y):
        if self.mode == "none":
            return
        for zz, yy in zip(z, y):
            self.seen += 1
            if self.mode == "reservoir":
                ix = self.size if self.size < self.capacity else int(self.rng.integers(self.seen))
                if ix < self.capacity:
                    self.z[ix], self.y[ix] = zz.detach().half(), yy
                self.size = min(self.size+1, self.capacity)
            else:
                label = int(yy)
                self.count[label] += 1
                self.mean[label] += (zz.detach()-self.mean[label])/self.count[label]

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.z,self.y,self.mean,self.count) if t is not None)


class Learner:
    def __init__(self, architecture, memory, ema, seed, capacity=100):
        torch.manual_seed(seed)
        # 8,635 MLP versus 8,650 dendritic trainable parameters.
        self.model = nn.Sequential(nn.Linear(64,115),nn.ReLU(),nn.Linear(115,10)) if architecture == "mlp" else Student(64,32,4,architecture)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.05)
        self.memory = LatentMemory(memory, capacity=capacity, seed=seed+100)
        self.teacher = copy.deepcopy(self.model).requires_grad_(False) if ema else None
        self.updates, self.replayed, self.correct, self.presentations = 0,0,0,0

    def update(self, z, y):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(z)
        self.correct += int((logits.detach().argmax(-1)==y).sum())
        self.presentations += len(y)
        loss = F.cross_entropy(logits,y)
        old = self.memory.sample()
        if old is not None:
            rz, ry = old
            replay_logits = self.model(rz)
            loss = loss + F.cross_entropy(replay_logits,ry)
            self.replayed += len(ry)
        if self.teacher is not None:
            # No future teacher information and no boundary-specific refresh.
            tz = old[0] if old is not None else z
            with torch.no_grad():
                target = self.teacher(tz)
            loss = loss + distill(self.model(tz),target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)
        self.optimizer.step()
        if self.teacher is not None:
            with torch.no_grad():
                for t,p in zip(self.teacher.parameters(),self.model.parameters()):
                    t.mul_(.99).add_(p,alpha=.01)
        # Read before write: no current example can enter its own replay target.
        self.memory.observe(z,y)
        self.updates += 1


@torch.no_grad()
def accuracy(model, tests):
    return [float((model(z).argmax(-1)==y).float().mean()) for z,y in tests]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True)
    ap.add_argument("--seeds",default="0,1,2")
    ap.add_argument("--samples-per-domain",type=int,default=4000)
    ap.add_argument("--memory",type=int,default=100)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    if args.smoke:args.seeds,args.samples_per_domain="0",64
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    seeds=[int(v) for v in args.seeds.split(',')]
    protocol={**vars(args),"seeds":seeds,"arms":ARMS,
              "stream":"single pass: digit pairs 01/23/45/67/89 then 512 new training examples of 01; blend adjacent domains over transition windows; learner sees no boundaries",
              "encoder":"fixed random 196-to-64 ReLU feature map, normalized with fixed scale, no pretraining or future fitting",
              "memory":"Algorithm R bounded latent reservoir, FP16 features and labels; alternative one running feature centroid per observed digit",
              "decoder":False,"ema":"0.99 update every minibatch, no old-student snapshots; CE plus T=2 KL",
              "optimizer":"one SGD instance, constant lr .05, no resets; batch32; no consolidation at boundaries",
              "evaluation":"global ten-class accuracy; fixed periodic checks every 100 minibatches, before return (runner-only readback), and final; labels never used for inference routing",
              "limitations":"fixed random encoder cannot acquire new representations; fixed 10-class centroid budget; aggregate centroids are not generative replay; not full-model continual learning",
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    pools=build_pools(ROOT/'data_cache/mnist',side=14,task_tags=False)
    start=time.perf_counter()
    for seed in seeds:
        rng=np.random.default_rng(seed)
        gen=torch.Generator().manual_seed(seed+300)
        projection=torch.randn(196,64,generator=gen)/math.sqrt(196)
        def encode(images):
            x=torch.tensor(images.copy(),dtype=torch.float32).flatten(1)/255
            return F.relu(x@projection)*3
        blocks,tests,indices=[],[],[]
        for p in pools:
            ix=rng.choice(len(p.train_labels),args.samples_per_domain+512,False)
            te=np.concatenate([rng.choice(np.flatnonzero(p.test_labels==c),100,False) for c in p.classes])
            blocks.append((encode(p.train_images[ix[:args.samples_per_domain]]),torch.tensor(p.train_labels[ix[:args.samples_per_domain]].astype(np.int64))))
            tests.append((encode(p.test_images[te]),torch.tensor(p.test_labels[te].astype(np.int64))))
            indices.append({'train':ix.tolist(),'test':te.tolist()})
        # Mix neighboring distribution windows; evaluation labels stay in runner.
        z=torch.cat([v[0] for v in blocks]);y=torch.cat([v[1] for v in blocks])
        for i in range(1,5):
            mid=i*args.samples_per_domain;half=min(256,args.samples_per_domain//4)
            ix=torch.randperm(2*half,generator=gen)+mid-half
            z[mid-half:mid+half]=z[ix].clone();y[mid-half:mid+half]=y[ix].clone()
        a=pools[0];ri=np.array(indices[0]['train'][-512:])
        z=torch.cat((z,encode(a.train_images[ri])));y=torch.cat((y,torch.tensor(a.train_labels[ri].astype(np.int64))))
        (out/f'indices_seed{seed}.json').write_text(json.dumps(indices))
        report={'seed':seed,'arms':{},'frozen_encoder_bytes':projection.numel()*4,'stream_examples':len(y)}
        for arm,(arch,mem,ema) in ARMS.items():
            learner=Learner(arch,mem,ema,seed,args.memory)
            curve=[];elapsed=time.perf_counter()
            for offset in range(0,len(y),32):
                learner.update(z[offset:offset+32],y[offset:offset+32])
                if learner.updates%100==0 or learner.presentations==5*args.samples_per_domain or offset+32>=len(y):
                    curve.append({'updates':learner.updates,'examples':learner.presentations,'accuracy':accuracy(learner.model,tests)})
            count=sum(p.numel() for p in learner.model.parameters())
            before_return = next((row for row in curve if row['examples']==5*args.samples_per_domain), None)
            report['arms'][arm]={'parameters':count,'curve':curve,'final_accuracy':float(np.mean(curve[-1]['accuracy'])),
                                 'before_return_accuracy':float(np.mean(before_return['accuracy'])) if before_return else None,
                                 'prequential_accuracy':learner.correct/learner.presentations,
                                 'memory_bytes':learner.memory.bytes(),'teacher_bytes':4*count if ema else 0,
                                 'replay_presentations':learner.replayed,'updates':learner.updates,
                                 'seconds':time.perf_counter()-elapsed}
            (out/f'seed{seed}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'seed={seed} {arm} final={report["arms"][arm]["final_accuracy"]:.3f} online={report["arms"][arm]["prequential_accuracy"]:.3f}',flush=True)
    summary={}
    for arm in ARMS:
        rows=[json.loads((out/f'seed{s}.json').read_text())['arms'][arm] for s in seeds]
        summary[arm]={k:{'mean':float(np.mean([r[k] for r in rows])),
                        'sd':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None,
                        'per_seed':[r[k] for r in rows]} for k in ('final_accuracy','prequential_accuracy','seconds')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'COMPLETE {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':main()
