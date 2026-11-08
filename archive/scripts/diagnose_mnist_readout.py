"""Offline diagnostic ONLY: joint-data probes on frozen hybrid5 representations.
Never a replay-free benchmark result. No fitted heads are saved.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model

@torch.no_grad()
def features(m,x):
 mixed=[];allbranches=[]
 for bx in x.split(512):
  raw=m.responses(bx,False);z=raw.tanh();mixed.append((z*raw.abs().softmax(-1)).sum(-1));allbranches.append(z.flatten(1))
 return torch.cat(mixed),torch.cat(allbranches)


def ridge_score(x,y,v,vy):
 x=torch.cat([x,torch.ones(len(x),1)],1);v=torch.cat([v,torch.ones(len(v),1)],1)
 head=torch.linalg.solve(x.T@x+torch.eye(x.shape[1]),x.T@F.one_hot(y,10).float())
 return float(((v@head).argmax(-1)==vy).float().mean())


def main():
 torch.set_num_threads(2);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(ids):return (torch.tensor(raw['train_images'][ids].copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(raw['train_labels'][ids].copy(),dtype=torch.long)
 x,y=ds(split['train']);vx,vy=ds(split['validation']);rows=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   m=Model('baseline',seed);saved=torch.load(Path('results/mnist_gradient_branch_scoring_20261106')/f'hybrid5_{seed}_{first}.pt',weights_only=True);m.load_state_dict(saved['state']);m.requires_grad_(False)
   h,z=features(m,x);vh,vz=features(m,vx)
   with torch.no_grad():
    logits=m.head(vh);pred=logits.argmax(-1);mask=torch.arange(10)[None]//2==vy[:,None]//2;masked=logits.masked_fill(~mask,-float('inf')).argmax(-1);wrong=pred!=vy
    row=dict(seed=seed,first=first,current_head=float((pred==vy).float().mean()),true_pair_mask_diagnostic=float((masked==vy).float().mean()),cross_pair_fraction_of_mistakes=float(((pred[wrong]//2)!=(vy[wrong]//2)).float().mean()),joint_ridge_mixed168=ridge_score(h,y,vh,vy),joint_ridge_branches672=ridge_score(z,y,vz,vy))
   rows.append(row);print(row,flush=True)
 out=Path('results/mnist_readout_diagnosis_20261106');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(scope='DIAGNOSTIC ONLY, all original TRAIN labels used jointly; not replay-free result',checkpoints='six hybrid5 sparse MLPs',train_examples=len(y),validation_examples=len(vy),ridge=1.,bias='constant feature, also regularized',representations=['current soft magnitude mixed168','all672 unweighted tanh branch outputs'],mask='True-pair mask diagnostic requires label/task and is not image-only'),indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
