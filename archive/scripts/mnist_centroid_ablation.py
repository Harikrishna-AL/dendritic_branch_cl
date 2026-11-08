import json,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective

@torch.no_grad()
def feats(m,x):
 a=[];b=[]
 for bx in x.split(512):
  r=m.responses(bx,False);z=r.tanh();a.append((z*r.abs().softmax(-1)).sum(-1));b.append(z.flatten(1))
 return [torch.cat(a),torch.cat(b)]

def main():
 torch.set_num_threads(2);out=Path('results/mnist_centroid_ablation_20261106');out.mkdir(exist_ok=True);raw=load_raw('data_cache/mnist');s=np.load(SOURCE/'split_indices.npz')
 def ds(im,y):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(y.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][s['train']],raw['train_labels'][s['train']]);vx,vy=ds(raw['train_images'][s['validation']],raw['train_labels'][s['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);rows=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   torch.manual_seed(seed);m=Model('baseline',seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);means=[torch.zeros(10,168),torch.zeros(10,672)];step=0
   for stage,pair in enumerate(range(5) if first==0 else reversed(range(5))):
    ix=torch.where(y//2==pair)[0]
    for epoch in range(2):
     ids=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
     for batch in ids.split(128):
      opt.zero_grad(set_to_none=True);loss,feedback,_=objective(m,x[batch],y[batch],'hybrid5',step);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback);step+=1
    f=feats(m,x[ix])
    for j in range(2):
     for c in [2*pair,2*pair+1]:means[j][c]=f[j][y[ix]==c].mean(0)
   train=feats(m,x);valid=feats(m,vx);test=feats(m,tx)
   for j,dim in enumerate([168,672]):
    final=torch.stack([train[j][y==c].mean(0) for c in range(10)])
    for mode,mu in [('continual',means[j]),('joint_diagnostic',final)]:
     for metric in ['euclidean','cosine']:
      def score(f,labels):
       logits=2*f@mu.T-mu.square().sum(-1) if metric=='euclidean' else F.normalize(f,dim=-1)@F.normalize(mu,dim=-1).T
       return float((logits.argmax(-1)==labels).float().mean())
      rows.append(dict(seed=seed,first=first,dimensions=dim,mode=mode,metric=metric,validation=score(valid[j],vy),test=score(test[j],ty)))
   (out/'results.json').write_text(json.dumps(rows,indent=2));print(seed,first,flush=True)
 lines=['# Centroid classifier ablation','','Six hybrid5 sparse MLP runs, identical training recipe. Continual means computed from current-task training examples after its two epochs, then retained unchanged; no old examples revisited for these means. This requires an extra current-task feature pass. Joint diagnostic means use all training examples after final training and are NOT replay-free results. No test-based hyperparameter tuning; all prespecified cells reported.','','| Means | Features | Distance | Validation | Test |','|---|---:|---|---:|---:|']
 for mode in ['continual','joint_diagnostic']:
  for dim in [168,672]:
   for metric in ['euclidean','cosine']:
    r=[a for a in rows if a['mode']==mode and a['dimensions']==dim and a['metric']==metric];v=[statistics.mean(a['test']*100 for a in r if a['seed']==seed) for seed in [101,102,103]];lines.append(f'| {mode} | {dim} | {metric} | {statistics.mean(a["validation"] for a in r)*100:.2f}% | {statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}% |')
 lines+=['','SD uses three seed-level averages across task orders. All ten classes compete, no task identity or masking. Centroid storage: 6720 / 26880 bytes float32, plus counts if needed. Core branch network retained. Previously inspected test set; exploratory ablation. Joint means are diagnostic only.']
 (out/'REPORT.md').write_text('\n'.join(lines));(out/'source.py').write_bytes(Path(__file__).read_bytes());print('\n'.join(lines))
if __name__=='__main__':main()
