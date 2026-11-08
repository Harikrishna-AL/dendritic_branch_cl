"""Combined image-driven branch training on the sparse four-branch MNIST MLP."""
import argparse,json,time,statistics,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_sparse_reproducibility import SparseBranches

ARMS=['baseline','magnitude_joint','selector_hard','selector_soft','selector_soft_slow']
class Model(SparseBranches):
 def __init__(self,arm,seed):
  super().__init__(False,196,seed);self.variant=arm
  if arm.startswith('selector'):
   self.selector=nn.Sequential(nn.Linear(784,32),nn.Tanh(),nn.Linear(32,672))
 def prediction(self,x,raw=None,mode=None):
  raw=self.responses(x,False) if raw is None else raw;z=raw.tanh();mode=mode or self.variant
  if mode in ['baseline','magnitude_joint','magnitude_hard']:weights=F.one_hot(raw.abs().argmax(-1),4).float()
  elif mode=='mean':weights=torch.full_like(raw,.25)
  elif mode=='magnitude_soft':weights=(raw.abs()/1.).softmax(-1)
  else:
   scores=self.selector(x).reshape(-1,168,4);soft=scores.softmax(-1)
   if mode=='selector_hard':weights=F.one_hot(scores.argmax(-1),4).float()-soft.detach()+soft
   else:weights=soft
  return self.head((z*weights).sum(-1))

@torch.no_grad()
def evaluate(m,x,y,routes=None):
 modes=['primary']+(['mean','magnitude_soft'] if m.variant=='baseline' else [])
 correct={k:torch.zeros(5) for k in modes};oracle=torch.zeros(5);counts=torch.bincount(y//2,minlength=5)
 before={k:v.clone() for k,v in m.state_dict().items()}
 for bx,by in zip(x.split(256),y.split(256)):
  raw=m.responses(bx,False)
  for mode in modes:
   pred=m.prediction(bx,raw,None if mode=='primary' else mode).argmax(-1);correct[mode]+=torch.bincount((by//2)[pred==by],minlength=5)
  if routes is not None:
   hidden=raw.tanh().gather(-1,routes[by//2].long()[:,:,None]).squeeze(-1);pred=m.head(hidden).argmax(-1);oracle+=torch.bincount((by//2)[pred==by],minlength=5)
 assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
 out={k:dict(accuracy=float(v.sum()/len(y)),pairs=(v/counts).tolist()) for k,v in correct.items()}
 if routes is not None:out['oracle']=dict(accuracy=float(oracle.sum()/len(y)),pairs=(oracle/counts).tolist())
 return out

def fit(arm,seed,order,x,y,vx,vy):
 torch.manual_seed(seed);m=Model(arm,seed)
 groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith('selector.')],lr=.0005)]
 if hasattr(m,'selector'):groups.append(dict(params=m.selector.parameters(),lr=.0005))
 opt=torch.optim.Adam(groups);routes=torch.zeros(5,168,dtype=torch.uint8);stages=[];start=time.perf_counter()
 for stage,pair in enumerate(order):
  ix=torch.where(y//2==pair)[0]
  if arm=='selector_soft_slow' and stage:
   opt.param_groups[0]['lr']=.0001
  for epoch in range(2):
   ids_order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for ids in ids_order.split(128):
    bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True);raw=m.responses(bx,False);z=raw.tanh();gate=m.gates().detach().clone();hidden=(z*gate).sum(-1);trace=m.head(hidden)
    alt=trace[:,None,None,:]+(z-hidden[:,:,None])[:,:,:,None]*m.head.weight.T[None,:,None,:]
    target=by[:,None,None].expand(alt.shape[:-1]);err=F.cross_entropy(alt.reshape(-1,10),target.reshape(-1),reduction='none').reshape(alt.shape[:-1])
    loss=F.cross_entropy(trace,by)+.5*err.mean()
    if arm!='baseline':loss=loss+F.cross_entropy(m.prediction(bx,raw),by)
    if hasattr(m,'selector'):
     # Per-image, per-neuron relative substitution losses teach the selector.
     teacher=(-(err.detach()-err.detach().mean(-1,keepdim=True))/.2).softmax(-1)
     logprobs=m.selector(bx).reshape(-1,168,4).log_softmax(-1)
     loss=loss+.5*(-teacher*logprobs).sum(-1).mean()
    assert torch.isfinite(loss)
    loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(err.detach().mean(0))
  routes[pair]=m.gates().argmax(-1).to(torch.uint8);scores=evaluate(m,vx,vy,routes);stages.append(scores)
 elapsed=time.perf_counter()-start
 return m,routes,dict(arm=arm,seed=seed,order=order,stages=stages,parameters=sum(p.numel() for p in m.parameters()),model_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()),training_seconds=elapsed)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_combined_selector_20261105'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels'])
 (a.out/'protocol.json').write_text(json.dumps(dict(arms=ARMS,seeds=[101,102,103],orders=['forward','reverse'],neurons=168,branches=4,inputs_per_branch=196,epochs_per_task=2,selector='784 -> 32 tanh -> 168x4 scores',loss='trace CE + .5 counterfactual CE; nonbaseline adds image-route CE; selector arms add .5 soft-target cross entropy with relative error temperature .2',slow='after first task branch/head LR .0001; selector stays .0005',selection='validation averaged across six runs, then test only selected arm/mode and baseline',limits='No replay/prototypes/pretraining or task IDs at inference; known boundaries in training; selector has no explicit protection from forgetting; slow arm is a shared-weight slowdown, not importance-based protection'),indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes());results=[]
 for seed in [101,102,103]:
  for order in [list(range(5)),list(reversed(range(5)))]:
   for arm in ARMS:
    m,routes,r=fit(arm,seed,order,x,y,vx,vy);results.append(r);torch.save(dict(state=m.state_dict(),routes=routes),a.out/f'{arm}_{seed}_{order[0]}.pt');(a.out/'results.json').write_text(json.dumps(results,indent=2));print(arm,seed,order[0],{k:round(v['accuracy']*100,2) for k,v in r['stages'][-1].items()},flush=True)
 candidates=[]
 for arm in ARMS:
  for mode in (['primary','mean','magnitude_soft'] if arm=='baseline' else ['primary']):
   rs=[r for r in results if r['arm']==arm];candidates.append((statistics.mean(r['stages'][-1][mode]['accuracy'] for r in rs),arm,mode))
 best=max(candidates);selection=dict(validation_accuracy=best[0],arm=best[1],mode=best[2]);(a.out/'selection.json').write_text(json.dumps(selection,indent=2))
 tests=[]
 for arm in set(['baseline',best[1]]):
  for r in [r for r in results if r['arm']==arm]:
   m=Model(arm,r['seed']);saved=torch.load(a.out/f"{arm}_{r['seed']}_{r['order'][0]}.pt",weights_only=True);m.load_state_dict(saved['state']);scores=evaluate(m,tx,ty,saved['routes']);tests.append(dict(arm=arm,seed=r['seed'],order=r['order'],scores=scores))
 (a.out/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selection,flush=True)
if __name__=='__main__':main()
