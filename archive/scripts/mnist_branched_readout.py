"""Matched shared, branch-specific learned, and branch-specific fixed readouts."""
import json,time,statistics
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model

class Branched(Model):
 def __init__(self,arm,seed):
  super().__init__('baseline',seed);self.readout_arm=arm
  if arm!='shared':
   # Replicate original readout: identical function at initialization for any gates.
   self.head=nn.Linear(672,10)
   torch.manual_seed(seed);original=Model('baseline',seed)
   with torch.no_grad():self.head.weight.copy_(original.head.weight[:,:,None].expand(-1,-1,4).reshape(10,672));self.head.bias.copy_(original.head.bias)
   if arm=='branched_fixed':self.head.requires_grad_(False)
 def vectors(self):return self.head.weight[:,:,None].expand(-1,-1,4) if self.readout_arm=='shared' else self.head.weight.reshape(10,168,4)
 def components(self,x):
  raw=self.responses(x,False);z=raw.tanh();gate=self.gates().detach().clone();v=self.vectors();contrib=z[:,:,:,None]*v.permute(1,2,0)[None];selected=(contrib*gate[None,:,:,None]).sum(2);logits=selected.sum(1)+self.head.bias
  return raw,z,v,selected,logits

def objective(m,x,y,step):
 raw,z,v,selected,logits=m.components(x);ce=F.cross_entropy(logits,y)
 if step%5==0:
  alt=logits[:,None,None,:]+z[:,:,:,None]*v.permute(1,2,0)[None]-selected[:,:,None,:]
  err=F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1]);return ce+.5*err.mean(),err.detach().mean(0)
 with torch.no_grad():residual=logits.softmax(-1)-F.one_hot(y,10).float()
 g=torch.einsum('sc,cik->sik',residual,v)
 selected_g=(residual[:,:,None]*selected.permute(0,2,1)).sum(1)
 delta=g*z-selected_g[:,:,None];loss=1.5*ce+.5*delta.mean()
 with torch.no_grad():feedback=(F.cross_entropy(logits,y,reduction='none')[:,None,None]+delta.detach()).mean(0)
 return loss,feedback

@torch.no_grad()
def evaluate(m,x,y,routes):
 hits={k:torch.zeros(5) for k in ['soft','hard','oracle']};counts=torch.bincount(y//2,minlength=5)
 for bx,by in zip(x.split(256),y.split(256)):
  raw=m.responses(bx,False);z=raw.tanh();v=m.vectors()
  for mode in hits:
   if mode=='soft':g=raw.abs().softmax(-1)
   else:g=F.one_hot(routes[by//2].long() if mode=='oracle' else raw.abs().argmax(-1),4).float()
   logits=torch.einsum('sik,cik->sc',z*g,v)+m.head.bias;pred=logits.argmax(-1);hits[mode]+=torch.bincount((by//2)[pred==by],minlength=5)
 return {k:dict(accuracy=float(h.sum()/len(y)),pairs=(h/counts).tolist()) for k,h in hits.items()}

def main():
 torch.set_num_threads(2);out=Path('results/mnist_branched_readout_20261106');out.mkdir(exist_ok=True);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);rows=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   for arm in ['shared','branched','branched_fixed']:
    torch.manual_seed(seed);m=Branched(arm,seed);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0005);routes=torch.zeros(5,168,dtype=torch.uint8);step=0;seconds=0
    for stage,pair in enumerate(range(5) if first==0 else reversed(range(5))):
     ix=torch.where(y//2==pair)[0];tick=time.perf_counter()
     for epoch in range(2):
      ordered=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
      for ids in ordered.split(128):
       opt.zero_grad(set_to_none=True);loss,feedback=objective(m,x[ids],y[ids],step);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback);step+=1
     seconds+=time.perf_counter()-tick;routes[pair]=m.error.argmin(-1).to(torch.uint8)
    row=dict(arm=arm,seed=seed,first=first,validation=evaluate(m,vx,vy,routes),test=evaluate(m,tx,ty,routes),training_seconds=seconds,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad));rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(arm,seed,first,row['test']['soft']['accuracy'],flush=True)
 lines=['# Branch-specific readout ablation','','Three seeds × two orders. Sparse168x4 MLP,196 pixels/branch, Adam .0005,batch128,clip5,two epochs/task. Hybrid5 exact loss every fifth batch; first-order surrogate otherwise. Branch-specific substitution and Taylor scores use the full branch-specific class contribution. All ten classes compete; no task ID, replay or route archive in image-only inference. Oracle alone uses true task identity. All prespecified arms reported, no test-based selection.','','| Readout | Validation image-only | Test image-only | Oracle test | Parameters | Train seconds |','|---|---:|---:|---:|---:|---:|']
 for arm in ['shared','branched','branched_fixed']:
  rr=[r for r in rows if r['arm']==arm];v=[statistics.mean(r['test']['soft']['accuracy']*100 for r in rr if r['seed']==s) for s in [101,102,103]];lines.append(f'| {arm} | {statistics.mean(r["validation"]["soft"]["accuracy"] for r in rr)*100:.2f}% | {statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}% | {statistics.mean(r["test"]["oracle"]["accuracy"] for r in rr)*100:.2f}% | {rr[0]["parameters"]} | {statistics.mean(r["training_seconds"] for r in rr):.2f} |')
 lines+=['','Branch-specific readout initializes four branch class vectors as copies of the original shared vector: identical prediction function before training. Fixed variant freezes these replicated initial random vectors and bias; it does not use an orthogonal/ETF classifier. Fixed versus learned therefore tests this initialization only. Shared control is reimplemented with contribution tensors for mathematical matching, so timings are not optimized comparisons against the previous implementation. No new layers/nonlinearities, gradient clipping and optimizer unchanged. Larger readout adds5040 weights; parameter count is not exactly matched. Three seed-level means define SD. Fixed recipe, exploratory reused MNIST benchmark.']
 (out/'REPORT.md').write_text('\n'.join(lines));(out/'source.py').write_bytes(Path(__file__).read_bytes());print('\n'.join(lines))
if __name__=='__main__':main()
