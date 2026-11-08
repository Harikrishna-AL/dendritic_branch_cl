"""Matched training ablation accompanying frozen-network context-state test."""
import json,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model,evaluate

class StateModel(Model):
 def __init__(self,policy,seed):
  super().__init__('baseline',seed);self.policy=policy;self.register_buffer('context',torch.full((168,4),.25))
 def training_forward(self,x):
  raw=self.responses(x,False);z=raw.tanh()
  choices=raw.abs().argmax(-1) if self.policy=='no_state' else self.context.argmax(-1)[None].expand(len(x),-1)
  h=z.gather(-1,choices[:,:,None]).squeeze(-1);logits=self.head(h);alt=logits[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*self.head.weight.T[None,:,None,:]
  return logits,alt

def main():
 out=Path('results/mnist_short_term_context_20261105');torch.set_num_threads(2);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);rows=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   order=list(range(5)) if first==0 else list(reversed(range(5)))
   for policy in ['no_state','label_free']:
    torch.manual_seed(seed);m=StateModel(policy,seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);stages=[];start=time.perf_counter()
    for stage,pair in enumerate(order):
     ix=torch.where(y//2==pair)[0]
     for epoch in range(2):
      ordered=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
      for ids in ordered.split(128):
       opt.zero_grad(set_to_none=True);logits,alt=m.training_forward(x[ids]);target=y[ids,None,None].expand(alt.shape[:-1]);err=F.cross_entropy(alt.reshape(-1,10),target.reshape(-1),reduction='none').reshape(alt.shape[:-1]);loss=F.cross_entropy(logits,y[ids])+.5*err.mean()
       with torch.no_grad():
        pseudo=logits.argmax(-1);pe=F.cross_entropy(alt.detach().reshape(-1,10),pseudo[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1]);center=pe-pe.mean(-1,keepdim=True);pref=(-center/center.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).softmax(-1).mean(0)
       assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
       if policy=='label_free':m.context.mul_(.9).add_(pref,alpha=.1)
     stages.append(evaluate(m,tx,ty)['magnitude_soft'])
    val=evaluate(m,vx,vy)['magnitude_soft'];forget=statistics.mean(max(s['pairs'][pair] for s in stages[i:])-stages[-1]['pairs'][pair] for i,pair in enumerate(order[:-1]));rows.append(dict(seed=seed,first=first,policy=policy,stages=stages,validation=val,forgetting=forget,training_seconds=time.perf_counter()-start));(out/'training_ablation.json').write_text(json.dumps(rows,indent=2));print(seed,first,policy,round(stages[-1]['accuracy']*100,2),flush=True)
 # Existing supervised-trace checkpoints use identical initialization, batch order and CE+aux recipe.
 original=json.loads(Path('results/mnist_combined_selector_20261105/test_results.json').read_text());base=[r for r in original if r['arm']=='baseline'];lines=['# Matched training ablation','', 'Same sparse 168x4 MLP, fixed training recipe, three seeds x two orders. All reported final inference is stateless soft-magnitude combination, isolating learned weights. No tuning on these query metrics.','', '| Training selection | Final image-only test accuracy | Forgetting pp |','|---|---:|---:|']
 vals=[statistics.mean(r['scores']['magnitude_soft']['accuracy']*100 for r in base if r['seed']==s) for s in [101,102,103]];lines.append(f'| Supervised error trace (existing matched baseline) | {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}% | See original trajectory report |')
 for policy in ['no_state','label_free']:
  rr=[r for r in rows if r['policy']==policy];v=[statistics.mean(r['stages'][-1]['accuracy']*100 for r in rr if r['seed']==s) for s in [101,102,103]];lines.append(f'| {policy} | {statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}% | {statistics.mean(r["forgetting"] for r in rr)*100:.2f} |')
 lines+=['','No-state training selects each image’s greatest-magnitude branch per neuron, without history. Label-free training selects each neuron’s highest EMA pseudo-error preference: branch substitution CE against the model’s own current predicted class, standardized across four branches; EMA decay .9, updated only after each training minibatch. Labels still train the common CE+aux weights, but do not form that state. State persists across batches/tasks; no boundary resets. Batch-mean updates during training approximate stream context, unlike single-image updates in the frozen inference test. All alternatives retain the same auxiliary loss and optimizer.','', 'This is a fixed-recipe ablation, not proof that every unsupervised-state training rule must fail. Extra context buffer is 2,688 bytes; original error buffer is unused in these alternatives. State is not used during final stateless inference. No-state versus label-free gradient flow differs through their chosen branches; that is the intervention.']
 (out/'TRAINING_ABLATION.md').write_text('\n'.join(lines)+'\n');(out/'training_ablation_source.py').write_bytes(Path(__file__).read_bytes());print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
