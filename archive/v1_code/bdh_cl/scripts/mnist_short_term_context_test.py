"""Causal frozen-network short-term-state test; supervised feedback is explicit control."""
import json,time,argparse,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model

@torch.no_grad()
def features(m,x,y):
 result={k:[] for k in ['z','current','pseudo','supervised']}
 for bx,by in zip(x.split(256),y.split(256)):
  raw=m.responses(bx,False);z=raw.tanh();cur=raw.abs().softmax(-1);h=(z*cur).sum(-1);logits=m.head(h)
  alt=logits[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*m.head.weight.T[None,:,None,:]
  for name,target in [('pseudo',logits.argmax(-1)),('supervised',by)]:
   err=F.cross_entropy(alt.reshape(-1,10),target[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1])
   # Scale-free within-neuron preferences; labels affect control updates only.
   centered=err-err.mean(-1,keepdim=True);pref=(-centered/centered.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).softmax(-1)
   result[name].append(pref.numpy())
  result['z'].append(z.numpy());result['current'].append(cur.numpy())
 return {k:np.concatenate(v) for k,v in result.items()}

def stream_order(y,seed,kind):
 gen=np.random.default_rng(seed)
 if kind=='mixed':return gen.permutation(len(y))
 queues=[list(gen.permutation(np.flatnonzero(y//2==g))) for g in range(5)];parts=[]
 while any(queues):
  for q in queues:
   if q:parts.extend(q[:200]);del q[:200]
 return np.array(parts)

def run_stream(cache,y,order,w,b,policy,decay=.95,strength=.5,intervention='retain',seed=0):
 state=None;correct=[];groups=[];usage=np.zeros((5,168,4));counts=np.zeros(5);rng=np.random.default_rng(seed);updates=cache[{'magnitude':'current','pseudo':'pseudo','supervised':'supervised','none':'current'}[policy]]
 for ix in order:
  cur=cache['current'][ix]
  if intervention=='reset_each':state=None
  history=state
  if history is not None and intervention=='shuffle_branches':history=np.take_along_axis(history,np.argsort(rng.random(history.shape),axis=1),axis=1)
  if history is None or policy=='none':gate=cur
  else:gate=(1-strength)*cur+strength*history
  hidden=(cache['z'][ix]*gate).sum(-1);pred=int((w@hidden+b).argmax());group=int(y[ix]//2)
  correct.append(int(pred==y[ix]));groups.append(group);usage[group]+=gate;counts[group]+=1
  # Update only AFTER this image's prediction; only supervised uses its true label.
  observation=updates[ix];state=observation.copy() if state is None else decay*state+(1-decay)*observation
 correct=np.array(correct);groups=np.array(groups);pair=[float(correct[groups==g].mean()) for g in range(5)]
 starts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1];ends=np.r_[starts[1:],len(groups)];early=[];late=[];bins=[[] for _ in range(10)]
 for j,(start,end) in enumerate(zip(starts,ends)):
  if j==0 or end-start<100:continue
  early.extend(correct[start:start+20]);late.extend(correct[start+100:min(end,start+200)])
  for k in range(10):bins[k].extend(correct[start+20*k:min(end,start+20*(k+1))])
 means=usage/counts[:,None,None];overlap=statistics.mean(np.minimum(means[i],means[j]).sum(-1).mean() for i in range(5) for j in range(i+1,5))
 return dict(accuracy=float(correct.mean()),pair_accuracy=pair,early_after_switch=float(np.mean(early)) if early else None,late_after_switch=float(np.mean(late)) if late else None,switch_curve=[float(np.mean(v)) if v else None for v in bins],context_gate_overlap=float(overlap))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_short_term_context_20261105'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels'])
 proto=dict(scope='Frozen-checkpoint inference-state experiment, not training/weight-forgetting ablation',models='Three seeds x two training orders; sparse 168x4 MLP, 196 pixels/branch, original hard supervised-trace training',policies=dict(none='Current-image magnitude soft gate only',magnitude='EMA of previous images magnitude-soft branch weights',pseudo='EMA branch preferences from counterfactual CE against own prediction; no labels',supervised='Matched EMA branch preferences from true-label counterfactual CE, explicit postprediction-feedback control'),preferences='Counterfactual errors relative to stateless prediction; standardize within each neuron then softmax(-error). Fixed probes; not recomputed for state-conditioned prediction.',grid=dict(decay=[.9,.99],strength=[.5,.9]),selection='Choose each state policy by mean validation accuracy on persistent-context stream over six checkpoints; no test tuning',streams='All images once: persistent context rotates blocks of 200 within digit pair, or random mixed stream; no boundaries passed to algorithm',controls=['retain','reset_each','shuffle_branches'],causality='Predict current image before adding its observation. Only supervised feedback control receives true labels after prediction.',limits='Pretrained checkpoint architecture previously selected on same validation pool; exploratory secondary evaluation. Labels only construct benchmark streams, metrics, and explicit supervised control. No task-ID input, no prototypes, no query-label input to label-free policies. State resets only at start of independent streams except specified intervention.')
 (a.out/'protocol.json').write_text(json.dumps(proto,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes());val=[];cache_store={}
 for seed in [101,102,103]:
  for first in [0,4]:
   m=Model('baseline',seed);saved=torch.load(Path('results/mnist_combined_selector_20261105')/f'baseline_{seed}_{first}.pt',weights_only=True);m.load_state_dict(saved['state']);m.requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()};v=features(m,vx,vy);w=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy();order=stream_order(vy.numpy(),seed+first,'blocked')
   for policy in ['magnitude','pseudo','supervised']:
    for decay in [.9,.99]:
     for strength in [.5,.9]:
      score=run_stream(v,vy.numpy(),order,w,b,policy,decay,strength);val.append(dict(seed=seed,first=first,policy=policy,decay=decay,strength=strength,score=score))
   cache_store[(seed,first)]=(features(m,tx,ty),w,b)
   assert all(torch.equal(before[k],v) for k,v in m.state_dict().items());print('VALIDATED',seed,first,flush=True)
 (a.out/'validation.json').write_text(json.dumps(val,indent=2));selected={}
 for policy in ['magnitude','pseudo','supervised']:
  choices=[(statistics.mean(r['score']['accuracy'] for r in val if r['policy']==policy and r['decay']==d and r['strength']==s),d,s) for d in [.9,.99] for s in [.5,.9]];acc,d,s=max(choices);selected[policy]=dict(decay=d,strength=s,validation_accuracy=acc)
 (a.out/'selection.json').write_text(json.dumps(selected,indent=2));rows=[]
 for (seed,first),(cache,w,b) in cache_store.items():
  for kind in ['blocked','mixed']:
   order=stream_order(ty.numpy(),seed+first,kind)
   rows.append(dict(seed=seed,first=first,stream=kind,policy='none',intervention='retain',score=run_stream(cache,ty.numpy(),order,w,b,'none')))
   for policy,config in selected.items():
    for intervention in ['retain','reset_each','shuffle_branches']:
     result=run_stream(cache,ty.numpy(),order,w,b,policy,config['decay'],config['strength'],intervention,seed+first);rows.append(dict(seed=seed,first=first,stream=kind,policy=policy,intervention=intervention,score=result))
   (a.out/'results.json').write_text(json.dumps(rows,indent=2));print('TESTED',seed,first,kind,flush=True)
if __name__=='__main__':main()
