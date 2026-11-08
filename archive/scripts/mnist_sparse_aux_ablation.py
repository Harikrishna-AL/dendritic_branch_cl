"""Matched auxiliary-loss ablation preserving identical error-trace feedback."""
import argparse,json,time,statistics,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model,evaluate
from bdh_cl.scripts.mnist_short_term_context_test import features,run_stream,stream_order


def objective(m,x,y,coefficient):
 raw=m.responses(x,False);z=raw.tanh();h=(z*m.gates().detach().clone()).sum(-1);logits=m.head(h)
 if coefficient:
  alt=logits[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*m.head.weight.T[None,:,None,:]
  errors=F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1])
  loss=F.cross_entropy(logits,y)+coefficient*errors.mean();feedback=errors.detach().mean(0)
 else:
  # Selection feedback still evaluates all single-neuron substitutions,
  # but it has no backward graph and does not change the CE-only objective.
  with torch.no_grad():
   alt=logits.detach()[:,None,None,:]+(z.detach()-h.detach()[:,:,None])[:,:,:,None]*m.head.weight.detach().T[None,:,None,:]
   errors=F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1]);feedback=errors.mean(0)
  loss=F.cross_entropy(logits,y)
 return loss,feedback


def run(coefficient,seed,first,data):
 x,y,vx,vy,tx,ty=data;torch.manual_seed(seed);m=Model('baseline',seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);order=list(range(5)) if first==0 else list(reversed(range(5)));routes=torch.zeros(5,168,dtype=torch.uint8);history=[];seconds=0.
 for stage,pair in enumerate(order):
  ids=torch.where(y//2==pair)[0];start=time.perf_counter()
  for epoch in range(2):
   ordered=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for ix in ordered.split(128):
    opt.zero_grad(set_to_none=True);loss,feedback=objective(m,x[ix],y[ix],coefficient);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback)
  seconds+=time.perf_counter()-start;routes[pair]=m.error.argmin(-1).to(torch.uint8);history.append(evaluate(m,vx,vy,routes))
 test=evaluate(m,tx,ty,routes);forget=statistics.mean(max(s['magnitude_soft']['pairs'][pair] for s in history[i:])-history[-1]['magnitude_soft']['pairs'][pair] for i,pair in enumerate(order[:-1]));acquisition=statistics.mean(history[i]['magnitude_soft']['pairs'][pair] for i,pair in enumerate(order))
 before={k:v.clone() for k,v in m.state_dict().items()};m.requires_grad_(False);cache=features(m,tx,ty);w=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy();streams={}
 for kind in ['blocked','mixed']:
  streams[kind]=run_stream(cache,ty.numpy(),stream_order(ty.numpy(),seed+first,kind),w,b,'pseudo',decay=.9,strength=.9)
 assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
 return m,routes,dict(aux_coefficient=coefficient,seed=seed,first=first,order=order,validation_history=history,test=test,validation_forgetting=forget,validation_acquisition=acquisition,training_seconds=seconds,streams=streams,parameters=sum(p.numel() for p in m.parameters()),model_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()))


def report(out,records):
 lines=['# Auxiliary-loss ablation on the current sparse MLP','', 'Matched three seeds × two task orders × two loss coefficients. Four branches per neuron, 168 neurons, 196 random pixel connections per branch, no per-image LayerNorm, continuous Adam .0005, batch128, clip5, two epochs/task. Same initialization, data split, minibatch order, EMA .95 supervised substitution-error selection and evaluation in both arms. Only auxiliary-loss coefficient changes: .5 versus 0.','', '| Auxiliary coefficient | Image-only soft test | Image-only hard test | Oracle test | Validation forgetting pp | Train s |','|---|---:|---:|---:|---:|---:|']
 summary=[]
 for coefficient in [.5,0.]:
  rows=[r for r in records if r['aux_coefficient']==coefficient]
  if len(rows)!=6:continue
  metrics={}
  for name,path in [('soft',('test','magnitude_soft','accuracy')),('hard',('test','primary','accuracy')),('oracle',('test','oracle','accuracy')),('memory_blocked',('streams','blocked','accuracy')),('memory_mixed',('streams','mixed','accuracy'))]:
   def value(r):
    for k in path:r=r[k]
    return r
   vals=[statistics.mean(value(r)*100 for r in rows if r['seed']==s) for s in [101,102,103]];metrics[name]=dict(mean=statistics.mean(vals),seed_sd=statistics.stdev(vals))
  seconds=statistics.mean(r['training_seconds'] for r in rows);forget=statistics.mean(r['validation_forgetting'] for r in rows)*100;summary.append(dict(aux_coefficient=coefficient,metrics=metrics,training_seconds=seconds,validation_forgetting_pp=forget));fmt=lambda k:f"{metrics[k]['mean']:.2f} ± {metrics[k]['seed_sd']:.2f}%"
  lines.append(f'| {coefficient} | {fmt("soft")} | {fmt("hard")} | {fmt("oracle")} | {forget:.2f} | {seconds:.2f} |')
 if len(summary)==2:
  diffs=[];paired=[]
  for r in [r for r in records if r['aux_coefficient']==0]:
   b=next(b for b in records if b['aux_coefficient']==.5 and b['seed']==r['seed'] and b['first']==r['first']);paired.append(dict(seed=r['seed'],first=r['first'],difference_pp=(r['test']['magnitude_soft']['accuracy']-b['test']['magnitude_soft']['accuracy'])*100))
  diffs=[statistics.mean(r['difference_pp'] for r in paired if r['seed']==s) for s in [101,102,103]];mean=statistics.mean(diffs);half=4.30265273*statistics.stdev(diffs)/math.sqrt(3)
  lines+=['',f'CE-only minus auxiliary: **{mean:+.2f} pp** image-only soft accuracy; descriptive seed-level 95% t interval [{mean-half:+.2f}, {mean+half:+.2f}] pp. Positive in {sum(r["difference_pp"]>0 for r in paired)}/6 seed/order pairs.','', '| Auxiliary coefficient | Label-free state: persistent contexts | Label-free state: mixed stream |','|---|---:|---:|']
  for r in summary:lines.append(f"| {r['aux_coefficient']} | {r['metrics']['memory_blocked']['mean']:.2f} ± {r['metrics']['memory_blocked']['seed_sd']:.2f}% | {r['metrics']['memory_mixed']['mean']:.2f} ± {r['metrics']['memory_mixed']['seed_sd']:.2f}% |")
  (out/'paired.json').write_text(json.dumps(paired,indent=2))
 lines+=['','Even with coefficient zero, substitution losses still update supervised branch-error traces; only their gradient contribution is removed. CE-only uses no-grad substitution feedback, so trace-evaluation compute remains necessary. No replay, pretraining, output masking, task ID or archived-route lookup in image-only inference. Oracle uses true task identity and saved branch indices.','', 'Short-term state uses the previously selected pseudo-error preferences, decay .9, strength .9; no retuning for CE-only. Persistent streams rotate 200-image task blocks; mixed streams shuffle the same images. Labels never enter this inference state. All network tensors are checked unchanged by evaluation.','', 'Both models have 134,074 parameters and 1.593 MB persistent tensor state; inference cost/size is unchanged by removing a training loss. Runtime measures training loops only, including no-grad feedback, excluding evaluation; local pilot timing, not peak memory or hardware-independent FLOPs. Accuracy ± SD averages the two orders within each of three seeds. Fixed shared recipe, not independently tuned methods; previously inspected MNIST test set, so these are exploratory ablations. No noninferiority margin was specified.']
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_sparse_aux_ablation_20261106'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 data=(*ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]),*ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]),*ds(raw['test_images'],raw['test_labels']))
 (a.out/'source.py').write_bytes(Path(__file__).read_bytes());(a.out/'protocol.json').write_text(json.dumps(dict(aux_coefficient=[.5,0],seeds=[101,102,103],orders=['forward','reverse'],epochs=2,learning_rate=.0005,batch_size=128,trace_decay=.95,selection='Prespecified matched ablation, no test-based hyperparameter selection',state_evaluation='Previously chosen decay .9 and strength .9, no retuning',train_examples=len(data[1]),validation_examples=len(data[3]),test_examples=len(data[5])),indent=2));records=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   # Alternate run order to reduce systematic timing-order bias.
   for coefficient in ([.5,0.] if (seed+first)%2 else [0.,.5]):
    m,routes,r=run(coefficient,seed,first,data);records.append(r);torch.save(dict(state=m.state_dict(),routes=routes),a.out/f'aux{coefficient}_{seed}_{first}.pt');(a.out/'results.json').write_text(json.dumps(records,indent=2));report(a.out,records);print(seed,first,coefficient,{k:round(v['accuracy']*100,2) for k,v in r['test'].items()},'memory',round(r['streams']['blocked']['accuracy']*100,2),flush=True)
 print((a.out/'REPORT.md').read_text(),flush=True)
if __name__=='__main__':main()
