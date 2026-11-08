"""Train a masked decoder to rank useful gates and/or improve one-step gate inference.
Core branch/classifier gradients are preserved: new losses train decoder only.
"""
import json,time,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_masked_gate_refinement import DecoderModel,masked_loss
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
ARMS=['decoder_only','ranking','meta','ranking_meta']


def views(m,x,gen):
 with torch.no_grad():
  raw=m.responses(x,False);z=raw.tanh();mask=torch.rand((len(x),2,784),generator=gen)<.3
  masked=x[:,None,:].expand(-1,2,-1).masked_fill(mask,0)
  zm=m.responses(masked.reshape(-1,784),False).tanh().reshape(len(x),2,168,4)
  target=(x*.30810780+.13062755).clamp(0,1)
 return raw.detach(),z.detach(),zm.detach(),mask,target


def rec_error(m,zm,logits,target,mask,initial=None):
 gate=logits.softmax(-1);h=(zm*gate[:,None]).sum(-1);per=masked_loss(m.decoder(h),target[:,None],mask).mean(1)
 if initial is not None:
  kl=(gate*(gate.clamp_min(1e-8).log()-initial.log_softmax(-1))).sum(-1).mean(-1);return per+.01*kl
 return per


def gate_step(loss,s,create_graph):
 grad=torch.autograd.grad(loss.sum(),s,create_graph=create_graph)[0]
 # Same smooth RMS-normalized update at training and evaluation.
 rms=grad.square().mean((-1,-2),keepdim=True).add(1e-12).sqrt()
 return s-.1*grad/(rms+1e-6)


def decoder_objective(m,x,y,arm,gen):
 raw,z,zm,mask,target=views(m,x,gen);trace=m.gates().detach().clone()
 # Anchor: ordinary masked reconstruction under current trace route.
 h=(zm*trace[None,None]).sum(-1);rec=masked_loss(m.decoder(h),target[:,None],mask).mean();loss=rec;metrics={}
 if 'ranking' in arm:
  with torch.no_grad():
   route_gen=torch.Generator();route_gen.set_state(gen.get_state());batch=len(x);gates=trace[None,None].expand(batch,4,-1,-1).clone();gates[:,1]=raw.abs().softmax(-1)
   for j,rate in [(2,.25),(3,.5)]:
    replace=torch.rand((batch,168),generator=route_gen)<rate;choices=torch.randint(4,(batch,168),generator=route_gen);random=F.one_hot(choices,4).float();gates[:,j]=torch.where(replace[:,:,None],random,gates[:,j])
   hidden=(z[:,None]*gates).sum(-1);logits=m.head(hidden);ce=F.cross_entropy(logits.reshape(-1,10),y[:,None].expand(-1,4).reshape(-1),reduction='none').reshape(batch,4)
   best=ce.argmin(-1);worst=ce.argmax(-1);index=torch.arange(batch);valid=(logits[index,best].argmax(-1)==y)&((ce[index,worst]-ce[index,best])>.1)
   gp=gates[index,best];gn=gates[index,worst]
  hp=(zm*gp[:,None]).sum(-1);hn=(zm*gn[:,None]).sum(-1);rp=masked_loss(m.decoder(hp),target[:,None],mask).mean(1);rn=masked_loss(m.decoder(hn),target[:,None],mask).mean(1)
  rank=(F.relu(.02+rp-rn)*valid).sum()/valid.sum().clamp_min(1);loss=loss+.5*rank
  metrics['valid_fraction']=float(valid.float().mean());metrics['rank_accuracy']=float(((rp<rn)&valid).float().sum()/valid.sum().clamp_min(1))
 if 'meta' in arm:
  s=raw.abs().clone().requires_grad_();r0=rec_error(m,zm,s,target,mask);s1=gate_step(r0,s,True)
  hidden=(z*s1.softmax(-1)).sum(-1)
  # Frozen classifier in the outer loss; only decoder receives meta-gradients.
  logits=F.linear(hidden,m.head.weight.detach(),m.head.bias.detach());outer=F.cross_entropy(logits,y);loss=loss+.5*outer
  metrics['outer_ce']=float(outer.detach())
 return loss,metrics


def fit(arm,seed,first,x,y):
 torch.manual_seed(seed);m=DecoderModel(seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);gen=torch.Generator().manual_seed(seed+12000);step=0;seconds=0;stats=[]
 for stage,pair in enumerate(range(5) if first==0 else reversed(range(5))):
  ix=torch.where(y//2==pair)[0];start=time.perf_counter()
  for epoch in range(2):
   order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for ids in order.split(128):
    bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True);anchor,feedback,_=objective(m,bx,by,'hybrid5',step);dloss,metrics=decoder_objective(m,bx,by,arm,gen);loss=anchor+dloss
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for n,p in m.named_parameters() if not n.startswith('decoder.')],5);torch.nn.utils.clip_grad_norm_(m.decoder.parameters(),5);opt.step();m.observe(feedback);step+=1
    if step%50==0:stats.append(dict(stage=stage,step=step,**metrics))
  seconds+=time.perf_counter()-start
 reference=torch.load(Path('results/mnist_gradient_branch_scoring_20261106')/f'hybrid5_{seed}_{first}.pt',weights_only=True)['state']
 differences={k:float((v-reference[k]).abs().max()) for k,v in m.state_dict().items() if not k.startswith('decoder.')}
 assert max(differences.values())==0.,differences
 return m,seconds,stats


def refine(m,x,steps=5):
 raw,z,zm,mask,target=views(m,x,torch.Generator().manual_seed(24680));initial=raw.abs();s=initial.clone().requires_grad_();snap={}
 for step in range(steps+1):
  r=rec_error(m,zm,s,target,mask)
  if step in [0,1,3,5]:
   with torch.no_grad():snap[step]=dict(prediction=m.head((z*s.softmax(-1)).sum(-1)).argmax(-1).clone(),reconstruction=r.detach().clone())
  if step==steps:break
  objective=rec_error(m,zm,s,target,mask,initial);s=gate_step(objective,s,False).detach().requires_grad_()
 return snap


def evaluate(m,x,y):
 m.requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()};parts={s:dict(correct=[],rec=[]) for s in [0,1,3,5]};start=time.perf_counter()
 for bx,by in zip(x.split(128),y.split(128)):
  result=refine(m,bx)
  for step,v in result.items():parts[step]['correct'].append(v['prediction']==by);parts[step]['rec'].append(v['reconstruction'])
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());base=torch.cat(parts[0]['correct']);scores={}
 for step,v in parts.items():
  correct=torch.cat(v['correct']);scores[step]=dict(accuracy=float(correct.float().mean()),reconstruction=float(torch.cat(v['rec']).mean()),corrected=int((correct&~base).sum()),harmed=int((~correct&base).sum()))
 return scores,time.perf_counter()-start


def main():
 torch.set_num_threads(2);out=Path('results/mnist_reconstruction_alignment_20261106');out.mkdir(exist_ok=True);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);records=[]
 protocol=dict(arms=ARMS,seeds=[101,102,103],orders=['forward','reverse'],scope='New objectives train decoder only, preserving proven hybrid5 branch/head weights bitwise; labels teach decoder during training only',training='Two epochs/task, Adam .0005, batch128, independent clip5 for decoder and core',decoder='168->32 tanh->784 sigmoid, 31280 params',views='Two masked views per image,30% pixels hidden, target-weighted masked MSE as earlier, no skip links',ranking='4 candidate gates: trace, magnitude-soft,25%/50% trace perturbations. Pick min/max true ten-class CE; rank only if best correct and CE gap>.1. loss ordinary trace reconstruction + .5 relu(.02+Rgood-Rbad). No gradient through candidate selection or classifier features.',meta='Ordinary trace reconstruction + .5 labeled CE AFTER one differentiable reconstruction-gradient gate update; fixed detached image features and classifier. Update s1=s0-.1 grad/(per-image RMSgrad+1e-6), second-order graph retained into decoder.',combined='Anchor reconstruction + ranking + post-update classification loss',inference='Weights frozen; per-image gates only, fixed RMS-normalized reconstruction steps0/1/3/5; KL.01 to initial gate on later steps; no labels/task IDs/replay/history',selection='Overall winner and best positive-step candidate selected by aggregate validation only; test best positive-step arm and ordinary decoder control at all prespecified steps, report zero-step baseline',limits='Decoder-only isolation avoids harming backbone but may limit joint solutions; decoder is still trained sequentially and can forget; margin/rate fixed not tuned on test; full meta training through one step only')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in [101,102,103]:
  for first in [0,4]:
   for arm in ARMS:
    m,seconds,stats=fit(arm,seed,first,x,y);val,elapsed=evaluate(m,vx,vy);records.append(dict(seed=seed,first=first,arm=arm,validation=val,training_seconds=seconds,evaluation_seconds=elapsed,training_metrics=stats,core_bitwise_matches_hybrid5=True));torch.save(m.state_dict(),out/f'{arm}_{seed}_{first}.pt');(out/'results.json').write_text(json.dumps(records,indent=2));print(seed,first,arm,{k:round(v['accuracy']*100,2) for k,v in val.items()},flush=True)
 choices=[(statistics.mean(r['validation'][step]['accuracy'] for r in records if r['arm']==arm),arm,step) for arm in ARMS for step in [0,1,3,5]];winner=max(choices);positive=max(c for c in choices if c[2]>0);selection=dict(overall=dict(arm=winner[1],steps=winner[2],validation_accuracy=winner[0]),positive=dict(arm=positive[1],steps=positive[2],validation_accuracy=positive[0]));(out/'selection.json').write_text(json.dumps(selection,indent=2));tests=[]
 for arm in sorted(set(['decoder_only',positive[1]])):
  for r in [r for r in records if r['arm']==arm]:
   m=DecoderModel(r['seed']);m.load_state_dict(torch.load(out/f"{arm}_{r['seed']}_{r['first']}.pt",weights_only=True));scores,seconds=evaluate(m,tx,ty);tests.append(dict(seed=r['seed'],first=r['first'],arm=arm,scores=scores,seconds=seconds))
 (out/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selection,flush=True)
if __name__=='__main__':main()
