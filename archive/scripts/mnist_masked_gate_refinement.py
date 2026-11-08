"""Masked decoder supervision and per-image gate refinement, no inference labels."""
import json,time,statistics
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
ARMS=['decoder_only','joint_0.1','joint_1.0']

class DecoderModel(Model):
 def __init__(self,seed):
  super().__init__('baseline',seed)
  with torch.random.fork_rng():
   torch.manual_seed(seed+9000);self.decoder=nn.Sequential(nn.Linear(168,32),nn.Tanh(),nn.Linear(32,784),nn.Sigmoid())

def masked_loss(pred,target,mask):
 weights=mask*(1+4*target)
 return ((pred-target).square()*weights).sum(-1)/weights.sum(-1).clamp_min(1.)


def fit(arm,seed,first,x,y):
 torch.manual_seed(seed);m=DecoderModel(seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);step=0;seconds=0;gen=torch.Generator().manual_seed(seed+8000)
 for stage,pair in enumerate(range(5) if first==0 else reversed(range(5))):
  ix=torch.where(y//2==pair)[0];start=time.perf_counter()
  for epoch in range(2):
   order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for ids in order.split(128):
    bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True);loss,feedback,_=objective(m,bx,by,'hybrid5',step)
    mask=torch.rand(bx.shape,generator=gen)<.3;masked=bx.masked_fill(mask,0);z=m.responses(masked,False).tanh();h=(z*m.gates().detach().clone()).sum(-1)
    if arm=='decoder_only':h=h.detach()
    recon=masked_loss(m.decoder(h),(bx*.30810780+.13062755).clamp(0,1),mask).mean();coefficient=1. if arm=='decoder_only' else float(arm.split('_')[1]);loss=loss+coefficient*recon
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for n,p in m.named_parameters() if not n.startswith('decoder.')],5);torch.nn.utils.clip_grad_norm_(m.decoder.parameters(),5);opt.step();m.observe(feedback);step+=1
  seconds+=time.perf_counter()-start
 return m,seconds


def refine(m,x,steps=10):
 # No labels accepted. Each image has independent temporary gates.
 with torch.no_grad():
  raw=m.responses(x,False);fullz=raw.tanh();initial=raw.abs();target=(x*.30810780+.13062755).clamp(0,1);gen=torch.Generator().manual_seed(24680)
  masks=torch.rand((len(x),4,784),generator=gen)<.3;masked=x[:,None,:].expand(-1,4,-1).masked_fill(masks,0);z=m.responses(masked.reshape(-1,784),False).tanh().reshape(len(x),4,168,4)
 gate_logits=initial.detach().clone().requires_grad_();opt=torch.optim.Adam([gate_logits],lr=.1);snapshots={}
 for step in range(steps+1):
  gate=gate_logits.softmax(-1);hidden=(z*gate[:,None]).sum(-1);recon=m.decoder(hidden);per=masked_loss(recon,target[:,None],masks).mean(1)
  if step in [0,1,5,10]:
   with torch.no_grad():snapshots[step]=dict(prediction=m.head((fullz*gate).sum(-1)).argmax(-1).clone(),reconstruction=per.detach().clone())
  if step==steps:break
  kl=(gate*(gate.clamp_min(1e-8).log()-initial.log_softmax(-1))).sum(-1).mean(-1)
  # Sum independent per-image objectives to avoid batch-size-dependent gradients.
  opt.zero_grad();(per+.01*kl).sum().backward();opt.step()
 return snapshots


def evaluate(m,x,y):
 m.requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()};outputs={s:dict(correct=[],reconstruction=[]) for s in [0,1,5,10]};start=time.perf_counter()
 for bx,by in zip(x.split(128),y.split(128)):
  result=refine(m,bx)
  for step,v in result.items():outputs[step]['correct'].append(v['prediction']==by);outputs[step]['reconstruction'].append(v['reconstruction'])
 assert all(torch.equal(before[k],v) for k,v in m.state_dict().items());base=torch.cat(outputs[0]['correct']);scores={}
 for step,r in outputs.items():
  correct=torch.cat(r['correct']);scores[step]=dict(accuracy=float(correct.float().mean()),reconstruction_mse=float(torch.cat(r['reconstruction']).mean()),corrected=int((correct&~base).sum()),harmed=int((~correct&base).sum()))
 return scores,time.perf_counter()-start


def main():
 torch.set_num_threads(2);out=Path('results/mnist_masked_gate_refinement_20261106');out.mkdir(exist_ok=True);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);rows=[]
 protocol=dict(arms=ARMS,seeds=[101,102,103],orders=['forward','reverse'],training='Original sparse168x4 MLP, hybrid5 CE+aux, Adam .0005, batch128, clip5, two epochs/task, sequential data only',decoder='168 ->32 tanh ->784 sigmoid, 31280 trainable params; decoder_only detaches branch features; joint arms propagate masked-reconstruction loss into branch encoder',reconstruction='30% random masked pixels replaced with standardized mean0; loss only masked pixels, foreground weighting1+4*target raw intensity',adaptation='Per-image gate logits only, all weights frozen; four masked views, Adam .1, KL .01 to initial magnitude gate, steps0/1/5/10; no labels, support examples, task grouping or state across images',selection='Mean validation across six runs selects arm and steps; test only selected decoder arm at baseline0 and selected steps. All candidates fixed before queries.',limits='Decoder is trained continually and can forget. Reconstruction error need not track classification. Fixed masked-view draws per batch can correlate examples; no cross-image adaptation. Decoder adds training and inference compute.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in [101,102,103]:
  for first in [0,4]:
   for arm in ARMS:
    m,seconds=fit(arm,seed,first,x,y);val,elapsed=evaluate(m,vx,vy);rows.append(dict(seed=seed,first=first,arm=arm,validation=val,training_seconds=seconds,validation_seconds=elapsed,parameters=sum(p.numel() for p in m.parameters())));torch.save(m.state_dict(),out/f'{arm}_{seed}_{first}.pt');(out/'results.json').write_text(json.dumps(rows,indent=2));print(seed,first,arm,{k:round(v['accuracy']*100,2) for k,v in val.items()},flush=True)
 choices=[(statistics.mean(r['validation'][step]['accuracy'] for r in rows if r['arm']==arm),arm,step) for arm in ARMS for step in [0,1,5,10]];best=max(choices);selection=dict(arm=best[1],steps=best[2],validation_accuracy=best[0]);(out/'selection.json').write_text(json.dumps(selection,indent=2));tests=[]
 for r in [r for r in rows if r['arm']==best[1]]:
  m=DecoderModel(r['seed']);m.load_state_dict(torch.load(out/f"{r['arm']}_{r['seed']}_{r['first']}.pt",weights_only=True));score,seconds=evaluate(m,tx,ty);tests.append(dict(seed=r['seed'],first=r['first'],scores=score,seconds=seconds))
 (out/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selection,flush=True)
if __name__=='__main__':main()
