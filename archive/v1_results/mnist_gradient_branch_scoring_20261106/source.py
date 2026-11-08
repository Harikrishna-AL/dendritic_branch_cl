"""Taylor branch scoring / auxiliary surrogate and periodic exact corrections."""
import json,time,statistics,math,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model,evaluate
from bdh_cl.scripts.mnist_short_term_context_test import features,run_stream,stream_order
ARMS=['exact','linear','quadratic','hybrid5','hybrid10']

def terms(m,x,y):
 z=m.responses(x,False).tanh();h=(z*m.gates().detach().clone()).sum(-1);logits=m.head(h);ce=F.cross_entropy(logits,y);delta=z-h[:,:,None]
 # Per-example CE derivative, not mean-batch derivative (no accidental 1/B).
 with torch.no_grad():p=logits.softmax(-1);residual=p-F.one_hot(y,10).to(p.dtype)
 g=F.linear(residual,m.head.weight.T) # B x neurons; differentiable in head W for surrogate.
 linear=(g[:,:,None]*delta).mean()
 curvature=F.linear(p,m.head.weight.T.square())-F.linear(p,m.head.weight.T).square()
 quadratic=.5*(curvature[:,:,None]*delta.square()).mean()
 with torch.no_grad():
  base=F.cross_entropy(logits.detach(),y,reduction='none')[:,None,None]
  approx_linear=base+g.detach()[:,:,None]*delta.detach()
  approx_quadratic=approx_linear+.5*curvature.detach()[:,:,None]*delta.detach().square()
 return z,h,logits,ce,delta,linear,quadratic,approx_linear,approx_quadratic

def exact_errors(m,y,z,h,logits):
 alt=logits[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*m.head.weight.T[None,:,None,:]
 return F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1])

def objective(m,x,y,arm,step):
 z,h,logits,ce,delta,lin,quad,approx1,approx2=terms(m,x,y)
 exact=arm=='exact' or (arm.startswith('hybrid') and step%int(arm[6:])==0)
 if exact:
  err=exact_errors(m,y,z,h,logits);loss=ce+.5*err.mean();feedback=err.detach().mean(0)
 else:
  loss=1.5*ce+.5*(lin+(quad if arm=='quadratic' else 0));feedback=(approx2 if arm=='quadratic' else approx1).mean(0)
 return loss,feedback,exact

@torch.no_grad()
def approximation_quality(m,x,y,stage,step):
 z,h,logits,_,_,_,_,a1,a2=terms(m,x,y);exact=exact_errors(m,y,z,h,logits);base=F.cross_entropy(logits,y,reduction='none')[:,None,None];rows=[]
 for name,estimate in [('linear',a1),('quadratic',a2)]:
  e=exact.mean(0);a=estimate.mean(0);chosen=a.argmin(-1);best=e.argmin(-1);regret=e.gather(-1,chosen[:,None]).squeeze(-1)-e.min(-1).values;valid=(exact-base).abs()>1e-6
  rows.append(dict(stage=stage,step=step,approximation=name,batch_best_branch_agreement=float((chosen==best).float().mean()),sample_best_branch_agreement=float((estimate.argmin(-1)==exact.argmin(-1)).float().mean()),mean_exact_regret=float(regret.mean()),mean_absolute_loss_error=float((estimate-exact).abs().mean()),sign_agreement=float((((estimate-base)>0)==((exact-base)>0))[valid].float().mean())))
 return rows

def fit(arm,seed,first,data):
 x,y,vx,vy=data;torch.manual_seed(seed);m=Model('baseline',seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);order=list(range(5)) if first==0 else list(reversed(range(5)));routes=torch.zeros(5,168,dtype=torch.uint8);stages=[];quality=[];seconds=0.;step=exact_count=0
 for stage,pair in enumerate(order):
  ix=torch.where(y//2==pair)[0]
  for epoch in range(2):
   ordered=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for ids in ordered.split(128):
    # Quality diagnostics only on exact baseline, outside timed training step.
    if arm=='exact' and step%100==0:quality+=approximation_quality(m,x[ids],y[ids],stage,step)
    tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss,feedback,is_exact=objective(m,x[ids],y[ids],arm,step);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback);seconds+=time.perf_counter()-tick;step+=1;exact_count+=int(is_exact)
  routes[pair]=m.error.argmin(-1).to(torch.uint8);stages.append(evaluate(m,vx,vy,routes))
 return m,routes,dict(arm=arm,seed=seed,first=first,order=order,validation_history=stages,training_seconds=seconds,steps=step,exact_steps=exact_count,quality=quality,parameters=sum(p.numel() for p in m.parameters()),model_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_gradient_branch_scoring_20261106'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);records=[]
 protocol=dict(arms=ARMS,seeds=[101,102,103],orders=['forward','reverse'],epochs=2,batch_size=128,learning_rate=.0005,trace_decay=.95,architecture='Sparse original 168x4 MLP, 196 inputs per branch, 134074 parameters',gradient='Per-example CE derivative g=(softmax(logits)-onehot(y))@W; closed-form equivalent of local hidden gradient, uses current linear ten-class head',surrogate='1.5*CE + .5 mean(g*delta); residual probabilities stopped, W and delta differentiated. Quadratic adds .25 mean(curvature*delta^2), with curvature Var_p(W_i). This includes base CE and head-weight gradients unlike a stopped-g-only branch surrogate.',feedback='Estimated absolute substitution CE = current per-example CE + g*delta (+ .5 curvature*delta² for quadratic); EMA .95 unchanged',hybrids='Exact objective+feedback every 5/10 global training batches, first batch exact; linear surrogate+feedback otherwise; no inverse-frequency loss scaling',selection='Pick approximate arm by aggregate final validation magnitude-soft accuracy; official tests only exact baseline and selected approximate arm',timing='Training-step wall time including branch feedback, excludes diagnostics and evaluation; CPU two threads. Exact baseline currently computes unused Taylor terms as well; separate clean benchmark will address overhead.',limits='Closed-form gradients/curvature for final linear classifier; deeper layers require local backprop gradients and curvature approximations. All branch responses still computed. Fixed training recipe; no replay, pretraining, task ID or saved routes in image-only inference.')
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in [101,102,103]:
  for first in [0,4]:
   arms=ARMS if (seed+first)%2 else list(reversed(ARMS))
   for arm in arms:
    m,routes,r=fit(arm,seed,first,(x,y,vx,vy));records.append(r);torch.save(dict(state=m.state_dict(),routes=routes),a.out/f'{arm}_{seed}_{first}.pt');(a.out/'results.json').write_text(json.dumps(records,indent=2));print(arm,seed,first,round(r['validation_history'][-1]['magnitude_soft']['accuracy']*100,2),round(r['training_seconds'],3),flush=True)
 selected=max(ARMS[1:],key=lambda arm:statistics.mean(r['validation_history'][-1]['magnitude_soft']['accuracy'] for r in records if r['arm']==arm));(a.out/'selection.json').write_text(json.dumps(dict(arm=selected,metric='magnitude_soft'),indent=2));tests=[]
 for arm in ['exact',selected]:
  for r in [r for r in records if r['arm']==arm]:
   m=Model('baseline',r['seed']);saved=torch.load(a.out/f"{arm}_{r['seed']}_{r['first']}.pt",weights_only=True);m.load_state_dict(saved['state']);test=evaluate(m,tx,ty,saved['routes']);m.requires_grad_(False);cache=features(m,tx,ty);streams={kind:run_stream(cache,ty.numpy(),stream_order(ty.numpy(),r['seed']+r['first'],kind),m.head.weight.detach().numpy(),m.head.bias.detach().numpy(),'pseudo',.9,.9) for kind in ['blocked','mixed']};tests.append(dict(arm=arm,seed=r['seed'],first=r['first'],scores=test,streams=streams))
 (a.out/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
