"""Matched-budget depth comparison with factual CE and layer-specific oracle audit."""
import argparse, json, math, time, statistics as st
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts import multilayer_branch_comparison as base
from bdh_cl.data.mnist import load_raw, SPLITS
OUT=Path('results/controlled_branch_depth_20261020')
ARMS=['mlp1','mlp2','branch1_exact','branch1_proxy','branch2_exact','branch2_proxy']
class Branch(nn.Module):
 def __init__(self,depth):
  super().__init__(); self.depth=depth; self.n=25 if depth==1 else 24
  self.layers=nn.ModuleList([nn.Linear(784 if i==0 else self.n,self.n*4) for i in range(depth)])
  self.head=nn.Linear(self.n,10); self.register_buffer('error',torch.full((depth,self.n,4),math.log(10)))
 def routes(self):return self.error.argmin(-1)
 def forward(self,x,record=False,routes=None):
  route=self.routes() if routes is None else routes; records=[]
  for i,layer in enumerate(self.layers):
   b=torch.tanh(layer(x)).reshape(-1,self.n,4); h=b.gather(-1,route[i][None,:,None].expand(len(b),-1,-1)).squeeze(-1)
   if record:
    if h.requires_grad:h.retain_grad()
    records.append((b,h))
   x=h
  logits=self.head(x)
  return (logits,records,route.clone()) if record else logits
 def alternatives(self,records,route,logits):
  all_alts=[]; n=self.n
  for i,(b,h) in enumerate(records):
   if i==self.depth-1:a=logits[:,None,None,:]+(b-h[:,:,None])[:,:,:,None]*self.head.weight.T[None,:,None,:]
   else:
    changed=h[:,None,None,:]+(b-h[:,:,None])[:,:,:,None]*torch.eye(n)[None,:,None,:]
    v=changed.reshape(-1,n)
    for j in range(i+1,self.depth):
     z=torch.tanh(self.layers[j](v)).reshape(-1,n,4);v=z.gather(-1,route[j][None,:,None].expand(len(z),-1,-1)).squeeze(-1)
    a=self.head(v).reshape(len(h),n,4,10)
   all_alts.append(a)
  return torch.stack(all_alts,1)
 @torch.no_grad()
 def update(self,est):self.error.mul_(.95).add_(est,alpha=.05)
 @torch.no_grad()
 def selected_forward(self,x):
  for i,layer in enumerate(self.layers):
   ix=torch.arange(self.n)*4+self.routes()[i];x=torch.tanh(F.linear(x,layer.weight[ix],layer.bias[ix]))
  return self.head(x)
def make(seed,arm):
 torch.manual_seed(seed)
 if arm.startswith('branch'):return Branch(int(arm[6]))
 width=99 if arm=='mlp1' else 89; depth=int(arm[-1]); layers=[]
 for i in range(depth):layers.extend([nn.Linear(784 if i==0 else width,width),nn.Tanh()])
 return nn.Sequential(*layers,nn.Linear(width,10))
def step(m,opt,x,y,arm):
 opt.zero_grad(set_to_none=True)
 if arm.startswith('mlp'):loss=F.cross_entropy(m(x),y);loss.backward()
 else:
  log,recs,route=m(x,True);loss=F.cross_entropy(log,y)
  if arm.endswith('exact'):
   with torch.no_grad():
    alt=m.alternatives(recs,route,log);target=y[:,None,None,None].expand(alt.shape[:-1]).flatten();est=F.cross_entropy(alt.reshape(-1,10),target,reduction='none').reshape(alt.shape[:-1]).mean(0)
   loss.backward()
  else:
   loss.backward();est=torch.stack([loss.detach()+(h.grad.detach()[:,:,None]*(b.detach()-h.detach()[:,:,None])).sum(0) for b,h in recs])
 torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 if not arm.startswith('mlp'):m.update(est)
 return float(loss.detach())
def fit(*args,**kwargs):
 arm=args[1];m,r=base.fit(*args,**kwargs)
 if 'routes' in r:r['bank_bytes_packed']=len(r['routes'])*m.depth*m.n
 return m,r
@torch.no_grad()
def audit(m,r,x,y,order):
 before={k:v.clone() for k,v in m.state_dict().items()};bank=torch.tensor(r['routes']);latest=m.routes().clone(); out={}
 subsets={'all_historical':list(range(m.depth)),'all_latest':[]}
 if m.depth==2:subsets.update(first_historical=[0],second_historical=[1])
 for name,layers in subsets.items():
  pred=torch.empty_like(y);changes=[]
  for task,pair in enumerate(order):
   route=latest.clone()
   for layer in layers:route[layer]=bank[task,layer]
   ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten();pred[ids]=base.scores(m,x[ids],route).argmax(-1)
   changes.append([(bank[task,i]!=latest[i]).float().mean().item() for i in range(m.depth)])
  out[name]=base.metrics(pred,y,order)
 out['fraction_route_changes_from_snapshot_by_task_layer']=changes
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 assert abs(out['all_historical']['accuracy']-r['oracle_routes']['accuracy'])<1e-7
 assert abs(out['all_latest']['accuracy']-r['accuracy'])<1e-7
 return out
def verify():
 result={};x=torch.rand(5,784);y=torch.arange(5)
 for depth in [1,2]:
  m=make(1,f'branch{depth}_exact');log,rec,route=m(x,True);alt=m.alternatives(rec,route,log)
  for i in range(depth):
   for n,k in [(0,1),(m.n-1,3)]:
    changed=route.clone();changed[i,n]=k;assert torch.allclose(m(x,routes=changed),alt[:,i,n,k],atol=1e-6)
  assert torch.allclose(m(x),m.selected_forward(x),atol=1e-6)
  loss=F.cross_entropy(log,y);grads=torch.autograd.grad(loss,[h for b,h in rec],retain_graph=True)
  for i,((b,h),g) in enumerate(zip(rec,grads)):
   direction=torch.randn_like(h);eps=.001
   def downstream(v):
    for j in range(i+1,depth):
     z=torch.tanh(m.layers[j](v)).reshape(-1,m.n,4);v=z.gather(-1,route[j][None,:,None].expand(len(z),-1,-1)).squeeze(-1)
    return F.cross_entropy(m.head(v),y)
   finite=(downstream(h+eps*direction)-downstream(h-eps*direction))/(2*eps)
   assert torch.allclose(finite,(g*direction).sum(),atol=.001,rtol=.03)
  result[f'depth{depth}_exact_and_gradient_checks']=True
 result['parameters']={a:sum(p.numel() for p in make(1,a).parameters()) for a in ARMS};return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=ARMS);p.add_argument('--verify',action='store_true');args=p.parse_args();torch.set_num_threads(2);base.make=make;base.step=step
 # base.fit treats literal mlp specially; normalize only its arm tests without touching its source.
 import inspect
 source=inspect.getsource(base.fit).replace("arm!='mlp'","not arm.startswith('mlp')").replace("arm=='mlp'","arm.startswith('mlp')")
 exec(source,base.__dict__)
 if args.verify:print(json.dumps(verify(),indent=2));return
 if args.profile:
  # The base profiler also has literal arm tests; adapt its branching, not the algorithm.
  src=inspect.getsource(base.profile).replace("arm!='mlp'","not arm.startswith('mlp')").replace("arm=='mlp'","arm.startswith('mlp')")
  exec(src,base.__dict__);base.profile(args.profile,OUT/f'profile_{args.profile}.json');return
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();base.save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 base.save(OUT/'protocol.json',{'question':'Controlled one-versus-two hidden layers plus layer-specific route restoration.', 'arms':ARMS,'budgets':'branch1 78760; branch2 78010; mlp1 78715; mlp2 78775. Within1%. Matched parameters, not pooled widths. One layer25x4, two layers24x4 each.','loss':'Factual CE only in all arms. Exact counterfactuals detached; proxy first order from factual backward. EMA .95, Adam clip5, batch128,2 passes/current pair, no replay/pretraining/protection/input retrieval.','development':'Seed530 stratified50k/10k; each arm and sequential/joint mode separately select LR .0003,.001,.003 by final validation accuracy, acquisition tiebreak. No test selection.','confirmation':'Seeds531-535, forward/reverse five digit pairs; full60k/10k; joint controls same120k presentations. Five independent seeds, orders paired. 36dev+90confirmation.','oracle':'Task identity selects historical routes; all10outputs eligible, final weights unchanged. First-only,second-only,both,neither restore for depth2. MLP pair mask different assistance. Layer effects include interactions, not causal locality of storage.','profiling':'Fresh process per arm, CPU2threads, same base profiler. No GPU/deep asymptotic claims.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
 gen=torch.Generator().manual_seed(530);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;selected={};rows=[]
 for mode in ['sequential','joint']:
  selected[mode]={}
  for arm in ARMS:
   candidates=[]
   for lr in [.0003,.001,.003]:
    _,r=fit(530,arm,lr,x[mask],y[mask],x[valid],y[valid],SPLITS,mode,False);candidates.append(r);print('DEV',mode,arm,lr,round(100*r['accuracy'],2),flush=True)
   selected[mode][arm]={'chosen':max(candidates,key=lambda z:(z['accuracy'],z.get('acquisition',0))),'candidates':candidates};base.save(OUT/'selection.json',selected)
 for seed in range(531,536):
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   for arm in ARMS:
    m,r=fit(seed,arm,selected['sequential'][arm]['chosen']['lr'],x,y,tx,ty,order,'sequential');r.update(seed=seed,order=name)
    if arm.startswith('branch'):r['layer_restoration']=audit(m,r,tx,ty,order)
    rows.append(r);base.save(OUT/'results.json',rows);torch.save({'state_dict':m.state_dict(),'routes':r.get('routes'),'arm':arm},OUT/'models'/f'{seed}_{name}_{arm}.pt');print('TEST',seed,name,arm,round(100*r['accuracy'],2),'oracle',round(100*r.get('oracle_routes',r['final'])['accuracy'],2),flush=True)
  for arm in ARMS:
   _,r=fit(seed,arm,selected['joint'][arm]['chosen']['lr'],x,y,tx,ty,SPLITS,'joint');r.update(seed=seed,order='mixed');rows.append(r);base.save(OUT/'results.json',rows);print('JOINT',seed,arm,round(100*r['accuracy'],2),flush=True)
 base.save(OUT/'completion.json',{'complete':len(rows)==90,'confirmation_runs':len(rows),'development_runs':36})
if __name__=='__main__':main()
