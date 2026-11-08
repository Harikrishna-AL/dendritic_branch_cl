"""Conventional trained-head SGD feature learner, evaluated by class prototypes."""
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.class_incremental_branch_prototypes import Proto,evaluate,save
OUT=Path('results/class_incremental_mlp_prototypes_20261013')
class Ordinary(Proto):
 def __init__(self,seed,lr):
  super().__init__(seed,'dense',.25,40000);self.weight=torch.nn.Parameter(self.model.weight.clone());generator=torch.Generator().manual_seed(seed+8000)
  self.head=torch.nn.Parameter(torch.randn(256,10,generator=generator)*.02);self.bias=torch.nn.Parameter(torch.zeros(10));self.opt=torch.optim.SGD([self.weight,self.head,self.bias],lr=lr)
  # Removed localcoded state from ordinarymodel; only inputweight tensor reused.
  self.model=None;self.lr=lr
 def response(self,x):return torch.tanh(torch.cat([F.normalize(x,dim=1),torch.ones(len(x),1)],1)@self.weight)
 def learn(self,x,y):
  self.opt.zero_grad(set_to_none=True);h=self.response(x);F.cross_entropy(h@self.head+self.bias,y).backward();self.opt.step()
  with torch.no_grad():
   h=self.response(x)
   for c in y.unique():
    ix=y==c;n=int(ix.sum());total=int(self.count[c])+n
    self.prototype[c].mul_(float(self.count[c])/total).add_(h[ix].sum(0)/total);self.count[c]=total
 def bytes(self):return sum(t.numel()*t.element_size() for t in [self.weight,self.head,self.bias,self.prototype,self.count])
def fit(seed,lr,x,y,ex,ey,order):
 m=Ordinary(seed,lr);history=[];seconds=0.;saved={}
 for stage,pair in enumerate(order):
  oldclasses=(m.count>0).nonzero().flatten();old=m.prototype[oldclasses].clone();ids=torch.isin(y,torch.tensor(pair)).nonzero().flatten();t=time.perf_counter()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(seed*10000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):m.learn(x[ix],y[ix])
  seconds+=time.perf_counter()-t;sums=torch.zeros(10,256);counts=torch.zeros(10,dtype=torch.int64)
  with torch.no_grad():
   for ix in ids.split(256):
    h=m.response(x[ix]);by=y[ix]
    for c in by.unique():sums[c]+=h[by==c].sum(0);counts[c]+=(by==c).sum()
   for c in pair:m.prototype[c]=sums[c]/counts[c];m.count[c]=counts[c]
  assert torch.equal(old,m.prototype[oldclasses]);result,means=evaluate(m,ex,ey,['flat'],order,stage+1)
  for c in pair:saved[c]=means[c].clone()
  history.append({'stage':stage+1,'readouts':result,'test_class_mean_drift':{str(c):float((means[c]-v).square().mean()) for c,v in saved.items()},'training_seconds':seconds})
 return m,history

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);rows=[];selection={};forward=[[0,1],[2,3],[4,5],[6,7],[8,9]]
 save(OUT/'protocol.json',{'architecture':'256tanhhiddenunits, sameinitialinputweightsandnormalizedinput asbranches. Learned10-classlinearhead trainsfeaturesviaalltenclasscrossentropyandSGDwithoutmomentum. Primaryevaluationnearestfullresponseprototype, currentclassprototypefittedaftertwoepochs. No IDs/replay/pairmask at inference. Oldprototypes unchanged.', 'development':'Same421/431devsplit;SGDlr[.03,.3,1] independently selected onfinal10classvalidationprototypeaccuracy. Not officialtest. Fixedlrconfirmation;notcompleteoptimizersearch orstateofartCL.', 'confirmation':'Same422-424/432-434pairedseedsandforward/reverseorders asbranches.12models. All60ktrain/full10ktest.', 'storage':'824440persistentnumericbytes includinglearnedhead,prototypes/counts;about0.2%more thanbranches822800. Gradientworkingmemoryexcluded.'})
 (OUT/'source.py').write_text(Path(__file__).read_text())
 for ds,dev in [('mnist',421),('fashion_mnist',431)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  rng=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;candidates=[]
  for lr in [.03,.3,1.]:
   m,h=fit(dev,lr,x[mask],y[mask],x[val],y[val],forward);a=h[-1]['readouts']['flat']['accuracy'];candidates.append({'lr':lr,'accuracy':a,'history':h});print('dev',ds,lr,round(a*100,2),flush=True)
  chosen=max(candidates,key=lambda a:a['accuracy']);selection[ds]={'chosen':chosen,'candidates':candidates};save(OUT/'selection.json',selection)
  for seed in range(dev+1,dev+4):
   for ordername,order in [('forward',forward),('reverse',forward[::-1])]:
    m,h=fit(seed,chosen['lr'],x,y,tx,ty,order);rows.append({'dataset':ds,'seed':seed,'order':ordername,'arm':'backprop_mlp','lr':chosen['lr'],'history':h,'numeric_bytes':m.bytes()});save(OUT/'results.json',rows);print('test',ds,seed,ordername,round(h[-1]['readouts']['flat']['accuracy']*100,2),flush=True)
 save(OUT/'verification.json',{'complete_12_models':len(rows)==12,'old_prototypes_unchanged_each_stage':True,'numeric_bytes':sorted(set(x['numeric_bytes'] for x in rows))})
if __name__=='__main__':main()
