"""Paired class-incremental prototype test of local branch versus dense learning."""
import json,time,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_mechanism_factorial import Mechanism

OUT=Path('results/class_incremental_branch_prototypes_20261013')

def save(path,obj):path.write_text(json.dumps(obj,indent=2))

class Proto:
 def __init__(self,seed,comp,rate,tau,protection='absolute'):
  s={'rate':rate,'tau':tau,'gamma':.1,'gated':True,'loss':'mse','protection':protection,'allocation':'novelty','threshold':.4,'competition':comp}
  self.model=Mechanism(seed,s);self.comp=comp;self.frozen=False
  self.prototype=torch.zeros(10,256);self.count=torch.zeros(10,dtype=torch.int64)
 @torch.no_grad()
 def response(self,x):return torch.tanh(self.model.data_features(x)@self.model.weight)
 @torch.no_grad()
 def learn(self,x,y):
  if not self.frozen:self.model.learn(x,y)
  h=self.response(x)
  for c in y.unique():
   ix=y==c;n=int(ix.sum());total=int(self.count[c])+n
   self.prototype[c].mul_(float(self.count[c])/total).add_(h[ix].sum(0)/total);self.count[c]=total
 @torch.no_grad()
 def scores(self,x,mode):
  h=self.response(x);classes=(self.count>0).nonzero().flatten();v=-(h[:,None]-self.prototype[classes][None]).square()
  if mode=='flat':score=v.sum(-1)
  elif mode=='grouped':score=v.reshape(-1,len(classes),64,4).max(-1).values.sum(-1)
  else:
   v=v.reshape(-1,len(classes),64,4);counts=self.model.links+1.;edge=(counts/counts.sum(-1,keepdim=True)).log();dp=v[:,:,0]
   for n in range(1,64):dp=(dp[:,:,:,None]+.1*edge[n-1][None,None]).max(2).values+v[:,:,n]
   score=dp.max(-1).values
  return classes[score.argmax(-1)]
 def bytes(self):return self.model.bytes()+self.prototype.numel()*4+self.count.numel()*8

@torch.no_grad()
def evaluate(m,x,y,modes,order,stages):
 seen=torch.cat([torch.tensor(p) for p in order[:stages]]);mask=torch.isin(y,seen);x=x[mask];y=y[mask]
 preds={mode:[] for mode in modes};sums=torch.zeros(10,256);counts=torch.zeros(10)
 for start in range(0,len(y),256):
  bx=x[start:start+256];by=y[start:start+256];h=m.response(bx)
  for c in by.unique():sums[c]+=h[by==c].sum(0);counts[c]+=(by==c).sum()
  for mode in modes:preds[mode].append(m.scores(bx,mode))
 means=sums/counts.clamp_min(1)[:,None]
 result={}
 for mode in modes:
  pred=torch.cat(preds[mode]);pairs=[float((pred[torch.isin(y,torch.tensor(p))]==y[torch.isin(y,torch.tensor(p))]).float().mean()) for p in order[:stages]]
  result[mode]={'accuracy':float((pred==y).float().mean()),'pair_accuracy':pairs,'old_accuracy':st.mean(pairs[:-1]) if stages>1 else None,'new_accuracy':pairs[-1]}
 return result,means

def fit(seed,comp,rate,tau,protection,x,y,ex,ey,order,frozen=False,development=False):
 m=Proto(seed,comp,rate,tau,protection);m.frozen=frozen;history=[];saved_means={};seconds=0
 modes=['flat'] if comp=='dense' or frozen or development else ['flat','grouped','communicating']
 for stage,pair in enumerate(order):
  old_classes=(m.count>0).nonzero().flatten();old=m.prototype[old_classes].clone();ids=torch.isin(y,torch.tensor(pair)).nonzero().flatten();t=time.perf_counter()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(seed*10000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):m.learn(x[ix],y[ix])
  seconds+=time.perf_counter()-t;assert torch.equal(old,m.prototype[old_classes])
  result,means=evaluate(m,ex,ey,modes,order,stage+1)
  for c in pair:saved_means[c]=means[c].clone()
  drift={str(c):float((means[c]-v).square().mean()) for c,v in saved_means.items()}
  history.append({'stage':stage+1,'readouts':result,'test_class_mean_drift':drift,'training_seconds':seconds})
 return m,history

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);rows=[];selections={}
 protocol={'question':'Does local branch learning preserve class-incremental representations better than dense learning under a shared prototype classifier?',
 'datasets':['mnist','fashion_mnist'],'stream':'Five sequential disjoint class pairs. Forward and reversed pair order; all seen classes compete in one classifier. Two current-class passes, no replay, no task ID at inference, no masking to current pair. The harness uses pairs only to construct the stream and score metrics; learner gets inputs/labels only.',
 'prototype':'Ten full256-dimensional FP32 class means plus ten int64 counts. Online cumulative post-update response means of current labeled batches. Old prototypes never refreshed; repeated training presentations count repeatedly. No saved individual inputs or branch configurations. Explicit prototype memory counted. Only observed classes eligible.',
 'primary':'Both models use identical full-response squared Euclidean nearest-prototype classification. Branch model grouped64x4 competition; dense trainsall256units. Same fixedcode supervision, inputnormalization, tanh and usage protection. Dense lacksgraph andupdates4timesas manyunits; nominalrate/tau scaling acknowledged, not equalcompute.',
 'secondary':'Branch-only grouped max compatibility and learnedneighbor chainmaxsum using eachcandidate class prototype; allcandidate classes scored without true-label routing. Same finalweights/prototypes forflat/grouped/communicating. Secondary is not sameclassifier as dense.',
 'development':'One development seed per dataset421/431;1000heldout/class fromtrain,50kfit;forwardpairorder. Eacharchitecture independently selects flatprototype finalaccuracy over rate[.25,1] tau[1000,10000]; dense rate/4,tau*4. No official test for selection.',
 'confirmation':'Fresh422-424MNIST/432-434Fashion, forward/reversepairorder;all60ktrain/full10ktest; fixeddevelopmentselection. Protection-off reuses chosen settings, not independently retuned. Frozeninitialdensefeaturebaseline no supervisedweightupdates. Order repeats paired, not sixindependentseeds.',
 'drift':'Test class means used only for diagnostic drift from first acquisition to finalweights; never enter training, prototypes or selection inconfirmation. Prototype-to-finalclassmean mismatch also reported. Feature mean drift is a proxy, not a complete forgetting metric.',
 'memory':'Branches822800bytes;dense/frozen818768bytes incl prototypes. About0.5% difference not padded. Workingmemory/gradients excluded.'}
 save(OUT/'protocol.json',protocol);(OUT/'source.py').write_text(Path(__file__).read_text())
 # Verify Euclidean inference equals the explicit distance classifier; labels absent from score.
 torch.manual_seed(9);m=Proto(9,'dense',.25,40000);m.frozen=True;xx=torch.rand(20,784);yy=torch.arange(20)%2;m.learn(xx,yy)
 expected=(-(m.response(xx)[:,None]-m.prototype[:2][None]).square().sum(-1)).argmax(-1)
 assert torch.equal(m.scores(xx,'flat'),expected)
 save(OUT/'verification.json',{'nearest_prototype_matches_explicit_distance':True,'old_prototype_invariance_asserted_each_stage':True})
 for ds,dev in [('mnist',421),('fashion_mnist',431)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  rng=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;forward=[[0,1],[2,3],[4,5],[6,7],[8,9]];selected={}
  for comp in ['grouped','dense']:
   candidates=[]
   for base_rate in [.25,1.]:
    for base_tau in [1000.,10000.]:
     rate=base_rate/(4 if comp=='dense' else 1);tau=base_tau*(4 if comp=='dense' else 1)
     m,h=fit(dev,comp,rate,tau,'absolute',x[mask],y[mask],x[val],y[val],forward,development=True)
     a=h[-1]['readouts']['flat']['accuracy'];candidates.append({'rate':rate,'tau':tau,'accuracy':a,'history':h});print('dev',ds,comp,rate,tau,round(a*100,2),flush=True)
   selected[comp]={'chosen':max(candidates,key=lambda c:c['accuracy']),'candidates':candidates}
  selections[ds]=selected;save(OUT/'selection.json',selections)
  for seed in range(dev+1,dev+4):
   for order_name,order in [('forward',forward),('reverse',forward[::-1])]:
    for arm,comp,protection,frozen in [('branch_protected','grouped','absolute',False),('dense_protected','dense','absolute',False),('branch_unprotected','grouped','off',False),('dense_unprotected','dense','off',False),('frozen','dense','absolute',True)]:
     s=selected[comp]['chosen'];m,h=fit(seed,comp,s['rate'],s['tau'],protection,x,y,tx,ty,order,frozen)
     counts=torch.bincount(ty,minlength=10).float();sums=torch.zeros(10,256)
     with torch.no_grad():
      for start in range(0,len(ty),256):
       hh=m.response(tx[start:start+256]);by=ty[start:start+256]
       for cc in by.unique():sums[cc]+=hh[by==cc].sum(0)
     mismatch=(sums/counts[:,None]-m.prototype).square().mean(-1).tolist()
     row={'dataset':ds,'seed':seed,'order':order_name,'pairs':order,'arm':arm,'rate':s['rate'],'tau':s['tau'],'history':h,'numeric_bytes':m.bytes(),'final_prototype_classmean_mse':mismatch};rows.append(row);save(OUT/'results.json',rows)
     print('test',ds,seed,order_name,arm,{mode:round(a['accuracy']*100,2) for mode,a in h[-1]['readouts'].items()},flush=True)
 save(OUT/'completion.json',{'trained_confirmation_models':len(rows),'complete':len(rows)==60})
if __name__=='__main__':main()
