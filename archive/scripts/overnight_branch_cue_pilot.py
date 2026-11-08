"""Validation-only pilot: activity versus per-image usefulness as local cue targets."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/cue_pilot')
def save(p,x):p.write_text(json.dumps(x,indent=2))
@torch.no_grad()
def cues(x):
 globalq=e.keys(x);im=x.reshape(-1,28,28);parts=torch.stack([im[:,top:top+7,left:left+9].flatten(1) for top in [0,5,10,15,21] for left in [0,5,10,15,19]],1);parts=F.normalize(parts,dim=-1);parts=F.normalize(torch.cat([parts,torch.full((*parts.shape[:2],1),.2)],-1),dim=-1)
 return {'global':np.broadcast_to(globalq[:,None],(len(x),25,64)),'patch':parts.numpy().copy()},globalq
class Local:
 def __init__(self,decay):self.sums=np.zeros((25,4,64),np.float32);self.count=np.zeros((25,4),np.float32);self.decay=decay
 def update(self,q,target):
  assignment=np.eye(4,dtype=np.float32)[target];count=assignment.sum(0);active=count>0;self.sums[active]*=self.decay;self.count[active]*=self.decay;self.sums+=np.einsum('bnk,bnd->nkd',assignment,q);self.count+=count
 def routes(self,q):
  score=np.einsum('bnd,nkd->bnk',q,e.norm(self.sums));return np.where(self.count[None]>0,score,-1e3).argmax(-1)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());save(OUT/'protocol.json',{'scope':'Development validation only; no official test labels/images loaded. Seed560,50kfit/10kstratifiedvalidation. OriginalbackbonefactualCE/proxyusefulnessAdam.003,2currentepochs. No memory writes see IDs/boundaries/futureconfigs. Utilityteacher uses current labels via single-neuron exact alternative errors; not claimed biologicallylocal. Global/patch63+bias64keys,activity/usefulperimage targets,decay1/.995,stability1/10. All16localconditions retained,26000B each. Globalridge routing controls pertarget41984B anddirectclassridge18944B. No oldtask replay or routebank in predictions.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);gen=torch.Generator().manual_seed(560);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;vx,vy=x[valid],y[valid];x,y=x[mask],y[mask];m=e.backbone.make(560,'branch1_proxy');opt=torch.optim.Adam(m.parameters(),lr=.003)
 local={(cue,target,decay,stable):Local(decay) for cue in ['global','patch'] for target in ['activity','useful'] for decay in [1.,.995] for stable in [1,10]};ridge={target:e.Ridge(100) for target in ['activity','useful']};direct=e.Ridge(10);previous=None;streak=0;start=time.perf_counter();history=[]
 for stage,pair in enumerate(SPLITS):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   gen=torch.Generator().manual_seed(560*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
    bx,by=x[ix],y[ix]
    with torch.no_grad():
     q,gq=cues(bx);route=m.routes()[0].numpy();streak=streak+1 if previous is not None and np.array_equal(route,previous) else 1;previous=route.copy();log,rec,r=m(bx,True);alt=m.alternatives(rec,r,log)[:,0];loss=F.cross_entropy(alt.reshape(-1,10),by[:,None,None].expand(alt.shape[:-1]).flatten(),reduction='none').reshape(alt.shape[:-1]);targets={'activity':np.broadcast_to(route,(len(bx),25)),'useful':loss.argmin(-1).numpy()}
     for (cue,target,decay,stable),obj in local.items():
      if streak>=stable:obj.update(q[cue],targets[target])
     for target,obj in ridge.items():obj.update(gq,np.eye(4,dtype=np.float32)[targets[target]].reshape(-1,100))
     direct.update(gq,F.one_hot(by,10).float().numpy())
    e.backbone.step(m,opt,bx,by,'branch1_proxy')
  print('STAGE',stage+1,round(time.perf_counter()-start,1),flush=True)
 with torch.no_grad():
  qparts={k:[] for k in ['global','patch']};gparts=[];responses=[]
  for bx in vx.split(512):
   qs,gq=cues(bx)
   for k in qparts:qparts[k].append(qs[k])
   gparts.append(gq);responses.append(torch.tanh(m.layers[0](bx)).reshape(-1,25,4))
  q={k:np.concatenate(v) for k,v in qparts.items()};gq=np.concatenate(gparts);b=torch.cat(responses);rows=[]
  def accuracy(route):return float((m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
  for setting,obj in local.items():
   cue,target,decay,stable=setting;rows.append({'method':'local','cue':cue,'target':target,'decay':decay,'stable':stable,'accuracy':accuracy(obj.routes(q[cue])),'extra_bytes':26000})
  for target,obj in ridge.items():rows.append({'method':'global_ridge','target':target,'accuracy':accuracy((gq@obj.weight(.1).T).reshape(-1,25,4).argmax(-1)),'extra_bytes':obj.size()})
  rows.append({'method':'direct_ridge','accuracy':float(((gq@direct.weight(1.).T).argmax(-1)==vy.numpy()).mean()),'extra_bytes':direct.size()});ordinary=accuracy(np.broadcast_to(m.routes()[0].numpy(),(len(vy),25)))
 save(OUT/'results.json',{'validation_only':True,'rows':rows,'ordinary_accuracy':ordinary,'all_candidate_seconds':time.perf_counter()-start});torch.save(m.state_dict(),OUT/'backbone.pt');save(OUT/'completion.json',{'complete':len(rows)==19,'official_test_accessed':False});print('RESULTS',sorted(rows,key=lambda r:-r['accuracy']),flush=True)
if __name__=='__main__':main()
