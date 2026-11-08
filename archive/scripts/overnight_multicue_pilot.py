"""Validation-only multimodal branch cues with size and feature-quality controls."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/multicue_pilot')
def compress(q,d):
 if d==64:return q
 z=F.adaptive_avg_pool1d(torch.tensor(q[:,:63])[:,None],d-1).squeeze(1).numpy();z=e.norm(z);return e.norm(np.concatenate([z,np.full((len(q),1),.2,np.float32)],-1)).astype(np.float32)
class Multi:
 def __init__(self,d,slots,decay):self.d=d;self.slots=slots;self.decay=decay;self.sums=np.zeros((25,4,slots,d),np.float32);self.count=np.zeros((25,4,slots),np.float32)
 def update(self,q,route):
  q=compress(q,self.d);ix=np.arange(25);active=self.sums[ix,route].copy();counts=self.count[ix,route].copy();new=counts.sum(-1)==0
  if new.any():
   chosen=[0]
   while len(chosen)<self.slots:
    similarity=q@q[chosen].T;chosen.append(int(similarity.max(-1).argmin()))
   active[new]=q[chosen][None];counts[new]=1
  sim=np.einsum('bd,nsd->bns',q,e.norm(active));winner=sim.argmax(-1);assignment=np.eye(self.slots,dtype=np.float32)[winner];added=assignment.sum(0);used=added>0;active[used]*=self.decay;counts[used]*=self.decay;active+=np.einsum('bns,bd->nsd',assignment,q);counts+=added;self.sums[ix,route]=active;self.count[ix,route]=counts
 def routes(self,q):
  q=compress(q,self.d);scores=np.einsum('bd,nksd->bnks',q,e.norm(self.sums));scores=np.where(self.count[None]>0,scores,-1e3);return scores.max(-1).argmax(-1)
 def size(self):return self.sums.nbytes+self.count.nbytes
class Study:
 def __init__(self):self.models={(d,s,decay):Multi(d,s,decay) for d,s in [(16,1),(16,4),(32,1),(32,2),(64,1),(64,2),(64,4)] for decay in [1.,.995]};self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.models.values():obj.update(q,route)
  self.local.update(q,route);target=np.zeros((len(q),25,4),np.float32);target[:,np.arange(25),route]=1;self.ridge.update(q,target.reshape(-1,100));self.accepted+=len(q)
class Direct:
 def __init__(self,d):self.d=d;self.cov=np.eye(d,dtype=np.float32);self.cross=np.zeros((10,d),np.float32)
 def update(self,q,y):q=compress(q,self.d);self.cov+=q.T@q;self.cross+=F.one_hot(y,10).float().numpy().T@q
 def predict(self,q):w=np.linalg.solve(self.cov.astype(np.float64),self.cross.T.astype(np.float64)).T;return (compress(q,self.d)@w.T).argmax(-1)
def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=560);seed=parser.parse_args().seed
 if seed!=560:OUT=OUT.parent/f"multicue_pilot_seed{seed}"
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'seed':seed,'scope':'Developmentvalidationonly;noofficialtestloaded. Seed-specific50k/10k,unchangedbackbone. Multicomponentinputkeyprototypes perbranch,single/multiple and16/32/64Dcontrols; fixedbudgetapprox26KB:64D1slot/32D2slots/16D4slots. Also64D2/4slotscapacityscaling. Componentmeans receive currentkey/currentpreupdateglobalrouteonly; no labels/taskIDs/boundaries/futurearchives. Prototypes initialized fromcurrentbatchfarthest-first and updated via minibatchcosineassignment andsum/count. They are explicit aggregateinputmemory,notclaimedzeromemory,neverreplayed. Decay1/.995,stable1/10 all28conditionsretained. Directclassridge16/32/64featurecontrols,labelsupervisedcurrentonly;globalridge64routecontrol. No task replay.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);g=torch.Generator().manual_seed(seed);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=g)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;vx,vy=x[valid],y[valid];x,y=x[mask],y[mask]
 # Reuse original trainer; direct feature-quality controls see identical current labels through keys hook.
 direct={d:Direct(d) for d in [16,32,64]};original_step=e.backbone.step
 def step(m,opt,bx,by,arm):
  q=e.keys(bx)
  for obj in direct.values():obj.update(q,by)
  return original_step(m,opt,bx,by,arm)
 e.backbone.step=step;e.Memory=Study;m,mem,unused,bank,audit,timing=e.train(seed,x,y,SPLITS);
 if seed==560:
  saved=torch.load(OUT.parent/'cue_pilot/backbone.pt',weights_only=True);assert all(torch.equal(v,saved[k]) for k,v in m.state_dict().items())
 torch.save(m.state_dict(),OUT/'backbone.pt')
 with torch.no_grad():
  q=np.concatenate([e.keys(b) for b in vx.split(512)]);responses=torch.cat([torch.tanh(m.layers[0](b)).reshape(-1,25,4) for b in vx.split(512)])
  def acc(route):return float((m.head(responses.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
  rows=[]
  for stable,mm in mem.items():
   for (d,slots,decay),obj in mm.models.items():rows.append({'method':'multicue','dimension':d,'slots':slots,'decay':decay,'stable':stable,'accuracy':acc(obj.routes(q)),'extra_bytes':obj.size()})
   rows.append({'method':'original_local','stable':stable,'accuracy':acc(mm.local.routes(q,0)),'extra_bytes':26000});rows.append({'method':'global_ridge','stable':stable,'accuracy':acc((q@mm.ridge.weight(.1).T).reshape(-1,25,4).argmax(-1)),'extra_bytes':mm.ridge.size()})
  for d,obj in direct.items():rows.append({'method':'direct_ridge','dimension':d,'accuracy':float((obj.predict(q)==vy.numpy()).mean()),'extra_bytes':obj.cov.nbytes+obj.cross.nbytes})
 e.save(OUT/'results.json',{'validation_only':True,'rows':rows,'timing_all_candidates':timing,'backbone_matches_initial':True if seed==560 else None});e.save(OUT/'completion.json',{'complete':len(rows)==35,'official_test_accessed':False});print('TOP',sorted(rows,key=lambda r:-r['accuracy'])[:10],flush=True)
if __name__=='__main__':main()
