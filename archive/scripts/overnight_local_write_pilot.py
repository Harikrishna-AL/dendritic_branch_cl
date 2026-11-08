"""Development-only: a branch-local persistence trace controls memory writes."""
import json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/local_write_pilot')
class Gated:
 def __init__(self,d,slots,age):self.memory=Multi(d,slots,.995 if d==16 else 1.);self.threshold=age;self.age=np.zeros(25,np.uint8);self.previous=np.full(25,255,np.uint8);self.accepted=np.zeros(25,np.int64)
 def update(self,q,route):
  self.age=np.where(route==self.previous,np.minimum(self.age.astype(np.int16)+1,255),1).astype(np.uint8);self.previous=route.copy().astype(np.uint8);allowed=self.age>=self.threshold;ix=np.arange(25);old=self.memory.sums[ix,route].copy(),self.memory.count[ix,route].copy();self.memory.update(q,route);self.memory.sums[ix[~allowed],route[~allowed]]=old[0][~allowed];self.memory.count[ix[~allowed],route[~allowed]]=old[1][~allowed];self.accepted[allowed]+=len(q)
 def routes(self,q):return self.memory.routes(q)
 def size(self):return self.memory.size()+self.age.nbytes+self.previous.nbytes
class Study:
 def __init__(self):self.gated={(d,s,age):Gated(d,s,age) for d,s in [(16,4),(64,4)] for age in [1,5,10,25]};self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.gated.values():obj.update(q,route)
  self.local.update(q,route);t=np.zeros((len(q),25,4),np.float32);t[:,np.arange(25),route]=1;self.ridge.update(q,t.reshape(-1,100));self.accepted+=len(q)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'scope':'Freshdevelopment580/581,50kfit/10kTRAINvalidation only. Noofficialtestmetricsorlabelsused. Branch-localstableactivityage1/5/10/25gatesprototypewritingperneuron,notglobalpattern. Traceagescapped255andneverresetattaskboundaries,50Bextra. D16S4decay.995,D64S4decay1fixedfromearlierdevelopment. Currentq+preupdatebranchchoicesonly,noIDs/labels/boundaries/futurebank inroutingwrites. BackboneunchangedCE/proxy2epochs/currentpair. Compareage1withinidenticalstates plusglobalstability10 andoriginalaverage/ridge/directcontrols. Allconditionsretained.'})
 raw=load_raw('data_cache/mnist');allx=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;ally=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);records=[];e.Memory=Study
 for seed in [580,581]:
  gen=torch.Generator().manual_seed(seed);valid=torch.cat([torch.nonzero(ally==c).flatten()[torch.randperm(int((ally==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(ally),dtype=torch.bool);mask[valid]=False;vx,vy=allx[valid],ally[valid];m,mem,direct,bank,audit,timing=e.train(seed,allx[mask],ally[mask],SPLITS)
  with torch.no_grad():
   q=np.concatenate([e.keys(b) for b in vx.split(512)]);b=torch.cat([torch.tanh(m.layers[0](bx)).reshape(-1,25,4) for bx in vx.split(512)])
   def acc(route):return float((m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
   rows=[]
   for (d,slots,age),obj in mem[1].gated.items():rows.append({'method':'local_age','dimension':d,'slots':slots,'age':age,'accuracy':acc(obj.routes(q)),'extra_bytes':obj.size(),'mean_accepted_per_neuron':float(obj.accepted.mean())})
   for d in [16,64]:
    obj=mem[10].gated[(d,4,1)];rows.append({'method':'global_stability10','dimension':d,'accuracy':acc(obj.routes(q)),'extra_bytes':obj.size()})
   rows.append({'method':'original_local','accuracy':acc(mem[1].local.routes(q,0)),'extra_bytes':26000});rows.append({'method':'global_ridge','accuracy':acc((q@mem[1].ridge.weight(.1).T).reshape(-1,25,4).argmax(-1)),'extra_bytes':41984});rows.append({'method':'direct_ridge','accuracy':float(((q@direct.weight(1.).T).argmax(-1)==vy.numpy()).mean()),'extra_bytes':direct.size()})
  records.append({'seed':seed,'validation_only':True,'rows':rows});e.save(OUT/'results.json',records);print('RESULT',seed,rows,flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==2,'validation_readouts':26,'official_test_used':False})
if __name__=='__main__':main()
