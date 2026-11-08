"""New development seeds, fixed local prototype states, three scoring rules."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi,compress,Direct
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/centered_cue_pilot');SHAPES=[(16,4,.995),(64,1,.995),(64,4,1.)]
def decode(obj,q,mode):
 q=compress(q,obj.d);p=obj.sums/np.maximum(obj.count[:,:,:,None],1)
 if mode=='local_centered':
  mean=obj.sums.sum((1,2))/np.maximum(obj.count.sum((1,2))[:,None],1);a=e.norm(q[:,None]-mean[None]);b=e.norm(p-mean[:,None,None]);scores=np.einsum('bnd,nksd->bnks',a,b)
 elif mode=='spatial_centered':
  a=q[:,:-1]-q[:,:-1].mean(-1,keepdims=True);b=p[:,:,:,:-1]-p[:,:,:,:-1].mean(-1,keepdims=True);scores=np.einsum('bd,nksd->bnks',e.norm(a),e.norm(b))
 else:raise ValueError(mode)
 scores=np.where(obj.count[None]>0,scores,-1e3);return scores.max(-1).argmax(-1)
class Study:
 def __init__(self):self.models={(d,s):Multi(d,s,decay) for d,s,decay in SHAPES};self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.models.values():obj.update(q,route)
  self.local.update(q,route);target=np.zeros((len(q),25,4),np.float32);target[:,np.arange(25),route]=1;self.ridge.update(q,target.reshape(-1,100));self.accepted+=len(q)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'scope':'Newdevelopmentseeds570/571,50kfit/10kheldoutTRAINvalidation. Noofficialtestarraysusedinmetrics/settings. ExistingfixedSHAPES16D4slotsdecay.995,64D1slot.995,64D4slot1. No new hyperparameter search. Exactsameassociationstates, comparecosineversuslocalmean-centeredcosineversuspervectorfeature-centeredcosine(excludesbias). Localmean derived fromper-neuronbranchprototype sums/counts,noextra persistentstate/fullcovariance. Samecurrentkey/activitywrites,noIDs/boundaries/classlabels/futurebank/oldreplay. Originalbackboneunchanged. Controlsoriginalaverage,globalridge,directclassridge64. Centeringneednothelp;allconditionsretained.'})
 raw=load_raw('data_cache/mnist');allx=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;ally=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);records=[];e.Memory=Study
 for seed in [570,571]:
  gen=torch.Generator().manual_seed(seed);valid=torch.cat([torch.nonzero(ally==c).flatten()[torch.randperm(int((ally==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(ally),dtype=torch.bool);mask[valid]=False;x,y=allx[mask],ally[mask];vx,vy=allx[valid],ally[valid];m,mem,direct,bank,audit,timing=e.train(seed,x,y,SPLITS)
  with torch.no_grad():
   q=np.concatenate([e.keys(b) for b in vx.split(512)]);responses=torch.cat([torch.tanh(m.layers[0](b)).reshape(-1,25,4) for b in vx.split(512)])
   def acc(route):return float((m.head(responses.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
   rows=[]
   for (d,slots),obj in mem[1].models.items():
    before=obj.sums.copy(),obj.count.copy()
    for mode in ['cosine','local_centered','spatial_centered']:
     route=obj.routes(q) if mode=='cosine' else decode(obj,q,mode);rows.append({'method':'multicue','dimension':d,'slots':slots,'metric':mode,'accuracy':acc(route),'extra_bytes':obj.size()})
    assert np.array_equal(obj.sums,before[0]) and np.array_equal(obj.count,before[1])
   rows.append({'method':'original_local','accuracy':acc(mem[1].local.routes(q,0)),'extra_bytes':26000});rows.append({'method':'global_ridge','accuracy':acc((q@mem[1].ridge.weight(.1).T).reshape(-1,25,4).argmax(-1)),'extra_bytes':41984});rows.append({'method':'direct_ridge','accuracy':float(((q@direct.weight(1.).T).argmax(-1)==vy.numpy()).mean()),'extra_bytes':direct.size()})
  records.append({'seed':seed,'validation_only':True,'rows':rows});e.save(OUT/'results.json',records);print('RESULT',seed,rows,flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==2,'validation_readouts':24,'official_test_used':False})
if __name__=='__main__':main()
