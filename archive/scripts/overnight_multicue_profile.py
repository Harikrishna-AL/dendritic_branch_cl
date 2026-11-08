"""Selected-state, fresh-process CPU costs; no accuracy selection."""
import copy,json,time,statistics as st
import numpy as np
import torch
from bdh_cl.scripts import overnight_multicue_confirmation as c
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi,Direct
from bdh_cl.data.mnist import load_raw
@torch.no_grad()
def main():
 torch.set_num_threads(2);root=c.OUT;settings=json.loads((root/'selection.json').read_text());a=np.load(root/'memories/562_forward.npz');model=e.backbone.make(562,'branch1_proxy');model.load_state_dict(torch.load(root/'models/562_forward.pt',weights_only=True)['state_dict']);raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'][:128].copy(),dtype=torch.float32).flatten(1)/255;q=e.keys(x);route=model.routes()[0].numpy();cases=[]
 for s in settings:
  d,slots,stable=s['dimension'],s['slots'],s['stable'];obj=Multi(d,slots,s['decay']);obj.sums=a[f'stable{stable}_d{d}s{slots}_sums'].copy();obj.count=a[f'stable{stable}_d{d}s{slots}_count'].copy();cases.append(({'method':'multicue',**s},obj,lambda q,obj=obj:obj.routes(q)))
 obj=e.Local(.995)
 for field in ['sums','count','links']:setattr(obj,field,a[f'stable1_original_local_{field}'].copy())
 cases.append(({'method':'original_local'},obj,lambda q,obj=obj:obj.routes(q,0)))
 obj=e.Ridge(100);obj.cov=a['stable1_ridge_cov'].copy();obj.cross=a['stable1_ridge_cross'].copy();cases.append(({'method':'global_ridge'},obj,lambda q,obj=obj:(q@obj.weight(.1).T).reshape(-1,25,4).argmax(-1)))
 for d in [16,32,64]:
  obj=Direct(d);obj.cov=a[f'direct{d}_cov'].copy();obj.cross=a[f'direct{d}_cross'].copy();cases.append(({'method':'direct_ridge','dimension':d},obj,lambda q,obj=obj:obj.predict(q)))
 def bench(fn):
  for _ in range(3):fn()
  blocks=[]
  for _ in range(5):
   start=time.perf_counter()
   for _ in range(20):fn()
   blocks.append((time.perf_counter()-start)*1000/20)
  return st.median(blocks)
 rows=[]
 for setting,obj,decode in cases:
  row={**setting,'inference_ms':{}}
  for batch in [1,128]:
   bx=x[:batch]
   def fn():
    key=e.keys(bx);chosen=decode(key)
    if setting['method']=='direct_ridge':return chosen
    b=torch.tanh(model.layers[0](bx)).reshape(-1,25,4);h=b.gather(-1,torch.tensor(chosen.astype(np.int64))[:,:,None]).squeeze(-1);return model.head(h).argmax(-1)
   row['inference_ms'][batch]=bench(fn)
  clone=copy.deepcopy(obj)
  if setting['method']=='global_ridge':
   target=np.zeros((128,25,4),np.float32);target[:,np.arange(25),route]=1;fn=lambda:clone.update(q,target.reshape(-1,100))
  elif setting['method']=='direct_ridge':fn=lambda:clone.update(q,torch.arange(128)%10)
  else:fn=lambda:clone.update(q,route)
  row['accepted_write_ms_batch128']=bench(fn);rows.append(row)
 e.save(root/'profile.json',{'scope':'Freshprocess,2CPUthreads,562forwardstate;3warmups,5x20calls. Inferenceincludeskeycreation/compression,routing,neuralheadwhereused;ridgeweights solvedeachcall(noadditionalcache). AcceptedwritesexcludeoriginalNNgradientstep/keygeneration,clonedmemoryrepeatedbatch;directlabelsconstructedinthecall. CPUmicrobenchmarks,notend-to-endtrainingtime. All-candidateexperimentruntimeisnotdeployablecost.','rows':rows});print('profile_complete',len(rows))
if __name__=='__main__':main()
