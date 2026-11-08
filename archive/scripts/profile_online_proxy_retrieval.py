"""Fresh-process deployment costs for selected no-bank online routing methods."""
import json,time,statistics as st,copy
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.data.mnist import load_raw
@torch.no_grad()
def main():
 torch.set_num_threads(2);root=e.OUT;seed=541;name='forward';ck=torch.load(root/'models'/f'{seed}_{name}.pt',weights_only=True);model=e.backbone.make(seed,'branch1_proxy');model.load_state_dict(ck['state_dict']);arrays=np.load(root/'memories'/f'{seed}_{name}.npz');settings=json.loads((root/'selection.json').read_text());mem={s:e.Memory() for s in [1,10]};direct=e.Ridge(10)
 for stable,mm in mem.items():
  for label,obj in [(f'local{d}',v) for d,v in mm.local.items()]+[(k,getattr(mm,k)) for k in ['central96','central144','branch_rls']]:
   for field,v in vars(obj).items():
    if isinstance(v,np.ndarray):setattr(obj,field,arrays[f'stable{stable}_{label}_{field}'].copy())
 direct.cov=arrays['direct_cov'].copy();direct.cross=arrays['direct_cross'].copy();raw=load_raw('data_cache/mnist');x=torch.tensor(raw['test_images'][:128].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['test_labels'][:128].copy(),dtype=torch.long);q=e.keys(x);route=model.routes()[0].numpy().astype(np.uint8);before=e.digest(mem,direct)
 def infer(s,bx):
  method=s['method']
  if method=='ordinary':return model(bx).argmax(-1)
  key=e.keys(bx)
  if method=='direct_class_rls':return (key@direct.weight(s['ridge']).T).argmax(-1)
  mm=mem[s['stable']]
  if method.startswith('local'):chosen=mm.local[s['decay']].routes(key,s['gamma'])
  elif method.startswith('central'):chosen=getattr(mm,method).routes(key)
  else:chosen=(key@mm.branch_rls.weight(s['ridge']).T).reshape(-1,e.N,e.K).argmax(-1)
  response=torch.tanh(model.layers[0](bx)).reshape(-1,e.N,e.K);h=response.gather(-1,torch.tensor(chosen.astype(np.int64))[:,:,None]).squeeze(-1);return model.head(h).argmax(-1)
 rows=[]
 for s in [{'method':'ordinary'}]+settings:
  latency={}
  for batch in [1,128]:
   bx=x[:batch]
   for _ in range(5):infer(s,bx)
   times=[]
   for _ in range(5):
    t=time.perf_counter()
    for _ in range(20):infer(s,bx)
    times.append((time.perf_counter()-t)*1000/20)
   latency[batch]=st.median(times)
  rows.append({**s,'inference_ms':latency})
 assert e.digest(mem,direct)==before
 # Accepted-write costs exclude key generation, backbone gradient step and optional stability rejection.
 for row,s in zip(rows[1:],settings):
  method=s['method']
  if method=='direct_class_rls':obj=copy.deepcopy(direct);target=torch.nn.functional.one_hot(y,10).float().numpy();fn=lambda:obj.update(q,target)
  else:
   mm=mem[s['stable']]
   if method.startswith('local'):obj=copy.deepcopy(mm.local[s['decay']]);fn=lambda:obj.update(q,route)
   elif method.startswith('central'):obj=copy.deepcopy(getattr(mm,method));fn=lambda:obj.update(q,route)
   else:obj=copy.deepcopy(mm.branch_rls);target=np.zeros((128,e.N,e.K),np.float32);target[:,np.arange(e.N),route]=1;target=target.reshape(128,-1);fn=lambda:obj.update(q,target)
  for _ in range(5):fn()
  times=[]
  for _ in range(5):
   t=time.perf_counter()
   for _ in range(20):fn()
   times.append((time.perf_counter()-t)*1000/20)
  row['accepted_write_ms_batch128']=st.median(times)
 e.save(root/'profile.json',{'scope':'Freshprocess,2CPUthreads,selected541forwardstate;5warmups,5blocksof20calls. Inferenceincludeskeygeneration/routing/branchforward/head;RLSsolves64x64systemonEVERYcall,noextraweightcachecounted. Writecostexcludeskeygeneration,NNgradientstep,andstabilitygate;usesrepeatedacceptedbatchonclonedtrainedmemory. CPUonly,microbenchmarknotfullstreamoverhead. Localplainwriteimplementationupdatesringlinksbutdeployableplainmethodcanomitthose;itswritecostisaconservativeimplementationmeasurement.','methods':rows})
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
