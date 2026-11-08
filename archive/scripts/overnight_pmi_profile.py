import json,time,statistics as st
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi
from bdh_cl.scripts.overnight_pmi_message_pilot import decode
from bdh_cl.scripts.overnight_pmi_confirmation import OUT
from bdh_cl.data.mnist import load_raw
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=np.load(OUT/'memories/602_forward.npz');m=e.backbone.make(602,'branch1_proxy');m.load_state_dict(torch.load(OUT/'models/602_forward.pt',weights_only=True)['state_dict']);raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'][:128].copy(),dtype=torch.float32).flatten(1)/255;rows=[]
 for d in [16,64]:
  obj=Multi(d,4,.995 if d==16 else 1.);obj.sums=a[f'stable1_d{d}_sums'].copy();obj.count=a[f'stable1_d{d}_count'].copy()
  for method,gamma,scramble in [('no_message',0.,False),('learned_pmi',.01,False),('shuffled_pmi',.01,True)]:
   r={'method':method,'dimension':d,'inference_ms':{}}
   for batch in [1,128]:
    bx=x[:batch]
    def fn():
     q=e.keys(bx);route=decode(obj,q,a['stable1_cooc'],gamma,scramble);b=torch.tanh(m.layers[0](bx)).reshape(-1,25,4);return m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)
    for _ in range(3):fn()
    blocks=[]
    for _ in range(5):
     t=time.perf_counter()
     for _ in range(20):fn()
     blocks.append((time.perf_counter()-t)*1000/20)
    r['inference_ms'][batch]=st.median(blocks)
   rows.append(r)
 e.save(OUT/'profile.json',{'scope':'Freshprocess,2CPUthreads,602forward,train-imagebatch;3warmups,5x20calls. Fullkey/routing/branch/headinference;PMIcomputedfromcountseachcall,3rounds,graph40edges. No weightcache/latencyoptimization. CPUmicrobenchmark,nottrainingtime.','rows':rows})
if __name__=='__main__':main()
