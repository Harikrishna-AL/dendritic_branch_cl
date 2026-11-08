"""Independent backbone reproduction, key-drift and read-only-memory audits."""
import json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make,train_epoch
from bdh_cl.scripts.online_config_memory_benchmark import OUT,KEYS,Memory,keys,score,save

def memory_state(m):
 return {name+'.'+k:v.copy() for name in ['local','central96','central144','rls'] for k,v in vars(getattr(m,name)).items() if isinstance(v,np.ndarray)}
def main():
 torch.set_num_threads(2)
 assert json.loads((OUT/'completion.json').read_text())['complete']
 # Every inference method must leave all persistent arrays unchanged.
 rng=np.random.default_rng(50);q=rng.normal(size=(11,64)).astype(np.float32);q/=np.linalg.norm(q,axis=-1,keepdims=True);bank=rng.integers(0,4,(5,24),dtype=np.uint8);mem=Memory()
 for route in bank:mem.update(q,route)
 before=memory_state(mem)
 from bdh_cl.scripts.online_config_memory_benchmark import setting_list
 for s in setting_list():score(mem,q,bank,s)
 assert all(np.array_equal(v,memory_state(mem)[k]) for k,v in before.items())
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);rows=json.loads((OUT/'key_drift_audit.json').read_text()) if (OUT/'key_drift_audit.json').exists() else []
 for seed in [511,512,513]:
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   if any(a['seed']==seed and a['order']==name for a in rows):continue
   torch.manual_seed(seed);m=make('local','hard');initial=[m.first.weight.detach().clone(),m.first.bias.detach().clone()];opt=torch.optim.Adam(m.parameters(),lr=.001);saved={k:{} for k in KEYS};acq=[]
   for stage,pair in enumerate(order):
    ix=torch.nonzero(torch.isin(y,torch.tensor(pair)),as_tuple=True)[0]
    for e in range(2):train_epoch(m,opt,x,y,ix,seed*100000+stage*100+e,128)
    chosen=torch.isin(ty,torch.tensor(pair));bx=tx[chosen];by=ty[chosen]
    with torch.no_grad():
     pred=torch.cat([m(b) for b in bx.split(512)]).argmax(-1);acq.append(float((pred==by).float().mean()));q=keys(bx,m,initial)
    for k in KEYS:
     for c in pair:saved[k][c]=q[k][by.numpy()==c].mean(0)
   original=torch.load(OUT/'models'/f'{seed}_{name}.pt',weights_only=True)['state'];assert all(torch.equal(v,original[k]) for k,v in m.state_dict().items())
   allq={k:[] for k in KEYS}
   for bx in tx.split(512):
    current=keys(bx,m,initial)
    for k in KEYS:allq[k].append(current[k])
   drift={}
   for k in KEYS:
    q=np.concatenate(allq[k]);drift[k]={str(c):float(np.mean((q[ty.numpy()==c].mean(0)-saved[k][c])**2)) for c in range(10)}
   rows.append({'seed':seed,'order':name,'pairs':order,'acquisition':acq,'classmean_key_drift':drift,'backbone_matches_original_train_epoch_exactly':True});save(OUT/'key_drift_audit.json',rows);print(seed,name,'originaltrainingexact','evolvingoldkeydrift',np.mean([v for c,v in drift['evolving_responses'].items() if int(c) not in order[-1]]),flush=True)
 save(OUT/'audit_verification.json',{'all_6_backbones_exactly_match_independent_original_training':len(rows)==6,'all_memory_inference_rules_readonly_synthetic_check':True,'test_key_means_diagnostic_only':True})
 (OUT/'audit_source.py').write_text(Path(__file__).read_text())
if __name__=='__main__':main()
