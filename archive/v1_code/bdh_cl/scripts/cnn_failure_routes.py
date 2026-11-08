"""Route distinctness and within-pair oracle for the CIFAR branch arms (diagnostic)."""
import itertools,json,statistics as st
import numpy as np,torch
from bdh_cl.data import cifar10
from bdh_cl.scripts.cnn_failure_diagnosis import run
torch.set_num_threads(6)
d=cifar10.load('data_cache/cifar10')
data=[torch.from_numpy(np.array(v,copy=True)) for v in [d['images'][d['train']],d['labels'][d['train']],d['images'][d['validation']],d['labels'][d['validation']],d['test_images'],d['test_labels']]]
data=[v.long() if i%2 else v for i,v in enumerate(data)];tx,ty=data[4],data[5];out={}
for arm in ['cnn_learned','cnn_random','pixel_sparse']:
  for seed in [201,202]:
    m,routes,r=run(arm,seed,list(range(5)),data);m.eval()
    dis=st.mean(float((routes[i]!=routes[j]).float().mean()) for i,j in itertools.combinations(range(5),2))
    with torch.no_grad():
      L=torch.cat([m.logits(tx[i:i+500],'oracle',routes[ty[i:i+500]//2]) for i in range(0,len(ty),500)])
    pair=L.gather(1,torch.stack([2*(ty//2),2*(ty//2)+1],1)).argmax(1)+2*(ty//2)
    full=float((L.argmax(1)==ty).float().mean());within=float((pair==ty).float().mean())
    out[f'{arm}_{seed}']=dict(route_disagreement=dis,oracle_all10=full,oracle_within_pair=within,acq=r['acquisition'])
    print(arm,seed,'route disagreement',round(dis*100,1),'% | oracle all-10',round(full*100,1),'| oracle within true pair',round(within*100,1),'| mean acquisition',round(st.mean(r['acquisition'])*100,1),flush=True)
json.dump(out,open('results/cnn_failure_diagnosis_20261106/route_diagnostics.json','w'),indent=2)
