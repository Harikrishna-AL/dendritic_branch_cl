"""Inference cost of the label-free branch state, and a Taylor (no-label) shortcut.
The pseudo-error target is the model's OWN prediction, so the first-order trick
g_i = sum_c (p_c - onehot(pred)_c) W_ci works without true labels."""
import time,statistics,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_short_term_context_test import features
import bdh_cl.scripts.three_routes_temporal as T
torch.set_num_threads(2);raw=load_raw('data_cache/mnist')
tx=(torch.tensor(raw['test_images'].copy()).float().reshape(-1,784)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
def std_pref(err):c=err-err.mean(-1,keepdim=True);return (-c/c.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).softmax(-1)
@torch.no_grad()
def step(m,x,mode):
    r=m.responses(x,False);z=r.tanh();h=(z*r.abs().softmax(-1)).sum(-1);lg=m.head(h)
    if mode=='none':return lg
    pred=lg.argmax(-1);W=m.head.weight
    if mode=='exact':
        alt=lg[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*W.T[None,:,None,:]
        err=F.cross_entropy(alt.reshape(-1,10),pred[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none').reshape(alt.shape[:-1])
    else:
        g=(lg.softmax(-1)-F.one_hot(pred,10).float())@W;err=g[:,:,None]*(z-h[:,:,None])
    return std_pref(err)
out={}
for s in [101,102,103]:
  for f in [0,4]:
    m=Model('baseline',s);m.load_state_dict(torch.load(f'results/mnist_gradient_branch_scoring_20261106/hybrid5_{s}_{f}.pt',weights_only=True)['state']);m.requires_grad_(False)
    c=features(m,tx,ty);W=m.head.weight.numpy();b=m.head.bias.numpy();c['logits']=(c['z']*c['current']).sum(-1)@W.T+b;c['W']=W;c['b']=b;c['y']=ty.numpy()
    exact=c['pseudo'];tay=torch.cat([step(m,bx,'taylor') for bx in tx.split(1000)]).numpy()
    agree=float((exact.argmax(-1)==tay.argmax(-1)).mean())
    acc={}
    for name,pref in [('exact',exact),('taylor',tay)]:
        c['pseudo']=pref
        for k in T.STREAMS:o=T.order_of(c['y'],s+f,k);acc[(name,k)]=float((T.run_pseudo(c,o,.9,.9,.3)==c['y'][o]).mean())
    out[(s,f)]=dict(agree=agree,acc=acc);print(s,f,'best-branch agreement',round(agree*100,1),{f'{a}/{k}':round(v*100,2) for (a,k),v in acc.items()},flush=True)
print('\nmean best-branch agreement exact vs taylor:',round(statistics.mean(v['agree'] for v in out.values())*100,1),'%')
for name in ['exact','taylor']:
    print(name,{k:round(statistics.mean(v['acc'][(name,k)] for v in out.values())*100,2) for k in T.STREAMS})
# timing: one image at a time (streaming), median of 3 repeats x 2000 images
x1=tx[:2000]
for mode in ['none','exact','taylor']:
    reps=[]
    for _ in range(3):
        for i in range(50):step(m,x1[i:i+1],mode)
        t=time.perf_counter()
        for i in range(2000):step(m,x1[i:i+1],mode)
        reps.append((time.perf_counter()-t)/2000*1e6)
    print(f'{mode:7s} per-image latency {statistics.median(reps):7.1f} us')
