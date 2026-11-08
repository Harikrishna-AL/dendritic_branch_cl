import json,statistics,itertools,sys
from pathlib import Path
import numpy as np,torch
import bdh_cl.scripts.three_routes_temporal as T
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_short_term_context_test import features

def run_prior_gated(c,order,decay,strength,c0,gdecay=.9):
    st=None;pred=np.empty(len(order),int);per=None;prev=None
    for i,ix in enumerate(order):
        lg=c['logits'][ix];pair=int(lg.argmax())//2
        if prev is not None: per=(1. if pair==prev else 0.) if per is None else gdecay*per+(1-gdecay)*(pair==prev)
        prev=pair;s=strength*(0. if per is None else float(np.clip((per-c0)/(1-c0),0,1)))
        adj=lg if st is None else lg+s*np.log(st+1e-4);pred[i]=adj.argmax();p=T.sm(adj);st=p if st is None else decay*st+(1-decay)*p
    return pred
torch.set_num_threads(2);raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
data={'val':ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]),'test':ds(raw['test_images'],raw['test_labels'])};C={}
for s in [101,102,103]:
  for f in [0,4]:
    m=Model('baseline',s);m.load_state_dict(torch.load(f'results/mnist_gradient_branch_scoring_20261106/hybrid5_{s}_{f}.pt',weights_only=True)['state']);W=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy()
    for k,(x,y) in data.items():
      with torch.no_grad(): raw_=m.responses(x,False);lg=((raw_.tanh()*raw_.abs().softmax(-1)).sum(-1)).numpy()@W.T+b
      C[(s,f,k)]=dict(logits=lg,y=y.numpy())
def ev(split,cfg):
  out={k:[] for k in T.STREAMS}
  for s in [101,102,103]:
    per={k:[] for k in T.STREAMS}
    for f in [0,4]:
      c=C[(s,f,split)]
      for k in T.STREAMS:o=T.order_of(c['y'],s+f,k);per[k].append(float((run_prior_gated(c,o,**cfg)==c['y'][o]).mean()))
    for k in T.STREAMS:out[k].append(statistics.mean(per[k])*100)
  return out
grid=[dict(decay=d,strength=st,c0=c0) for d in [.8,.9,.95] for st in [.3,.5,.7,1.,1.5] for c0 in [.3,.5,.7]]
best=max(grid,key=lambda g:statistics.mean(statistics.mean(v) for v in ev('val',g).values()))
t=ev('test',best);print('prior_gated',best,{k:(round(statistics.mean(v),2),round(statistics.stdev(v),2)) for k,v in t.items()})
json.dump(dict(cfg=best,test=t),open('results/three_routes_20261106/temporal/prior_gated_test.json','w'),indent=2)
r=json.load(open('results/three_routes_20261106/temporal/test_results.json'))
def ci(a,b):
  d=[x-y for x,y in zip(a,b)];m=statistics.mean(d);h=4.303*statistics.stdev(d)/3**.5;return f'{m:+.2f} [{m-h:+.2f},{m+h:+.2f}]'
for k in T.STREAMS:
  print(k,'pseudo_gated - none',ci(r['pseudo_gated']['test'][k],r['none']['test'][k]),' pseudo_gated - prior_gated',ci(r['pseudo_gated']['test'][k],t[k]),' task_hmm_mix - none',ci(r['task_hmm_mix']['test'][k],r['none']['test'][k]))
