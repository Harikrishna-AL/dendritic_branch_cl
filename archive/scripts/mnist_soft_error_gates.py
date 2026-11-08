"""Soft error-trace training with fixed sparse MLP and magnitude-soft inference."""
import json,time,statistics,math,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model,evaluate

ARMS=['hard','soft_0.1','soft_0.01','soft_0.001','anneal','uniform']
class SoftErrorModel(Model):
 def __init__(self,arm,seed):
  super().__init__('baseline',seed);self.training_arm=arm;self.temperature=.1
 def gates(self):
  if self.training_arm=='hard':return F.one_hot(self.error.argmin(-1),4).to(self.error.dtype)
  if self.training_arm=='uniform':return torch.full_like(self.error,.25)
  return (-self.error/self.temperature).softmax(-1)

def set_temperature(m,epoch,step,total):
 if m.training_arm=='anneal':m.temperature=.1*(.001/.1)**((epoch*total+step)/max(2*total-1,1))
 elif m.training_arm.startswith('soft_'):m.temperature=float(m.training_arm.split('_')[1])

@torch.no_grad()
def scores(m,x,y,routes,weights):
 out=evaluate(m,x,y,routes);hits=torch.zeros(5);counts=torch.bincount(y//2,minlength=5)
 for bx,by in zip(x.split(256),y.split(256)):
  z=m.responses(bx,False).tanh();pred=m.head((z*weights[by//2]).sum(-1)).argmax(-1);hits+=torch.bincount((by//2)[pred==by],minlength=5)
 out['oracle_mixture']=dict(accuracy=float(hits.sum()/len(y)),pairs=(hits/counts).tolist());return out

def fit(arm,seed,order,x,y,vx,vy):
 torch.manual_seed(seed);m=SoftErrorModel(arm,seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);routes=torch.zeros(5,168,dtype=torch.uint8);weights=torch.zeros(5,168,4);stages=[];gate_stats=[];seconds=0
 for stage,pair in enumerate(order):
  ix=torch.where(y//2==pair)[0];total=math.ceil(len(ix)/128);entropy=maximum=0.;steps=0
  start=time.perf_counter()
  for epoch in range(2):
   ordered=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(seed*100000+stage*100+epoch))]
   for step,ids in enumerate(ordered.split(128)):
    set_temperature(m,epoch,step,total);opt.zero_grad(set_to_none=True);logits,alt=m(x[ids],counterfactual=True);target=y[ids,None,None].expand(alt.shape[:-1]);err=F.cross_entropy(alt.reshape(-1,10),target.reshape(-1),reduction='none').reshape(alt.shape[:-1]);loss=F.cross_entropy(logits,y[ids])+.5*err.mean()
    if not torch.isfinite(loss):raise ValueError('nonfinite')
    with torch.no_grad():
     gate=m.gates();entropy+=float(-(gate*gate.clamp_min(1e-30).log()).sum(-1).mean()/math.log(4));maximum+=float(gate.max(-1).values.mean());steps+=1
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(err.detach().mean(0))
  seconds+=time.perf_counter()-start;routes[pair]=m.error.argmin(-1).to(torch.uint8);weights[pair]=m.gates();gate_stats.append(dict(pair=pair,normalized_entropy=entropy/steps,mean_max_weight=maximum/steps,end_temperature=m.temperature));stages.append(scores(m,vx,vy,routes,weights))
 return m,routes,weights,dict(arm=arm,seed=seed,order=order,stages=stages,gate_stats=gate_stats,training_seconds=seconds,parameters=sum(p.numel() for p in m.parameters()),model_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_soft_error_gates_20261105'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels'])
 protocol=dict(arms=ARMS,seeds=[101,102,103],orders=['forward','reverse'],neurons=168,branches=4,inputs_per_branch=196,epochs=2,learning_rate=.0005,batch_size=128,trace_decay=.95,loss='Trace-mixture CE + .5 mean single-neuron branch-substitution CE; gates detached, EMA updated after optimizer step',annealing='Exponential .1 to .001 over two epochs, restarted at each known task boundary; EMA and Adam never reset',inference='Fixed softmax(abs(raw response)), weighted tanh branches; primary hard-magnitude also reported',oracle='Both one saved branch/neuron and saved soft mixtures, requiring true task identity',selection='Select training arm by average validation magnitude_soft accuracy, then test selected arm and hard baseline only',limits='Shared fixed recipe, no replay/prototypes/pretraining or task identity at inference; known boundaries during training; no claim of task-free training')
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes());records=[]
 for seed in [101,102,103]:
  for order in [list(range(5)),list(reversed(range(5)))]:
   for arm in ARMS:
    m,routes,weights,r=fit(arm,seed,order,x,y,vx,vy);records.append(r);torch.save(dict(state=m.state_dict(),routes=routes,weights=weights),a.out/f'{arm}_{seed}_{order[0]}.pt');(a.out/'results.json').write_text(json.dumps(records,indent=2));print(arm,seed,order[0],{k:round(v['accuracy']*100,2) for k,v in r['stages'][-1].items()},flush=True)
 selected=max(ARMS,key=lambda arm:statistics.mean(r['stages'][-1]['magnitude_soft']['accuracy'] for r in records if r['arm']==arm));(a.out/'selection.json').write_text(json.dumps(dict(arm=selected,metric='magnitude_soft'),indent=2));tests=[]
 for arm in sorted(set(['hard',selected])):
  for r in [r for r in records if r['arm']==arm]:
   m=SoftErrorModel(arm,r['seed']);saved=torch.load(a.out/f"{arm}_{r['seed']}_{r['order'][0]}.pt",weights_only=True);m.load_state_dict(saved['state']);tests.append(dict(arm=arm,seed=r['seed'],order=r['order'],scores=scores(m,tx,ty,saved['routes'],saved['weights'])))
 (a.out/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
