"""Clean step-time benchmark; exact reference does not compute Taylor diagnostics."""
import json,time,statistics
from pathlib import Path
import torch
import numpy as np
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
from bdh_cl.scripts.mnist_sparse_aux_ablation import objective as clean_exact

def main():
 torch.set_num_threads(2);out=Path('results/mnist_gradient_branch_scoring_20261106');raw=load_raw('data_cache/mnist');labels=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);ids=torch.where(labels//2==4)[0][:128];x=(torch.tensor(raw['train_images'][ids.numpy()].copy()).float().reshape(-1,784)/255-.13062755)/.30810780;y=labels[ids]
 saved=torch.load(out/'exact_101_0.pt',weights_only=True);records=[]
 for repeat in range(3):
  arms=['exact','linear','quadratic','hybrid5','hybrid10'];arms=arms[repeat:]+arms[:repeat]
  for arm in arms:
   torch.manual_seed(101);m=Model('baseline',101);m.load_state_dict(saved['state']);opt=torch.optim.Adam(m.parameters(),lr=.0005)
   def step(i):
    opt.zero_grad(set_to_none=True);is_exact=arm=='exact' or (arm.startswith('hybrid') and i%int(arm[6:])==0)
    if is_exact:loss,feedback=clean_exact(m,x,y,.5)
    else:loss,feedback,_=objective(m,x,y,'quadratic' if arm=='quadratic' else 'linear',i)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback)
   for i in range(10):step(i)
   tick=time.perf_counter()
   for i in range(100):step(i)
   seconds=time.perf_counter()-tick;records.append(dict(arm=arm,repeat=repeat,milliseconds_per_step=seconds*10))
 (out/'clean_benchmark.json').write_text(json.dumps(records,indent=2));print({arm:statistics.median(r['milliseconds_per_step'] for r in records if r['arm']==arm) for arm in arms})
if __name__=='__main__':main()
