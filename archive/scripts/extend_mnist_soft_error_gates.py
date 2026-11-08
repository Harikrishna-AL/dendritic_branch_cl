"""Validation-motivated extension to near-hard temperatures; no test-based selection."""
import json,statistics
from pathlib import Path
import numpy as np
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_soft_error_gates import fit,SoftErrorModel,scores

def main():
 p=Path('results/mnist_soft_error_gates_20261105');torch.set_num_threads(2);records=json.loads((p/'results.json').read_text());assert len(records)==36
 raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);vx,vy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels'])
 proto=json.loads((p/'protocol.json').read_text());proto['arms']+=['soft_0.0001','soft_0.00001'];proto['extension_reason']='Validation gate diagnostics showed .1/.01 near-uniform; add two near-hard temperatures. Original validation winner was hard. Extension uses no test comparison to select temperatures.';(p/'protocol.json').write_text(json.dumps(proto,indent=2));(p/'extension_source.py').write_bytes(Path(__file__).read_bytes())
 for seed in [101,102,103]:
  for order in [list(range(5)),list(reversed(range(5)))]:
   for arm in ['soft_0.0001','soft_0.00001']:
    m,routes,weights,r=fit(arm,seed,order,x,y,vx,vy);records.append(r);torch.save(dict(state=m.state_dict(),routes=routes,weights=weights),p/f'{arm}_{seed}_{order[0]}.pt');(p/'results.json').write_text(json.dumps(records,indent=2));print(arm,seed,order[0],{k:round(v['accuracy']*100,2) for k,v in r['stages'][-1].items()},flush=True)
 selected=max(proto['arms'],key=lambda arm:statistics.mean(r['stages'][-1]['magnitude_soft']['accuracy'] for r in records if r['arm']==arm));(p/'selection.json').write_text(json.dumps(dict(arm=selected,metric='magnitude_soft'),indent=2));tests=[]
 for arm in sorted(set(['hard',selected])):
  for r in [r for r in records if r['arm']==arm]:
   m=SoftErrorModel(arm,r['seed']);saved=torch.load(p/f"{arm}_{r['seed']}_{r['order'][0]}.pt",weights_only=True);m.load_state_dict(saved['state']);tests.append(dict(arm=arm,seed=r['seed'],order=r['order'],scores=scores(m,tx,ty,saved['routes'],saved['weights'])))
 (p/'test_results.json').write_text(json.dumps(tests,indent=2));print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
