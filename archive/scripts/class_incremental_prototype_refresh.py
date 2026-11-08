"""Current-stage prototype fitting control; no old-class refresh or replay."""
import json,time
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.class_incremental_branch_prototypes import Proto,evaluate,save
OUT=Path('results/class_incremental_prototype_refresh_20261013')

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);rows=[]
 source=Path('results/class_incremental_branch_prototypes_20261013');selection=json.loads((source/'selection.json').read_text())
 save(OUT/'protocol.json',{'question':'Are poor online-prototype results due to averaging features while they change during acquisition?', 'method':'Retrain identical protected models using fixed original development settings. After two gradient passes on eachcurrentpair, recompute only the currentclassprototypes in one additional nonlearning pass over currenttrainimages. Oldclassprototypes neverrevisited. Explicit currentstageboundary for timing this extra pass; no taskID input to learner or inference. Allseenclasses compete. No oldexamples stored/replayed. This is a paired followup, no independent replication or retuning.', 'primary':'Identical full256-dimensional squaredEuclideanprototypeclassifier forbranchesanddense. Branchgrouped/communication secondary. Same statebudgets asoriginal. Extra prototype-fitting compute and currentdata pass reported.', 'training':'Same currentpairdata/order/weights as original. Prototype tensors never used by local feature learning; refresh cannot alter weights/error/usage/coactivity.'})
 (OUT/'source.py').write_text(Path(__file__).read_text());forward=[[0,1],[2,3],[4,5],[6,7],[8,9]]
 for ds,dev in [('mnist',421),('fashion_mnist',431)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for seed in range(dev+1,dev+4):
   for order_name,order in [('forward',forward),('reverse',forward[::-1])]:
    for comp in ['grouped','dense']:
     s=selection[ds][comp]['chosen'];m=Proto(seed,comp,s['rate'],s['tau']);history=[];seconds=0.;fitseconds=0.;savedmeans={};modes=['flat','grouped','communicating'] if comp=='grouped' else ['flat']
     for stage,pair in enumerate(order):
      oldclasses=(m.count>0).nonzero().flatten();old=m.prototype[oldclasses].clone();ids=torch.isin(y,torch.tensor(pair)).nonzero().flatten();t=time.perf_counter()
      for epoch in range(2):
       rng=torch.Generator().manual_seed(seed*10000+stage*100+epoch)
       for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):m.learn(x[ix],y[ix])
      seconds+=time.perf_counter()-t;w=m.model.weight.clone();t=time.perf_counter();sums=torch.zeros(10,256);counts=torch.zeros(10,dtype=torch.int64)
      with torch.no_grad():
       for ix in ids.split(256):
        h=m.response(x[ix]);by=y[ix]
        for c in by.unique():sums[c]+=h[by==c].sum(0);counts[c]+=(by==c).sum()
       for c in pair:m.prototype[c]=sums[c]/counts[c];m.count[c]=counts[c]
      fitseconds+=time.perf_counter()-t;assert torch.equal(w,m.model.weight);assert torch.equal(old,m.prototype[oldclasses])
      result,means=evaluate(m,tx,ty,modes,order,stage+1)
      for c in pair:savedmeans[c]=means[c].clone()
      history.append({'stage':stage+1,'readouts':result,'test_class_mean_drift':{str(c):float((means[c]-v).square().mean()) for c,v in savedmeans.items()},'training_seconds':seconds,'prototype_fitting_seconds':fitseconds})
     row={'dataset':ds,'seed':seed,'order':order_name,'arm':'branch_protected' if comp=='grouped' else 'dense_protected','history':history,'numeric_bytes':m.bytes(),'rate':s['rate'],'tau':s['tau']};rows.append(row);save(OUT/'results.json',rows)
     print(ds,seed,order_name,comp,{mode:round(v['accuracy']*100,2) for mode,v in history[-1]['readouts'].items()},flush=True)
 save(OUT/'verification.json',{'prototype_fit_does_not_change_weights':True,'old_prototypes_unchanged_each_stage':True,'complete_24_models':len(rows)==24})
if __name__=='__main__':main()
