"""Diagnostic joint-training reference for the preceding local fixed-code learner."""
import json,time
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.class_incremental_branch_prototypes import Proto,save
OUT=Path('results/joint_fixed_code_reference_20261013')
@torch.no_grad()
def refit(m,x,y):
 sums=torch.zeros(10,256);counts=torch.zeros(10,dtype=torch.int64)
 for start in range(0,len(y),256):
  by=y[start:start+256];h=m.response(x[start:start+256]);sums.index_add_(0,by,h);counts+=torch.bincount(by,minlength=10)
 m.prototype.copy_(sums/counts[:,None]);m.count.copy_(counts)
@torch.no_grad()
def accuracy(m,x,y):return float((torch.cat([m.scores(b,'flat') for b in x.split(256)])==y).float().mean())
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);rows=[];selection=json.loads(Path('results/class_incremental_branch_prototypes_20261013/selection.json').read_text())
 save(OUT/'protocol.json',{'purpose':'Diagnostic: does previous fixed-code locallearner also learn goodfeatures when allclassesavailablejointly? Notpurelossablation: groupedselectionandusageprotection differfrom gradientnormprototype learner. No independentjointretuning; use earlierclassincrementalvalidationsettings.', 'data':'Samefresh442-444MNIST452-454Fashioninitialinputweights,minibatchorder,and60ktrain/full10ktest asjointprototype study;alltenclassesmixed;10epochs. Sameflat256responseprototype classifier. Refitclassmeansfromtrainingaftereach epoch. No testtraining.', 'settings_source':'class_incremental_branch_prototypes_20261013/selection.json','comparison_limit':'Previouspoorclassincrementalaccuracyversusnewjointaccuracycannotisolateobjectivechangesfromabsenceofforgetting. Thisreference helpscheckthat distinction.'})
 (OUT/'source.py').write_text(Path(__file__).read_text())
 for ds,start in [('mnist',442),('fashion_mnist',452)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for seed in range(start,start+3):
   for comp in ['grouped','dense']:
    s=selection[ds][comp]['chosen'];m=Proto(seed,comp,s['rate'],s['tau']);refit(m,x,y);h=[{'epoch':0,'accuracy':accuracy(m,tx,ty)}];t=time.perf_counter()
    for epoch in range(10):
     rng=torch.Generator().manual_seed(seed*1000+epoch)
     for ix in torch.randperm(len(y),generator=rng).split(128):m.learn(x[ix],y[ix])
     refit(m,x,y)
     if epoch+1 in [2,5,10]:h.append({'epoch':epoch+1,'accuracy':accuracy(m,tx,ty)})
    rows.append({'dataset':ds,'seed':seed,'arm':comp,'history':h,'accuracy':h[-1]['accuracy'],'numeric_bytes':m.bytes(),'seconds':time.perf_counter()-t,'rate':s['rate'],'tau':s['tau']});save(OUT/'results.json',rows);print(ds,seed,comp,round(h[-1]['accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==12,'models':len(rows)})
if __name__=='__main__':main()
