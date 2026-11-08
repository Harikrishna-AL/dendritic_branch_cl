"""Offline diagnostic only: rebuild all class means using old labeled training data."""
import json
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.sequential_prototype_representation import Sequential,save
OUT=Path('results/sequential_prototype_representation_20261013')
def main():
 torch.set_num_threads(2);assert json.loads((OUT/'completion.json').read_text())['complete'];rows=json.loads((OUT/'results.json').read_text());diagnostic=[]
 for ds in ['mnist','fashion_mnist']:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for r in [a for a in rows if a['dataset']==ds]:
   state=torch.load(OUT/'models'/f'{ds}_{r["seed"]}_{r["order"]}_{r["arm"]}.pt',weights_only=True);m=Sequential(r['seed'],r['arm'],r['lr'],r['temperature'])
   with torch.no_grad():m.weight.copy_(state['weight']);m.prototype.copy_(state['prototype']);m.count.copy_(state['count'])
   original,_=m.measure(tx,ty,r['pairs'],5);assert abs(original['accuracy']-r['accuracy'])<1e-9;w=m.weight.detach().clone();old=m.prototype.clone();m.refit(x,y);assert torch.equal(w,m.weight);rebuilt,_=m.measure(tx,ty,r['pairs'],5)
   if r['arm']=='frozen':assert abs(rebuilt['accuracy']-original['accuracy'])<.001
   diagnostic.append({'dataset':ds,'seed':r['seed'],'order':r['order'],'arm':r['arm'],'original':original,'offline_allclass_refit':rebuilt,'prototype_shift_mse':float((m.prototype-old).square().mean())});save(OUT/'refit_diagnostic.json',diagnostic);print(ds,r['seed'],r['order'],r['arm'],'refit',round(rebuilt['accuracy']*100,2),flush=True)
 save(OUT/'refit_protocol.json',{'only_diagnostic':True,'method':'Final networkweightsunchanged;rebuildalltenclassmeansusingall60klabeledtrainingimages,includingoldclasses. No gradientupdates. Evaluatealltenclasseswithouttestlabelrouting. OlddatausedinthisanalysisthereforeNOTvalidreplay-freeCLperformance. Primarylearnerresultsunchanged.', 'checks':'Savedprimarypredictionreproduced;inputweightsunchangedbyrefit;frozenaccuracyconsistentwithin0.1pp;48diagnosticmodels.'})
 (OUT/'refit_source.py').write_text(Path(__file__).read_text())
 assert len(diagnostic)==48
if __name__=='__main__':main()
