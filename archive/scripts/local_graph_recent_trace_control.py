"""Paired cumulative-error control for the recent branch error trace."""
import json,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_mechanism_factorial import Mechanism,evaluate_all
from bdh_cl.scripts.local_graph_resource_benchmark import permutations,dump

class Cumulative(Mechanism):
 @torch.no_grad()
 def learn(self,x,y):
  old=self.error.clone();seen=self.presentations;super().learn(x,y)
  alpha=len(x)/(seen+len(x));self.error.mul_(alpha).add_(old,alpha=1-alpha)

def main():
 out=Path('results/local_graph_recent_trace_control_20261013');out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);rows=[]
 dump(out/'protocol.json',{'question':'Does previous-minibatch error memory help beyond cumulative historical error memory?','comparison':'Same groupednovelty/protection/coactivity model, weights/init/order andstoredarrays; onlyerrortrace changes. Enew=Eold*(1-B/(past+B))+currentbatchMSE*B/(past+B). Currentbatcherrors measured beforeupdate exactlyasrecentbaseline. No historicalinputs revisited, labels/taskIDs not supplied atinference.','seeds':{'mnist':[391,392,393],'fashion_mnist':[401,402,403]},'domains':16,'protection':['absolute','off'],'training':'Two currentdomainpasses; same primaryfactorialcohort. Followup paired, not newindependentseeds. Settingsfixed before scoring cumulativecontrol. Existingmodelpresentationcounter used, no extra arrays. Full10ktest/domain; communicating/nocomm samefinalweights.'})
 (out/'source.py').write_text(Path(__file__).read_text())
 # A single first batch must match the recent model exactly.
 s={'rate':1.,'tau':10000.,'gamma':.1,'gated':True,'loss':'mse','protection':'absolute','allocation':'novelty','threshold':.4,'competition':'grouped'}
 torch.manual_seed(66);x=torch.rand(13,784);y=torch.arange(13)%10;a=Mechanism(7,s);b=Cumulative(7,s);a.learn(x,y);b.learn(x,y);assert all(torch.equal(getattr(a,k),getattr(b,k)) for k in ['weight','codes','error','usage','links'])
 for ds,seeds in [('mnist',[391,392,393]),('fashion_mnist',[401,402,403])]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for seed in seeds:
   ps=permutations(seed,16)
   for protection in ['absolute','off']:
    s={**s,'protection':protection};m=Cumulative(seed,s)
    for domain,p in enumerate(ps):
     for epoch in range(2):
      rng=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
      for ix in torch.randperm(len(y),generator=rng).split(128):m.learn(x[ix][:,p],y[ix])
    result=evaluate_all(m,tx,ty,ps,['communicating','grouped_no_communication'])
    for mode,a in result.items():
     r={'dataset':ds,'seed':seed,'trace':'cumulative','protection':protection,'readout':mode,'accuracy':st.mean(a),'old_accuracy':st.mean(a[:-1]),'first_four':st.mean(a[:4]),'newest':a[-1],'domain_accuracy':a,'numeric_bytes':m.bytes()};rows.append(r);dump(out/'results.json',rows);print(ds,seed,protection,mode,'all',round(r['accuracy']*100,2),'first4',round(r['first_four']*100,2),'new',round(r['newest']*100,2),flush=True)
 (out/'verification.json').write_text(json.dumps({'first_batch_exactly_matches_recent_trace_model':True,'all_numeric_state_812480bytes':all(r['numeric_bytes']==812480 for r in rows)},indent=2))
if __name__=='__main__':main()
