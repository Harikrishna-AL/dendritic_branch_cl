"""Sequential transfer of the fixed joint-selected prototype-distance learners."""
import json,time,statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.joint_prototype_representation import Learner,save
OUT=Path('results/sequential_prototype_representation_20261013')

class Sequential(Learner):
 def logits(self,h):
  # Only genuinely observed classes are eligible; old classes stay eligible.
  scores=super().logits(h)
  return scores.masked_fill((self.count==0)[None],-torch.inf)
 @torch.no_grad()
 def refit(self,x,y):
  sums=torch.zeros_like(self.prototype);counts=torch.zeros_like(self.count)
  for start in range(0,len(y),256):
   by=y[start:start+256];h=self.features(x[start:start+256]);sums.index_add_(0,by,h);counts+=torch.bincount(by,minlength=10)
  current=counts>0;self.prototype[current]=sums[current]/counts[current,None];self.count[current]=counts[current]
 def learn(self,x,y):
  h=self.features(x);loss=F.cross_entropy(self.logits(h),y);gradient=torch.autograd.grad(loss,self.weight)[0];active=self.active(gradient)
  with torch.no_grad():
   self.weight[:,active]-=self.lr*gradient[:,active];h=self.features(x)
   for c in y.unique():self.prototype[c].lerp_(h[y==c].mean(0),.1)
  return float(loss.detach())
 @torch.no_grad()
 def measure(self,x,y,order,stage):
  seen=(self.count>0);ix=seen[y];x=x[ix];y=y[ix];preds=[];sums=torch.zeros(10,256);counts=torch.zeros(10)
  for start in range(0,len(y),256):
   h=self.features(x[start:start+256]);by=y[start:start+256];preds.append(self.logits(h).argmax(-1));sums.index_add_(0,by,h);counts+=torch.bincount(by,minlength=10)
  pred=torch.cat(preds);pairs=[float((pred[torch.isin(y,torch.tensor(p))]==y[torch.isin(y,torch.tensor(p))]).float().mean()) for p in order[:stage]]
  return {'accuracy':float((pred==y).float().mean()),'pair_accuracy':pairs,'old_accuracy':st.mean(pairs[:-1]) if stage>1 else None,'new_accuracy':pairs[-1]},sums/counts.clamp_min(1)[:,None]

def verify():
 torch.manual_seed(91);x=torch.rand(30,784);y=torch.arange(30)%2;m=Sequential(7,'grouped',.3,.03);m.refit(x,y);old=m.prototype[:2].clone();oldcount=m.count[:2].clone()
 m.refit(x,y+2);assert torch.equal(m.prototype[:2],old) and torch.equal(m.count[:2],oldcount)
 m.learn(x,y+2);assert torch.equal(m.prototype[:2],old)
 score=m.logits(m.features(x));assert torch.isfinite(score[:,:4]).all() and torch.isneginf(score[:,4:]).all()
 # With all classes seen, sequential gradients and updates equal the joint learner.
 j=Learner(8,'dense',.3,.3);q=Sequential(8,'dense',.3,.3);all_y=torch.arange(30)%10;j.refit(x,all_y);q.refit(x,all_y);j.learn(x,all_y);q.learn(x,all_y)
 assert torch.equal(j.weight,q.weight) and torch.equal(j.prototype,q.prototype)
 return {'old_prototypes_and_counts_preserved_on_current_refit':True,'old_prototypes_preserved_on_new_class_learning':True,'old_classes_still_eligible_unseen_excluded':True,'all_seen_update_exactly_matches_joint_learner':True}

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();rows=[]
 selected=json.loads(Path('results/joint_prototype_representation_20261013/selection.json').read_text());save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 protocol={'question':'Do useful jointly validated prototype representations survive sequential class learning without refreshing old prototypes?',
 'training':'Five disjointclasspairs forward0/1,2/3,4/5,6/7,8/9 and reversepairorder;10epochs/currentpair,all60ktrainimagesavailableonlywhilecurrent. Same total600kgradientpresentationsas10epochjointstudy; extra currentdata-onlyprototypefitting/bootstrap passes. No replay, oldimage/configarchive,pretrainingorlearnedhead.',
 'learner':'ExactlyjointprototypeCEgradientlearner with eligibilitymask excludingunseenclasses only. Oldclasses alwayseligibleinlossandinference. Currentpairmeansbootstrapbeforelearningthenrefitaftereachcurrentepoch; oldmeans/countsunchanged. Oldprototypes serve as classifier competitors,notoldexamples. Training boundaries schedule currentprototypefitting; no taskID to predictions. Not boundary-free streamlearning.',
 'arms':'denseall256;groupedlargestgradientnormunitperfour64active;globalsparsetop64gradientnormglobally;frozeninitialweights. Same full256responseprototypeclassifier. No communication,usageprotection,old-errortrace. Fullgradientscomputedbeforemasking,notlocalbiologicallearning.',
 'settings':'Eacharmusesitsjointvalidationselectedlr/T;no sequential retuning, officialtestselection, or adaptiveprotection. Frozenlr0/T.3. Selection.jsoncopiedforprovenance.',
 'confirmation':'Same442-444MNIST/452-454Fashionseedsasjointstudyforpairedfollowup,notindependentreplication. Twoordersperseed;ordersnotindependentseeds. Commoninitialweightsandminibatchshuffle acrossarms. Full10ktestseenclassesateachstage;2/10epochfixedcheckpoints. Finalprototypeclassificationalwaysalltenclasses.',
 'memory':'Exactly814160persistentnumericbytesallarmsincluding10x256FP32means/10int64counts. Perstage testmeans/history/files are evaluation artifacts,nevermodelinputs; nohistoricalconfigretrieval. Finaltrainedmodelstatesavedforreproducibilityonly.',
 'drift':'TestclassmeanfeatureMSEfromacquisitiontofinalweights,diagnosticonly;testlabelsneverenterlearner. Backwardtransferfinalpairaccuracyminusacquisitionpairaccuracy averages firstfourpairs; includesnewclasscompetition.'}
 save(OUT/'protocol.json',protocol);save(OUT/'selection.json',selected);forward=[[0,1],[2,3],[4,5],[6,7],[8,9]]
 for ds,start in [('mnist',442),('fashion_mnist',452)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for seed in range(start,start+3):
   for ordername,order in [('forward',forward),('reverse',forward[::-1])]:
    for arm in ['dense','grouped','global_sparse','frozen']:
     setting=selected[ds][arm]['chosen'] if arm!='frozen' else {'lr':0.,'temperature':.3};m=Sequential(seed,arm,setting['lr'],setting['temperature']);initial=m.weight.detach().clone();history=[];means_at_acquisition={};train_seconds=0.;fit_seconds=0.
     for stage,pair in enumerate(order):
      seen=(m.count>0);old=m.prototype[seen].clone();oldcount=m.count[seen].clone();ids=torch.isin(y,torch.tensor(pair)).nonzero().flatten();cx=x[ids];cy=y[ids];t=time.perf_counter();m.refit(cx,cy);fit_seconds+=time.perf_counter()-t;checks=[]
      if arm!='frozen':
       for epoch in range(10):
        rng=torch.Generator().manual_seed(seed*10000+stage*100+epoch);t=time.perf_counter()
        for ix in torch.randperm(len(cy),generator=rng).split(128):m.learn(cx[ix],cy[ix])
        train_seconds+=time.perf_counter()-t;t=time.perf_counter();m.refit(cx,cy);fit_seconds+=time.perf_counter()-t
        if epoch+1 in [2,10]:v,means=m.measure(tx,ty,order,stage+1);checks.append({'epoch':epoch+1,**v})
      else:
       assert torch.equal(m.weight,initial);v,means=m.measure(tx,ty,order,stage+1);checks.append({'epoch':0,**v})
      assert torch.equal(m.prototype[seen],old) and torch.equal(m.count[seen],oldcount)
      for c in pair:means_at_acquisition[c]=means[c].clone()
      history.append({'stage':stage+1,'checkpoints':checks,'final':checks[-1],'classmean_drift':{str(c):float((means[c]-v).square().mean()) for c,v in means_at_acquisition.items()},'training_seconds':train_seconds,'prototype_fitting_seconds':fit_seconds})
     bwt=st.mean(history[-1]['final']['pair_accuracy'][i]-history[i]['final']['pair_accuracy'][i] for i in range(4));row={'dataset':ds,'seed':seed,'order':ordername,'pairs':order,'arm':arm,'lr':setting['lr'],'temperature':setting['temperature'],'numeric_bytes':m.bytes(),'history':history,'accuracy':history[-1]['final']['accuracy'],'backward_transfer':bwt,'weight_change_mse':float((m.weight.detach()-initial).square().mean())};rows.append(row);save(OUT/'results.json',rows)
     torch.save({'weight':m.weight.detach(),'prototype':m.prototype,'count':m.count,'arm':arm,'lr':m.lr,'temperature':m.temp},OUT/'models'/f'{ds}_{seed}_{ordername}_{arm}.pt')
     print(ds,seed,ordername,arm,'all',round(row['accuracy']*100,2),'old',round(history[-1]['final']['old_accuracy']*100,2),'new',round(history[-1]['final']['new_accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==48,'models':len(rows),'all_storage_equal':all(r['numeric_bytes']==814160 for r in rows)})
if __name__=='__main__':main()
