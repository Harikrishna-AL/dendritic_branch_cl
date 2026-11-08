"""Joint prototype-distance acquisition gate; grouped versus dense gradient learning."""
import json,time,statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
OUT=Path('results/joint_prototype_representation_20261013')
def save(p,x):p.write_text(json.dumps(x,indent=2))
class Learner:
 def __init__(self,seed,arm,lr,temp):
  gen=torch.Generator().manual_seed(seed);self.weight=torch.nn.Parameter(torch.randn(785,256,generator=gen)*.02)
  self.prototype=torch.zeros(10,256);self.count=torch.zeros(10,dtype=torch.int64);self.arm=arm;self.lr=lr;self.temp=temp
 def features(self,x):return torch.tanh(torch.cat([F.normalize(x,dim=1),torch.ones(len(x),1)],1)@self.weight)
 def logits(self,h):return (2*h@self.prototype.T-self.prototype.square().sum(-1)[None])/self.temp
 def active(self,gradient):
  power=gradient.square().sum(0)
  if self.arm=='grouped':return torch.arange(64)*4+power.reshape(64,4).argmax(-1)
  if self.arm=='global_sparse':return power.topk(64).indices
  return torch.arange(256)
 def learn(self,x,y):
  h=self.features(x);loss=F.cross_entropy(self.logits(h),y);gradient=torch.autograd.grad(loss,self.weight)[0];active=self.active(gradient)
  with torch.no_grad():
   self.weight[:,active]-=self.lr*gradient[:,active]
   h=self.features(x)
   for c in y.unique():self.prototype[c].lerp_(h[y==c].mean(0),.1)
  return float(loss)
 @torch.no_grad()
 def refit(self,x,y):
  sums=torch.zeros_like(self.prototype);counts=torch.zeros_like(self.count)
  for start in range(0,len(y),256):
   by=y[start:start+256];h=self.features(x[start:start+256]);sums.index_add_(0,by,h);counts+=torch.bincount(by,minlength=10)
  self.prototype.copy_(sums/counts[:,None]);self.count.copy_(counts)
 @torch.no_grad()
 def evaluate(self,x,y):
  pred=torch.cat([self.logits(self.features(b)).argmax(-1) for b in x.split(256)])
  return {'accuracy':float((pred==y).float().mean()),'class_accuracy':[float((pred[y==c]==c).float().mean()) for c in range(10)]}
 def bytes(self):return sum(t.numel()*t.element_size() for t in [self.weight,self.prototype,self.count])
def fit(seed,arm,lr,temp,x,y,ex,ey,epochs=10):
 m=Learner(seed,arm,lr,temp);m.refit(x,y);history=[{'epoch':0,**m.evaluate(ex,ey)}];training=0.;fitting=0.;weights=m.weight.detach().clone()
 if arm!='frozen':
  for epoch in range(epochs):
   rng=torch.Generator().manual_seed(seed*1000+epoch);losses=[];t=time.perf_counter()
   for ix in torch.randperm(len(y),generator=rng).split(128):losses.append(m.learn(x[ix],y[ix]))
   training+=time.perf_counter()-t;t=time.perf_counter();m.refit(x,y);fitting+=time.perf_counter()-t
   if epoch+1 in [2,5,10]:history.append({'epoch':epoch+1,**m.evaluate(ex,ey),'training_loss':st.mean(losses)})
 else:assert torch.equal(weights,m.weight)
 return {'arm':arm,'lr':lr,'temperature':temp,'history':history,'accuracy':history[-1]['accuracy'],'numeric_bytes':m.bytes(),'training_seconds':training,'prototype_fitting_seconds':fitting,'weight_change_mse':float((m.weight.detach()-weights).square().mean())}
def verify():
 torch.manual_seed(60);m=Learner(3,'dense',.03,.03);x=torch.rand(13,784);y=torch.arange(13)%10;m.refit(x,y);h=m.features(x)
 direct=-(h[:,None]-m.prototype[None]).square().sum(-1)/m.temp
 assert torch.allclose(F.log_softmax(m.logits(h),-1),F.log_softmax(direct,-1),atol=2e-6)
 grad=torch.autograd.grad(F.cross_entropy(m.logits(h),y),m.weight)[0];old=m.weight.detach().clone();m.learn(x,y)
 assert torch.allclose(m.weight,old-.03*grad,atol=1e-7)
 for arm in ['grouped','global_sparse']:
  g=Learner(3,arm,.03,.03);g.refit(x,y);before=g.weight.detach().clone();gradient=torch.autograd.grad(F.cross_entropy(g.logits(g.features(x)),y),g.weight)[0];chosen=g.active(gradient);g.learn(x,y);inactive=torch.ones(256,dtype=torch.bool);inactive[chosen]=False
  assert torch.equal(g.weight[:,inactive],before[:,inactive]);assert len(chosen)==64
  if arm=='grouped':assert torch.equal(torch.bincount(chosen//4,minlength=64),torch.ones(64,dtype=torch.int64))
 return {'prototype_logits_match_negative_squared_distance':True,'dense_update_matches_loss_gradient':True,'sparse_updates_change_only_selected_units':True,'grouped_update_selects_one_per_four':True}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);selection={};rows=[];save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 save(OUT/'protocol.json',{'question':'Can prototype-distance supervised learning improve representations over unchanged random features when all classes are available jointly?',
 'data':'RealcachedMNIST/Fashion60ktrain10ktest,alltenclassesjointlymixed. L2normalized784pixelsplusbias,256tanhresponses. Sameinitialinputweights/minibatchorder acrossarms perseed. No pretraining, replay ortaskIDs; thisisjointsupervisedlearning,notcontinuallearning.',
 'loss':'CE over negative squared Euclidean distances to detached classmeanprototypes;temperatureT. Bootstrapmeansfromtrainingbeforefirstupdate;postupdateminibatchclassmeansEMAalpha.1;refitallclassmeansfromtrainingaftereachepoch. Testlabels onlyaccuracy. Tenclass256FP32meansandint64counts;no learnedhead/fixedteachingcodes.',
 'arms':'denseupdatesall256units;groupedselectslargestgradientnormresponseperfour-unitgroup(64active);globalsparseselects64largestgradientnormunitsglobally;frozeninitialweightsno gradientlearning. AllinferencefullresponseEuclideanprototypes,so groupingrestrictslearningonly. Gradientcomputedforallweightsbeforemasking;notcomputationallylocallearningor64-unitgradientcost. No communication, usageprotectionorhistoricalshorttermmemoryinthisacquisitiongate.',
 'development':'Devseed441MNIST451Fashion,50kfit/10kheldoutstratified1000/class. Eachtrainablearmindependentlyselectslr[.03,.3],T[.03,.3] byvalidationaccuracyafter10epochs. Fourcandidatesperarm;fixed10epochtrainingbudget. Testnotusedforselection.',
 'confirmation':'Fresh442-444MNIST452-454Fashion,full60ktrain/full10ktest,10epochscommonbudget,checkpoints0/2/5/10. Frozenmeansfitusingall60ktrain. Learned/frozeninitialweightsidenticalperseed. Finitepersistentstorage814160bytesallarms;temporarygradients/workspacenotcounted. Prototypefittingextra trainingdatapassescountedseparately.'})
 for ds,dev in [('mnist',441),('fashion_mnist',451)]:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  gen=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;selected={}
  for arm in ['dense','grouped','global_sparse']:
   candidates=[]
   for lr in [.03,.3]:
    for temp in [.03,.3]:
     r=fit(dev,arm,lr,temp,x[mask],y[mask],x[val],y[val]);candidates.append(r);save(OUT/'development_progress.json',{'dataset':ds,'arm':arm,'candidates':candidates});print('dev',ds,arm,lr,temp,round(r['accuracy']*100,2),flush=True)
   selected[arm]={'chosen':max(candidates,key=lambda r:r['accuracy']),'candidates':candidates}
  selection[ds]=selected;save(OUT/'selection.json',selection)
  for seed in range(dev+1,dev+4):
   for arm in ['dense','grouped','global_sparse','frozen']:
    s=selected[arm]['chosen'] if arm!='frozen' else {'lr':0.,'temperature':.03};r=fit(seed,arm,s['lr'],s['temperature'],x,y,tx,ty);rows.append({'dataset':ds,'seed':seed,**r});save(OUT/'results.json',rows);print('test',ds,seed,arm,round(r['accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==24,'confirmation_models':len(rows),'development_models':24})
if __name__=='__main__':main()
