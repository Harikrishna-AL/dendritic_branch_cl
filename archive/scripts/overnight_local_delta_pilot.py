"""Validation-only: local error-correcting association versus positive key averages."""
import json,math,time
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/local_delta_pilot')
class Delta:
 def __init__(self,lr,loss,schedule):self.weight=np.zeros((100,64),np.float32);self.lr=lr;self.loss=loss;self.schedule=schedule;self.updates=0;self.failed=False
 def update(self,q,route):
  if self.failed:return
  pred=(q@self.weight.T).reshape(-1,25,4);target=np.zeros_like(pred);target[:,np.arange(25),route]=1
  if self.loss=='ce':
   pred-=pred.max(-1,keepdims=True);pred=np.exp(pred);pred/=pred.sum(-1,keepdims=True)
  residual=(target-pred).reshape(-1,100);rate=self.lr*(math.sqrt(100/(100+self.updates)) if self.schedule=='sqrt' else 1.);self.weight+=rate*(residual.T@q)/len(q);self.updates+=1
  if not np.isfinite(self.weight).all() or np.abs(self.weight).max()>1e6:self.failed=True
 def routes(self,q):return (q@self.weight.T).reshape(-1,25,4).argmax(-1)
class Study:
 def __init__(self):self.delta={(lr,loss,s):Delta(lr,loss,s) for lr in [.03,.1,.3,1.,3.] for loss in ['mse','ce'] for s in ['constant','sqrt']};self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.delta.values():obj.update(q,route)
  self.local.update(q,route);target=np.zeros((len(q),25,4),np.float32);target[:,np.arange(25),route]=1;self.ridge.update(q,target.reshape(-1,100));self.accepted+=len(q)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'scope':'Validation-only, same560stratified50k/10k asinitialpilot. Noofficialtestdataaccess. Local independent neuron delta-rule association of fixed64inputkeys to currentpreupdatebranchactivity; no sharedcovariance, labels, IDs,boundaries,futuresavedroutes oroldtaskreplay inassociationwrites. BackboneoriginalCE/proxyAdam.003. MSEorCE,lr.03/.1/.3/1/3,constantor1/sqrt decay,stable1/10. 40delta candidates,25600B each. Matchedcurrentkeylocalaverage26000B andglobalridge41984Bcontrols;directridge18944B. Allcandidates retained; no testtuning.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);g=torch.Generator().manual_seed(560);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=g)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;vx,vy=x[valid],y[valid];x,y=x[mask],y[mask];e.Memory=Study;m,mem,direct,bank,audit,timing=e.train(560,x,y,SPLITS);previous=torch.load(OUT.parent/'cue_pilot/backbone.pt',weights_only=True);assert all(torch.equal(v,previous[k]) for k,v in m.state_dict().items())
 with torch.no_grad():
  q=np.concatenate([e.keys(b) for b in vx.split(512)]);response=torch.cat([torch.tanh(m.layers[0](b)).reshape(-1,25,4) for b in vx.split(512)])
  def acc(route):return float((m.head(response.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
  rows=[]
  for stable,mm in mem.items():
   for (lr,loss,schedule),obj in mm.delta.items():rows.append({'method':'delta','stable':stable,'lr':lr,'loss':loss,'schedule':schedule,'failed':obj.failed,'accuracy':None if obj.failed else acc(obj.routes(q)),'extra_bytes':obj.weight.nbytes})
   rows.append({'method':'local_average','stable':stable,'accuracy':acc(mm.local.routes(q,0.)),'extra_bytes':26000});rows.append({'method':'global_ridge','stable':stable,'accuracy':acc((q@mm.ridge.weight(.1).T).reshape(-1,25,4).argmax(-1)),'extra_bytes':mm.ridge.size()})
  rows.append({'method':'direct_ridge','accuracy':float(((q@direct.weight(1.).T).argmax(-1)==vy.numpy()).mean()),'extra_bytes':direct.size()})
 e.save(OUT/'results.json',{'validation_only':True,'rows':rows,'timing_all_candidates':timing,'backbone_bitwise_matches_initial_pilot':True});e.save(OUT/'completion.json',{'complete':len(rows)==45,'official_test_accessed':False});print('RESULTS',sorted(rows,key=lambda r:-(r['accuracy'] or -1)),flush=True)
if __name__=='__main__':main()
