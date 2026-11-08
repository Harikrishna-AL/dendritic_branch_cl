"""Prospective no-bank online routing for the one-layer gradient-usefulness model."""
import json, time, statistics as st, argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.scripts import controlled_branch_depth as backbone
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/online_proxy_branch_retrieval_20261020');N=25;K=4;D=64
METHODS=['local','local_communication','central96','central144','branch_rls','direct_class_rls']
def save(p,x):p.write_text(json.dumps(x,indent=2))
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
@torch.no_grad()
def keys(x):
 p=F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),(7,9)).flatten(1);q=F.normalize(p,dim=-1);return F.normalize(torch.cat([q,torch.full((len(x),1),.2)],-1),dim=-1).numpy().copy()
class Local:
 def __init__(self,decay):self.sums=np.zeros((N,K,D),np.float32);self.count=np.zeros((N,K),np.float32);self.links=np.zeros((N,K,K),np.float32);self.decay=decay
 def update(self,q,route):
  ix=np.arange(N);self.sums[ix,route]*=self.decay;self.sums[ix,route]+=q.sum(0);self.count[ix,route]*=self.decay;self.count[ix,route]+=len(q);self.links[ix,route]*=self.decay;self.links[ix,route,np.roll(route,-1)]+=len(q)
 def routes(self,q,gamma):
  unary=np.einsum('bd,nkd->bnk',q,norm(self.sums));unary=np.where(self.count[None]>0,unary,-1e3)
  def softmax(v):
   v=v/.05;v-=v.max(-1,keepdims=True);p=np.exp(v);return p/p.sum(-1,keepdims=True)
  p=softmax(unary)
  if gamma:
   right=self.links/np.maximum(self.links.sum(2,keepdims=True),1);left=self.links/np.maximum(self.links.sum(1,keepdims=True),1)
   for _ in range(3):
    msg=.5*(np.einsum('bnl,nkl->bnk',np.roll(p,-1,axis=1),right)+np.einsum('bnk,nkl->bnl',np.roll(p,1,axis=1),np.roll(left,1,axis=0)));p=softmax(unary+gamma*msg)
  return p.argmax(-1)
 def size(self,comm):return self.sums.nbytes+self.count.nbytes+(self.links.nbytes if comm else 0)
class Central:
 def __init__(self,slots):self.mean=np.zeros((slots,D),np.float32);self.route=np.zeros((slots,N),np.uint8);self.count=np.zeros(slots,np.float32)
 def update(self,q,route):
  for item in q:
   used=self.count>0;eligible=np.flatnonzero(used & (self.route==route).all(1));best=-1
   if len(eligible):sim=norm(self.mean[eligible])@item;best=int(eligible[sim.argmax()])
   novel=best<0 or float(norm(self.mean[best:best+1])[0]@item)<.85
   if novel and (~used).any():best=int(np.flatnonzero(~used)[0]);self.mean[best]=item;self.route[best]=route;self.count[best]=1
   elif best<0:best=int(self.count.argmin());self.mean[best]=item;self.route[best]=route;self.count[best]=1
   else:self.count[best]+=1;self.mean[best]+=(item-self.mean[best])/self.count[best]
 def routes(self,q):
  used=self.count>0;assert used.any();return self.route[used][(q@norm(self.mean[used]).T).argmax(-1)].astype(np.int64)
 def size(self):return self.mean.nbytes+self.route.nbytes+self.count.nbytes
class Ridge:
 def __init__(self,outputs):self.cov=np.eye(D,dtype=np.float32);self.cross=np.zeros((outputs,D),np.float32)
 def update(self,q,target):self.cov+=q.T@q;self.cross+=target.T@q
 def weight(self,ridge):return np.linalg.solve((self.cov+(ridge-1)*np.eye(D)).astype(np.float64),self.cross.T.astype(np.float64)).T.astype(np.float32)
 def size(self):return self.cov.nbytes+self.cross.nbytes
class Memory:
 def __init__(self):self.local={d:Local(d) for d in [1.,.995]};self.central96=Central(96);self.central144=Central(144);self.branch_rls=Ridge(N*K);self.accepted=0
 def update(self,q,route):
  for local in self.local.values():local.update(q,route)
  self.central96.update(q,route);self.central144.update(q,route);target=np.zeros((len(q),N,K),np.float32);target[:,np.arange(N),route]=1;self.branch_rls.update(q,target.reshape(-1,N*K));self.accepted+=len(q)
def candidates():
 result=[]
 for stable in [1,10]:
  for decay in [1.,.995]:
   for gamma in [0.,.05,.2]:result.append({'method':'local' if gamma==0 else 'local_communication','stable':stable,'decay':decay,'gamma':gamma})
  for method in ['central96','central144']:result.append({'method':method,'stable':stable})
  for ridge in [.1,1.,10.]:result.append({'method':'branch_rls','stable':stable,'ridge':ridge})
 for ridge in [.1,1.,10.]:result.append({'method':'direct_class_rls','ridge':ridge})
 return result
def digest(mem,direct):
 import hashlib
 h=hashlib.sha256()
 for mm in mem.values():
  for obj in [*mm.local.values(),mm.central96,mm.central144,mm.branch_rls]:
   for v in vars(obj).values():
    if isinstance(v,np.ndarray):h.update(v.tobytes())
 for v in [direct.cov,direct.cross]:h.update(v.tobytes())
 return h.hexdigest()
def train(seed,x,y,order):
 m=backbone.make(seed,'branch1_proxy');opt=torch.optim.Adam(m.parameters(),lr=.003);mem={s:Memory() for s in [1,10]};direct=Ridge(10);bank=[];previous=None;streak=0;audit=[];t=time.perf_counter();model_seconds=0.
 for stage,pair in enumerate(order):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
    bx,by=x[ix],y[ix];q=keys(bx);route=m.routes()[0].numpy().astype(np.uint8);streak=streak+1 if previous is not None and np.array_equal(route,previous) else 1;previous=route.copy()
    # Association modules receive only current keys and PRE-update branch choices.
    for s,mm in mem.items():
     if streak>=s:mm.update(q,route)
    direct.update(q,F.one_hot(by,10).float().numpy());start=time.perf_counter();backbone.step(m,opt,bx,by,'branch1_proxy');model_seconds+=time.perf_counter()-start
  bank.append(m.routes()[0].clone());audit.append({'stage_for_analysis_only':stage,'accepted':{s:mm.accepted for s,mm in mem.items()}});print('STAGE',seed,stage+1,round(time.perf_counter()-t,1),flush=True)
 return m,mem,direct,torch.stack(bank),audit,{'combined_all_candidate_seconds':time.perf_counter()-t,'model_updates_seconds':model_seconds}
@torch.no_grad()
def evaluate(m,mem,direct,bank,x,y,order,settings):
 before={k:v.clone() for k,v in m.state_dict().items()};checksum=digest(mem,direct);q=np.concatenate([keys(b) for b in x.split(512)]);response=torch.cat([torch.tanh(m.layers[0](b)).reshape(-1,N,K) for b in x.split(512)]);target=y.numpy();truth=np.zeros(len(y),np.int64)
 for i,p in enumerate(order):truth[np.isin(target,p)]=i
 def metric(pred):
  pair=[float((pred[np.isin(target,p)]==target[np.isin(target,p)]).mean()) for p in order];return {'accuracy':float((pred==target).mean()),'pair_accuracy':pair,'old_accuracy':st.mean(pair[:-1]),'new_accuracy':pair[-1]}
 def classify(route):
  return m.head(response.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1).numpy()
 rows=[]
 for s in settings:
  method=s['method'];a=time.perf_counter()
  if method=='direct_class_rls':pred=(q@direct.weight(s['ridge']).T).argmax(-1);size=direct.size();extra={}
  else:
   mm=mem[s['stable']]
   if method.startswith('local'):route=mm.local[s['decay']].routes(q,s['gamma']);size=mm.local[s['decay']].size(method=='local_communication')
   elif method.startswith('central'):obj=getattr(mm,method);route=obj.routes(q);size=obj.size()
   else:route=(q@mm.branch_rls.weight(s['ridge']).T).reshape(-1,N,K).argmax(-1);size=mm.branch_rls.size()
   pred=classify(route);extra={'historical_branch_agreement_analysis':float((route==bank.numpy()[truth]).mean()),'whole_historical_route_match_analysis':float((route==bank.numpy()[truth]).all(-1).mean()),'accepted_presentations':mm.accepted}
  rows.append({**s,**metric(pred),'memory_bytes':size,'all_eval_seconds':time.perf_counter()-a,**extra})
 current=m.routes()[0].numpy();ordinary=metric(classify(np.broadcast_to(current,(len(x),N)).copy()));oracle=metric(classify(bank.numpy()[truth]));assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert digest(mem,direct)==checksum
 return rows,{'ordinary':ordinary,'oracle':oracle,'model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),'bank_not_used_for_predictions':True,'test_evaluation_writes_no_model_or_memory_state':True}
def dump_memory(path,mem,direct):
 arrays={}
 for stable,mm in mem.items():
  for label,obj in [(f'local{d}',v) for d,v in mm.local.items()]+[(k,getattr(mm,k)) for k in ['central96','central144','branch_rls']]:
   for field,v in vars(obj).items():
    if isinstance(v,np.ndarray):arrays[f'stable{stable}_{label}_{field}']=v
 arrays['direct_cov']=direct.cov;arrays['direct_cross']=direct.cross;np.savez_compressed(path,**arrays)
def verify():
 rng=np.random.default_rng(7);q=norm(rng.normal(size=(10,D)).astype(np.float32));route=rng.integers(0,4,N,dtype=np.uint8);m=Memory();m.update(q,route);assert m.local[1.].size(False)==26000 and m.local[1.].size(True)==27600 and m.central96.size()==27360 and m.branch_rls.size()==41984
 assert np.array_equal(m.local[1.].routes(q,0),np.broadcast_to(route,(10,N)));r=Ridge(10);target=np.eye(10,dtype=np.float32)
 for i in range(10):r.update(q[i:i+1],target[i:i+1])
 for ridge in [.1,1.,10.]:assert np.allclose(r.weight(ridge),np.linalg.solve(ridge*np.eye(D)+q.T@q,q.T@target).T,atol=3e-6)
 return {'local_current_activity_retrieval_synthetic_check':True,'online_ridge_matches_batch_reference':True,'memory_counts_checked':True,**backbone.verify()}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');a=parser.parse_args();torch.set_num_threads(2)
 if a.verify:print(json.dumps(verify(),indent=2));return
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();(OUT/'memories').mkdir();save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 save(OUT/'protocol.json',{'purpose':'Prospective no-bank online retrieval on one-layer25x4 gradient-estimated-usefulness backbone, not retrospective router training.','backbone':'78760parameters+100floaterrorstate, factualCE,Adam.003,clip5,batch128,2passes/currentpair. Chosen from prior depth development, not tuned on new tests.','write':'Fixed64pixelkey from7x9pooling,normalized+bias.2. Memory receives current key and PRE-update globalbranchchoices. No taskIDs, boundary notifications, classlabels or future endroutes enter routing memories. Direct class baseline alone receives currentlabels. Stabilitygate1or10 consecutive equalroutes; previous route/streak neverresetatboundary.','read':'Per-image branchchoices from memory only; no externally saved configuration bank. Alltenoutputs eligible; finalbackboneweightsfixed. Storedendroutes used ONLY after predictions for oracle/agreement analysis. No test updates.','local':'Perbranchkey sums/counts, cumulative or .995 decay per update to activebranch only. Dormantbranches do not decay. Ringcompatibility learned from contemporaneousneighborchoices;3inference rounds,gamma0/.05/.2. Localcuebroadcast but backboneordinarybackprop,notbiologicallocalclaim. Statistics are explicit auxiliarymemory.','controls':'Central96/144 online centroids tagged currentobservedroute,threshold.85; no saved individualexamples but explicitaggregate prototypes/configs. BranchRLS uses global64x64cov and100x64cross;ridge .1/1/10. DirectclassRLS uses samekeys, currentclasslabels,10outputs,ridge .1/1/10; noNNneeded.','selection':'Development seed540,50kfit/10kvalidationstratified1000/class; independently choose eachmethod by validation all10accuracy. Localdecay/stability/communication andRLSridge selectedhere. No officialtest tuning. All candidates retained.','confirmation':'Fresh541-545,bothforward/reverseorders,full60ktrain/full10ktest. Fiveindependentseeds;orderspaired. Allreadoutspairedsamebackbone. One dev+10confirmationbackbones.','scope':'New independent confirmation for no-bank decoding on proxy model. Multiple memorycandidates trained together only for efficiency; count perdeployablemodel notunion. Combined runtime notdeployablemethodcost. Currentlabels superviseNN/directbaseline, not routingassociation. Not a demonstration of dendrite-specific advantage.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
 gen=torch.Generator().manual_seed(540);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
 m,mem,direct,bank,audit,timing=train(540,x[mask],y[mask],SPLITS);rr,diag=evaluate(m,mem,direct,bank,x[valid],y[valid],SPLITS,candidates());save(OUT/'development.json',{'candidates':rr,'diagnostics':diag,'audit':audit,'timing':timing});chosen=[]
 for method in METHODS:
  best=max([r for r in rr if r['method']==method],key=lambda r:r['accuracy']);chosen.append({k:best[k] for k in ['method','stable','decay','gamma','ridge'] if k in best})
 save(OUT/'selection.json',chosen);print('SELECTED',chosen,flush=True);del m,mem,direct;records=[]
 for seed in range(541,546):
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   m,mem,direct,bank,audit,timing=train(seed,x,y,order);rr,diag=evaluate(m,mem,direct,bank,tx,ty,order,chosen);records.append({'seed':seed,'order':name,'methods':rr,'diagnostics':diag,'write_audit':audit,'timing':timing});save(OUT/'results.json',records);torch.save({'state_dict':m.state_dict(),'bank_for_analysis_only':bank},OUT/'models'/f'{seed}_{name}.pt');dump_memory(OUT/'memories'/f'{seed}_{name}.npz',mem,direct);print('RESULT',seed,name,'oracle',round(diag['oracle']['accuracy']*100,2),[(r['method'],round(r['accuracy']*100,2)) for r in rr],flush=True);del m,mem,direct
 save(OUT/'completion.json',{'complete':len(records)==10,'confirmation_backbones':10,'selected_readouts':sum(len(r['methods']) for r in records),'development_candidates':len(candidates())})
if __name__=='__main__':main()
