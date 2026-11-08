"""Online current-activity associations; future saved bank used at evaluation only."""
import json,time,statistics as st
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make
OUT=Path('results/online_config_memory_benchmark_20261020')
D=64
KEYS=['fixed_pixels','frozen_responses','evolving_responses']
def save(p,x):p.write_text(json.dumps(x,indent=2))
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
@torch.no_grad()
def keys(x,model,initial):
 p=F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),(7,9)).flatten(1)
 f=torch.tanh(F.linear(x,initial[0],initial[1]));e=torch.tanh(model.first(x))
 result={}
 for name,h in zip(KEYS,[p,F.adaptive_avg_pool1d(f[:,None],63).squeeze(1),F.adaptive_avg_pool1d(e[:,None],63).squeeze(1)]):
  q=F.normalize(h,dim=-1);q=F.normalize(torch.cat([q,torch.full((len(x),1),.2)],-1),dim=-1);result[name]=q.numpy().copy()
 return result
class Local:
 def __init__(self):self.sums=np.zeros((24,4,D),np.float32);self.count=np.zeros((24,4),np.float32);self.links=np.zeros((24,4,4),np.float32)
 def update(self,q,route):
  self.sums[np.arange(24),route]+=q.sum(0);self.count[np.arange(24),route]+=len(q);self.links[np.arange(24),route,np.roll(route,-1)]+=len(q)
 def scores(self,q,bank,gamma=0.):
  unary=np.einsum('bd,nkd->bnk',q,normalize(self.sums));unary=np.where(self.count[None]>0,unary,-1e3)
  p=np.exp((unary-unary.max(-1,keepdims=True))/.05);p/=p.sum(-1,keepdims=True)
  if gamma:
   right=self.links/np.maximum(self.links.sum(2,keepdims=True),1);left=self.links/np.maximum(self.links.sum(1,keepdims=True),1)
   for _ in range(3):
    message=.5*(np.einsum('bnl,nkl->bnk',np.roll(p,-1,axis=1),right)+np.einsum('bnk,nkl->bnl',np.roll(p,1,axis=1),np.roll(left,1,axis=0)))
    value=(unary+gamma*message)/.05;value-=value.max(-1,keepdims=True);p=np.exp(value);p/=p.sum(-1,keepdims=True)
  # Bank restricts inference; memory update never receives these future routes.
  logp=np.log(np.maximum(p,1e-30));return np.stack([logp[:,np.arange(24),route].sum(-1) for route in bank],-1)
 def size(self):return self.sums.nbytes+self.count.nbytes+self.links.nbytes
class Central:
 def __init__(self,slots):self.mean=np.zeros((slots,D),np.float32);self.routes=np.zeros((slots,24),np.uint8);self.count=np.zeros(slots,np.float32)
 def update(self,q,route):
  for item in q:
   used=self.count>0;eligible=np.flatnonzero(used & (self.routes==route).all(1));best=-1
   if len(eligible):
    sim=normalize(self.mean[eligible])@item;best=int(eligible[sim.argmax()])
   novel=best<0 or float(normalize(self.mean[best:best+1])[0]@item)<.85
   if novel and (~used).any():best=int(np.flatnonzero(~used)[0]);self.mean[best]=item;self.routes[best]=route;self.count[best]=1
   elif best<0:
    best=int(self.count.argmin());self.mean[best]=item;self.routes[best]=route;self.count[best]=1
   else:self.count[best]+=1;self.mean[best]+=(item-self.mean[best])/self.count[best]
 def scores(self,q,bank,penalty):
  used=self.count>0
  if not used.any():return np.zeros((len(q),len(bank)),np.float32)
  similarity=q@normalize(self.mean[used]).T;distance=(self.routes[used,None,:]!=bank[None]).mean(-1)
  return np.stack([(similarity-penalty*distance[:,j][None]).max(-1) for j in range(len(bank))],-1)
 def size(self):return self.mean.nbytes+self.routes.nbytes+self.count.nbytes
class RLS:
 def __init__(self):self.cov=np.eye(D,dtype=np.float32);self.cross=np.zeros((96,D),np.float32)
 def update(self,q,route):self.cov+=q.T@q;self.cross[np.arange(24)*4+route]+=q.sum(0)
 def scores(self,q,bank):
  # Exact online ridge sufficient statistics; solve transiently, no old samples.
  weight=np.linalg.solve(self.cov.astype(np.float64),self.cross.T.astype(np.float64)).T.astype(np.float32);out=(q@weight.T).reshape(-1,24,4)
  return np.stack([out[:,np.arange(24),route].sum(-1) for route in bank],-1)
 def size(self):return self.cov.nbytes+self.cross.nbytes
class Memory:
 def __init__(self):self.local=Local();self.central96=Central(96);self.central144=Central(144);self.rls=RLS();self.accepted=0
 def update(self,q,route):
  self.local.update(q,route);self.central96.update(q,route);self.central144.update(q,route);self.rls.update(q,route);self.accepted+=len(q)

def setting_list():
 return [{'method':'local','gamma':0.}]+[{'method':'local_communication','gamma':g} for g in [.05,.2]]+ [{'method':name,'penalty':p} for name in ['central96','central144'] for p in [.25,.5,1.]]+[{'method':'rls'}]
def score(memory,q,bank,s):
 if s['method'].startswith('local'):return memory.local.scores(q,bank,s['gamma'])
 if s['method'].startswith('central'):return getattr(memory,s['method']).scores(q,bank,s['penalty'])
 return memory.rls.scores(q,bank)
@torch.no_grad()
def route_logits(model,x,bank):
 out=[]
 for b in x.split(512):
  h=torch.tanh(model.first(b)).reshape(-1,24,4)[:,torch.arange(24)[None],torch.tensor(bank.astype(np.int64))];out.append(model.head(h))
 return torch.cat(out).numpy()
def result(logits,y,route,bank,order):
 pred=logits[np.arange(len(y)),route].argmax(-1);pairs=[float((pred[np.isin(y,p)]==y[np.isin(y,p)]).mean()) for p in order];true=np.zeros(len(y),np.int64)
 for i,p in enumerate(order):true[np.isin(y,p)]=i
 return {'accuracy':float((pred==y).mean()),'pair_accuracy':pairs,'old_accuracy':st.mean(pairs[:-1]),'new_accuracy':pairs[-1],'oracle_route_agreement':float((route==true).mean()),'route_fraction':(np.bincount(route,minlength=5)/len(y)).tolist()}

def train(seed,x,y,mask,order):
 torch.manual_seed(seed);m=make('local','hard');opt=torch.optim.Adam(m.parameters(),lr=.001);initial=[m.first.weight.detach().clone(),m.first.bias.detach().clone()];mem={(k,g):Memory() for k in KEYS for g in [1,10]};previous=None;streak=0;bank=[];audit=[];t=time.perf_counter()
 for stage,pair in enumerate(order):
  ids=torch.nonzero(mask & torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
    bx=x[ix];by=y[ix];route=m.gates().argmax(-1).numpy().astype(np.uint8);streak=streak+1 if previous is not None and np.array_equal(route,previous) else 1;previous=route.copy();q=keys(bx,m,initial)
    for k in KEYS:
     mem[(k,1)].update(q[k],route)
     if streak>=10:mem[(k,10)].update(q[k],route)
    opt.zero_grad(set_to_none=True);logits,candidates=m(bx,True);target=by[:,None,None].expand(candidates.shape[:-1]).flatten();error=F.cross_entropy(candidates.reshape(-1,10),target,reduction='none').reshape(candidates.shape[:-1]);loss=F.cross_entropy(logits,by)+.5*error.mean();feedback=error.detach().mean(0);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.observe(feedback)
  bank.append(m.gates().argmax(-1).numpy().astype(np.uint8));audit.append({'stage':stage+1,'accepted':{str(g):mem[('fixed_pixels',g)].accepted for g in [1,10]}});print('stage',seed,stage+1,'seconds',round(time.perf_counter()-t,1),flush=True)
 return m,initial,mem,np.stack(bank),audit,time.perf_counter()-t
@torch.no_grad()
def evaluate(m,initial,mem,bank,x,y,order,chosen=None):
 before={k:v.clone() for k,v in m.state_dict().items()};qparts={k:[] for k in KEYS}
 for bx in x.split(512):
  q=keys(bx,m,initial)
  for k in KEYS:qparts[k].append(q[k])
 qs={k:np.concatenate(v) for k,v in qparts.items()};logits=route_logits(m,x,bank);target=y.numpy();rows=[]
 conditions=chosen if chosen is not None else [{'key':k,'stable':g,**s} for k in KEYS for g in [1,10] for s in setting_list()]
 for s in conditions:
  scores=score(mem[(s['key'],s['stable'])],qs[s['key']],bank,s);route=scores.argmax(-1);z=mem[(s['key'],s['stable'])];method=s['method'];size=z.local.size() if method.startswith('local') else getattr(z,method).size();rows.append({**s,**result(logits,target,route,bank,order),'memory_bytes':size,'accepted_presentations':z.accepted})
 true=np.zeros(len(target),np.int64)
 for i,p in enumerate(order):true[np.isin(target,p)]=i
 diagnostics={'ordinary':result(logits,target,np.full(len(target),4,np.int64),bank,order),'oracle':result(logits,target,true,bank,order),'bank_bytes':bank.nbytes,'model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())}
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 return rows,diagnostics

def verify():
 rng=np.random.default_rng(88);q=normalize(rng.normal(size=(11,D)).astype(np.float32));routes=rng.integers(0,4,(11,24),dtype=np.uint8);a=RLS();target=np.zeros((11,96),np.float32)
 for i in range(11):a.update(q[i:i+1],routes[i]);target[i,np.arange(24)*4+routes[i]]=1
 expected=np.linalg.solve(np.eye(D)+q.astype(np.float64).T@q,q.astype(np.float64).T@target).T;actual=np.linalg.solve(a.cov.astype(np.float64),a.cross.T.astype(np.float64)).T;assert np.allclose(expected,actual,atol=2e-6)
 assert Local().size()==26496 and Central(96).size()==27264 and Central(144).size()==40896 and RLS().size()==40960
 return {'online_ridge_equals_offline_reference_on_synthetic_data':True,'memory_bytes_checked':True}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 save(OUT/'protocol.json',{'question':'Can online input/currentbranchassociations retrieve externallysavedconfigurations withoutoldimagerevisits?', 'training':'Originalhard24x4SplitMNISTnetwork,2passes/currentpair,Adam.001,clip5,CE+.5counterfactualCE,errorEMA.95. Routerssidecarsdonotaffectgradienttraining. No oldimage replay. Currentlabelsonlyoriginalnetworkloss;memoriesreceivecurrentkeyandpreupdateactivebranches,notclasslabels,taskID,boundaries,or futureendsavedroutes.', 'bank':'Five24-indexroutesexternallysavedatphaseendsandprovidedONLYatinference. Allmethodsmustchoosefromsamebank,alltenclassoutputscompete. Notautonomousconfigstorage. Taskboundariesconstructstream/archiveonly,notmemoryupdates.', 'keys':'D64:fixedpixelsadaptive7x9pooled63values;frozeninitial96branchresponsesadaptivepooled63;changing96branchresponsespooled63. EachL2normalizedthenappend.2biasandrenormalize. Frozenfeaturecontrolstoresadditional301440bytesinitialfirstlayer;otherkeyfunctionsdeterministicnolearnedparams. Differentkeyfamiliesare not equalrepresentationquality;frozen-vs-evolvingresponsesstartidentical.', 'memory':'Localbranchconditionedsums/counts+24-neuronring26496bytes. Communicating3loops,gamma.05/.2oroff. Central96onlinecentroidswithobservedroutes/counts27264bytes;central14440896bytes. Threshold.85;fullnewrouteevictsminimumcount. RLSridge1onlinekeycov/crossstat40960bytes;currentbranchonehot96targets. Exactridgeaddressrule,inspiredbyDeltaNet-PC,notfullpaperreproduction. No storedindividualreplayexamples;prototypes/associations/covarianceareexplicitmemory.', 'decoding':'LocalsoftmaxperbranchT.05,bankscoreaggregatesbranchlogprobabilities,optionalneighborbeliefmessages. Centralbankcompatibilitycosine-minusHammingpenalty.25/.5/1. RLSbankscoreaggregatespredictedbranchvalues. Allreceivetestimage/networkresponsesonly;testlabelsanalysisafterprediction.', 'development':'Seed510,50kfit/10kvalidationfromoriginaltrain(1000/class),forwardorder. Independentlyselectperkey/methodstability1or10consecutiveequalcurrentroutes;gamma/penaltywhereapplicable. Noofficialtestforselection. Allvalidationcandidatesretained.', 'confirmation':'Fresh511/512/513,forward/reverseorders;full60ktrain/full10ktest. Fixedselectedsettings,3pairedseeds,ordersnotindependentreplications. 6freshbackbones+1development. All routers pairedonsamebackbone.', 'scope':'Savedbankandglobalexact-routestabilitysignalareengineeringaids;localmemoryneedsglobalcuebroadcast,originalcounterfactualbackpropnotbiologicallocal. No testtimelearning. Statecountsincludeallassociations;inferenceworkingmemoryexcluded. Old82.51retrospectiveresultdifferentcohort/teaching/dataaccess,notpairedcomparison.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);gen=torch.Generator().manual_seed(510);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
 m,initial,mem,bank,audit,seconds=train(510,x,y,mask,SPLITS);dev,diag=evaluate(m,initial,mem,bank,x[valid],y[valid],SPLITS);save(OUT/'development.json',{'candidates':dev,'diagnostics':diag,'audit':audit,'seconds':seconds});chosen=[]
 for k in KEYS:
  for method in ['local','local_communication','central96','central144','rls']:
   best=max([r for r in dev if r['key']==k and r['method']==method],key=lambda r:r['accuracy']);chosen.append({p:best[p] for p in ['key','stable','method','gamma','penalty'] if p in best})
 save(OUT/'selection.json',chosen);print('selected',chosen,flush=True);records=[]
 del m,mem
 for seed in [511,512,513]:
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   m,initial,mem,bank,audit,seconds=train(seed,x,y,torch.ones(len(y),dtype=torch.bool),order);rr,diag=evaluate(m,initial,mem,bank,tx,ty,order,chosen);records.append({'seed':seed,'order':name,'methods':rr,'diagnostics':diag,'write_audit':audit,'training_and_all_memory_seconds':seconds});save(OUT/'results.json',records)
   torch.save({'state':m.state_dict(),'bank':bank.tolist()},OUT/'models'/f'{seed}_{name}.pt');print('RESULT',seed,name,'ordinary',round(diag['ordinary']['accuracy']*100,2),'oracle',round(diag['oracle']['accuracy']*100,2),[(r['key'],r['method'],round(r['accuracy']*100,2)) for r in rr],flush=True)
   del m,mem
 save(OUT/'completion.json',{'complete':len(records)==6,'backbones':len(records),'selected_router_evaluations':sum(len(r['methods']) for r in records)})
if __name__=='__main__':main()
