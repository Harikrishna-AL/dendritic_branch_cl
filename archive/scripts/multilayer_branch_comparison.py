"""Two hidden layers: matched MLP, exact deep counterfactuals, first-order routing."""
import argparse,json,math,time,statistics as st,resource,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/multilayer_branch_comparison_20261020')
def save(p,x):p.write_text(json.dumps(x,indent=2))
class DeepBranch(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.ModuleList([nn.Linear(784,96),nn.Linear(24,96)]);self.head=nn.Linear(24,10);self.register_buffer('error',torch.full((2,24,4),math.log(10)))
 def routes(self):return self.error.argmin(-1)
 def forward(self,x,record=False,routes=None):
  route=self.routes() if routes is None else routes;records=[]
  for i,layer in enumerate(self.layers):
   b=torch.tanh(layer(x)).reshape(-1,24,4);h=b.gather(-1,route[i][None,:,None].expand(len(b),-1,-1)).squeeze(-1)
   if record:
    if h.requires_grad:h.retain_grad()
    records.append((b,h))
   x=h
  logits=self.head(x);return (logits,records,route.detach().clone()) if record else logits
 def alternatives(self,records,route,logits):
  b1,h1=records[0];b2,h2=records[1];delta=b1-h1[:,:,None]
  # B x selected-neuron x replacement-branch x downstream-input-coordinate.
  first=h1[:,None,None,:]+delta[:,:,:,None]*torch.eye(24)[None,:,None,:]
  downstream=torch.tanh(self.layers[1](first.reshape(-1,24))).reshape(len(h1),24,4,24,4)
  pool=downstream.gather(-1,route[1][None,None,None,:,None].expand(len(h1),24,4,-1,-1)).squeeze(-1)
  a1=self.head(pool)
  a2=logits[:,None,None,:]+(b2-h2[:,:,None])[:,:,:,None]*self.head.weight.T[None,:,None,:]
  return torch.stack([a1,a2],1)
 @torch.no_grad()
 def selected_forward(self,x):
  # Gather one filter/weight row perneuron before each linear operation.
  route=self.routes()
  for i,layer in enumerate(self.layers):
   ids=torch.arange(24)*4+route[i];x=torch.tanh(F.linear(x,layer.weight[ids],layer.bias[ids]))
  return self.head(x)
 @torch.no_grad()
 def update(self,estimates):self.error.mul_(.95).add_(estimates,alpha=.05)
def make(seed,arm):
 torch.manual_seed(seed)
 return nn.Sequential(nn.Linear(784,88),nn.Tanh(),nn.Linear(88,88),nn.Tanh(),nn.Linear(88,10)) if arm=='mlp' else DeepBranch()
def step(m,opt,x,y,arm):
 opt.zero_grad(set_to_none=True)
 if arm=='mlp':loss=F.cross_entropy(m(x),y);loss.backward()
 else:
  logits,records,route=m(x,True);base=F.cross_entropy(logits,y)
  if arm=='branch_exact':
   alt=m.alternatives(records,route,logits);target=y[:,None,None,None].expand(alt.shape[:-1]).flatten();error=F.cross_entropy(alt.reshape(-1,10),target,reduction='none').reshape(alt.shape[:-1]);loss=base+.5*error.mean();est=error.detach().mean(0);loss.backward()
  else:
   loss=base;loss.backward()
   # h.grad includes1/B frommeanCE; SUM overbatch is meanperexamplelosschange.
   est=torch.stack([base.detach()+(h.grad.detach()[:,:,None]*(b.detach()-h.detach()[:,:,None])).sum(0) for b,h in records])
 nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 if arm!='mlp':m.update(est)
 return float(loss.detach())
@torch.no_grad()
def scores(m,x,routes=None):return torch.cat([m(b) if routes is None else m(b,routes=routes) for b in x.split(512)])
def metrics(pred,y,order):
 a=[float((pred[torch.isin(y,torch.tensor(p))]==y[torch.isin(y,torch.tensor(p))]).float().mean()) for p in order]
 return {'accuracy':float((pred==y).float().mean()),'pair_accuracy':a,'old_accuracy':st.mean(a[:-1]),'new_accuracy':a[-1]}
def fit(seed,arm,lr,x,y,ex,ey,order,mode,details=True):
 m=make(seed,arm);opt=torch.optim.Adam(m.parameters(),lr=lr);history=[];routes=[];seconds=0.;presentations=0
 groups=order if mode=='sequential' else [list(range(10))]
 for stage,pair in enumerate(groups):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten();t=time.perf_counter()
  for e in range(2):
   gen=torch.Generator().manual_seed(seed*100000+(stage*100 if mode=='sequential' else 9000)+e)
   for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):step(m,opt,x[ix],y[ix],arm);presentations+=len(ix)
  seconds+=time.perf_counter()-t
  if mode=='sequential':
   before={k:v.clone() for k,v in m.state_dict().items()};ss=scores(m,ex).argmax(-1);metric=metrics(ss,ey,order);history.append(metric)
   if arm!='mlp':routes.append(m.routes().clone())
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 final=metrics(scores(m,ex).argmax(-1),ey,order);result={'arm':arm,'lr':lr,'mode':mode,'accuracy':final['accuracy'],'final':final,'history':history,'training_seconds':seconds,'presentations':presentations,'parameters':sum(p.numel() for p in m.parameters()),'inference_model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),'optimizer_bytes':sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))}
 if mode=='sequential':
  result['acquisition']=st.mean(history[i]['pair_accuracy'][i] for i in range(5));result['forgetting']=st.mean(max(history[j]['pair_accuracy'][i] for j in range(i,5))-final['pair_accuracy'][i] for i in range(4))
  if arm!='mlp' and details:
   before={k:v.clone() for k,v in m.state_dict().items()};matrix=[];oracle=torch.empty_like(ey)
   for i,r in enumerate(routes):
    pred=scores(m,ex,r).argmax(-1);matrix.append(metrics(pred,ey,order)['pair_accuracy']);mask=torch.isin(ey,torch.tensor(order[i]));oracle[mask]=pred[mask]
   result['oracle_routes']=metrics(oracle,ey,order);result['route_by_pair_accuracy']=matrix;result['routes']=torch.stack(routes).tolist();result['bank_bytes_packed']=5*2*24
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
  elif arm=='mlp' and details:
   raw=scores(m,ex);mask=torch.ones_like(raw,dtype=torch.bool)
   for p in order:
    rows=torch.nonzero(torch.isin(ey,torch.tensor(p))).flatten();mask[rows[:,None],torch.tensor(p)[None]]=False
   result['oracle_pair_output_mask']=metrics(raw.masked_fill(mask,-torch.inf).argmax(-1),ey,order)
 return m,result

def verify():
 torch.manual_seed(6);x=torch.rand(7,784);y=torch.arange(7)%10;m=make(9,'branch_exact');logits,recs,route=m(x,True);alt=m.alternatives(recs,route,logits)
 for layer,n,k in [(0,0,1),(0,12,3),(1,5,2),(1,23,0)]:
  changed=route.clone();changed[layer,n]=k;direct=m(x,routes=changed);assert torch.allclose(direct,alt[:,layer,n,k],atol=1e-6,rtol=1e-5)
 assert torch.allclose(m(x),m.selected_forward(x),atol=1e-6)
 # Check first-order response perturbation against an actual infinitesimal change.
 loss=F.cross_entropy(logits,y);dh=torch.autograd.grad(loss,recs[0][1],retain_graph=True)[0];direction=torch.randn_like(recs[0][1]);eps=1e-4
 def downstream(h):
  b=torch.tanh(m.layers[1](h)).reshape(-1,24,4);h2=b.gather(-1,route[1][None,:,None].expand(len(b),-1,-1)).squeeze(-1);return F.cross_entropy(m.head(h2),y)
 finite=(downstream(recs[0][1]+eps*direction)-downstream(recs[0][1]-eps*direction))/(2*eps);analytic=(dh*direction).sum();assert torch.allclose(finite,analytic,atol=2e-3,rtol=.03)
 assert sum(p.numel() for p in m.parameters())==78010 and sum(p.numel() for p in make(1,'mlp').parameters())==77802
 return {'deep_counterfactual_logits_match_full_changed_route_forward':True,'selected_weight_inference_matches_full_branch_forward':True,'first_order_gradient_matches_finite_difference':True,'parameter_counts_checked':True}

def profile(arm,path):
 torch.set_num_threads(2);torch.manual_seed(999);x=torch.rand(128,784);y=torch.arange(128)%10
 unit=1 if sys.platform=='darwin' else 1024;baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit;m=make(999,arm);opt=torch.optim.Adam(m.parameters(),lr=.001)
 for _ in range(10):step(m,opt,x,y,arm)
 blocks=[]
 for _ in range(5):
  t=time.perf_counter()
  for _ in range(20):step(m,opt,x,y,arm)
  blocks.append((time.perf_counter()-t)*1000/20)
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
 # Deduplicated storage retained byautograd, excludingparameterstorage.
 parameters={p.untyped_storage().data_ptr() for p in m.parameters()};saved={}
 def pack(t):
  storage=t.untyped_storage();ptr=storage.data_ptr()
  if ptr not in parameters:saved[ptr]=storage.nbytes()
  return t
 with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
  if arm=='mlp':loss=F.cross_entropy(m(x),y)
  else:
   log,rec,route=m(x,True);loss=F.cross_entropy(log,y)
   if arm=='branch_exact':
    alt=m.alternatives(rec,route,log);target=y[:,None,None,None].expand(alt.shape[:-1]).flatten();loss=loss+.5*F.cross_entropy(alt.reshape(-1,10),target)
  retained=sum(saved.values());loss.backward()
 inference=[]
 with torch.no_grad():
  for batch in [1,128]:
   bx=x[:batch];funcs=[('standard',lambda:m(bx))]
   if arm!='mlp':funcs.append(('selected_weights',lambda:m.selected_forward(bx)))
   for label,fn in funcs:
    for _ in range(20):fn()
    times=[]
    for _ in range(10):
     t=time.perf_counter()
     for _ in range(100):fn()
     times.append((time.perf_counter()-t)*1000/100)
    inference.append({'batch':batch,'path':label,'median_ms':st.median(times)})
 save(path,{'arm':arm,'threads':2,'training_batch':128,'training_step_median_ms':st.median(blocks),'process_baseline_peak_rss_bytes':baseline,'process_peak_rss_after_training_bytes':peak,'increase_over_baseline_peak_bytes':max(0,peak-baseline),'autograd_retained_unique_nonparameter_storage_bytes':retained,'inference':inference,'scope':'Freshprocess/syntheticinputs.10warmupupdates,5x20trainingsteps;inference20warmups10x100calls. RSSincludesruntime/optimizer/allocatornotjustactivations;baselineispriorprocesshighwatermark,notcurrentRSS. Retainedstorageisforwardautogradsavedstorage,notfullpeak. CPUonly,notFLOPs/GPU.'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['mlp','branch_exact','branch_proxy']);p.add_argument('--profile-out',type=Path);args=p.parse_args()
 if args.profile:profile(args.profile,args.profile_out);return
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 protocol={'question':'Performance/compute of2-hidden-layerbranchednetworkversusparameter-matchedMLP onSplitMNIST.', 'architecture':'Branch784->(24neuronsx4branches)->24pooled->(24x4)->24pooled->10logits,78010parameters,192floaterrorstate. MLP784->88tanh->88tanh->10,77802parameters(0.27%fewer). SimilarMACcountsforfactualforward;differenthiddenpooledwidth/capacity.', 'rules':'ExactbranchCE+.5mean192one-neuroncounterfactualCEs,earlyalternativesrecompute nonlinear downstreamlayer,latealternativeslinearshortcut. ErrorEMA.95/selectminperlayer. ProxybranchplainCE,EMAofL+sum_batchdL/dh*(branch-h)foreachlayer;no auxiliaryalternativeloss. MLPplainCE. Proxyisnewapproximation,notunchangedoriginalalgorithm.', 'training':'Rawpixels/255,Adam,clip5,batch128,2passes/currentpair,5tasks,all10outputs,nopretrain/replay/taskIDs/boundaryresets. Gatesglobalacrossimages,notautonomousperinputretrieval. No usageprotection/sidecarmemory.', 'development':'Seed520,50kfit/10kheldoutfromtrainingstratified1000/class. Eachruleindependentlyselectslr.0003/.001/.003. Selectsequentialsettingsbyfinalall10accuracy(acquisitiontiebreak);selectjointsettingsindependentlybyjointall10accuracy. Fixedforwarddevelopmentorder. No officialtestselection.', 'confirmation':'Fresh521/522/523,bothforward/reverseorders,all60ktrain/full10ktest. 18sequentialrunsplus9jointcontrols,2fullcurrentpasses,120kpresentations/run. Jointallclassesmixedsameexposure. Threeindependentseeds;twoorderspaired.', 'metrics':'Finalall10,acquisition,oldpairforgetting,newpair,trainingseconds. BranchoraclerestoresBOTHlayerconfigurationsatfinalweights/headunchanged,all10classeseligible;taskawareonlydiagnostic. MLPoraclepairmaskdifferentinterventionlabeledseparately.', 'compute':'Freshprocessperrule,2CPUthreads,syntheticbatch128samearchitecture. Walltrainingstepincludesrouting/errorupdates/Adam. Fullruntrainingtimingincludescurrentbatchesonly,notvalidationevaluation. CPURSS/forwardretainedactivationstorage/inferencebatch1&128reported. Optimizedselectedweightsinferencevalidatedexact,usescurrentfixedrouteandnoexternalroutercost.'}
 save(OUT/'protocol.json',protocol);raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);gen=torch.Generator().manual_seed(520);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;selection={};rows=[]
 for mode in ['sequential','joint']:
  selection[mode]={}
  for arm in ['mlp','branch_exact','branch_proxy']:
   candidates=[]
   for lr in [.0003,.001,.003]:
    _,r=fit(520,arm,lr,x[mask],y[mask],x[valid],y[valid],SPLITS,mode,False);candidates.append(r);print('DEV',mode,arm,lr,round(r['accuracy']*100,2),flush=True)
   chosen=max(candidates,key=lambda z:(z['accuracy'],z.get('acquisition',0)));selection[mode][arm]={'chosen':chosen,'candidates':candidates};save(OUT/'selection.json',selection)
 for seed in [521,522,523]:
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   for arm in ['mlp','branch_exact','branch_proxy']:
    lr=selection['sequential'][arm]['chosen']['lr'];m,r=fit(seed,arm,lr,x,y,tx,ty,order,'sequential');r.update(seed=seed,order=name);rows.append(r);save(OUT/'results.json',rows);torch.save(m.state_dict(),OUT/'models'/f'{seed}_{name}_{arm}.pt');print('TEST',seed,name,arm,'all',round(r['accuracy']*100,2),'acq',round(r['acquisition']*100,2),'oracle',round(r.get('oracle_routes',r['final'])['accuracy']*100,2),'sec',round(r['training_seconds'],2),flush=True)
  for arm in ['mlp','branch_exact','branch_proxy']:
   lr=selection['joint'][arm]['chosen']['lr'];_,r=fit(seed,arm,lr,x,y,tx,ty,SPLITS,'joint');r.update(seed=seed,order='mixed');rows.append(r);save(OUT/'results.json',rows);print('JOINT',seed,arm,round(r['accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==27,'confirmation_runs':len(rows),'development_runs':18})
if __name__=='__main__':main()
