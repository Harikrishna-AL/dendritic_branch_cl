"""Fixed-checkpoint credit audit and supervised hard-gate recovery, not autonomous CL."""
import argparse,json,time,math,statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts import controlled_branch_depth as arch
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/branch_node_perturbation_20261023');CK=Path('results/controlled_branch_depth_20261020/models')
def save(p,x):p.write_text(json.dumps(x,indent=2))
def load(seed,depth,order):
 m=arch.make(seed,f'branch{depth}_proxy');state=torch.load(CK/f'{seed}_{order}_branch{depth}_proxy.pt',weights_only=True)['state_dict'];m.load_state_dict(state);m.eval();return m

def downstream(m,h,route):
 shape=h.shape[:-1];v=h.reshape(-1,m.n)
 for i in range(1,m.depth):
  b=torch.tanh(m.layers[i](v)).reshape(-1,m.n,4);v=b.gather(-1,route[i][None,:,None].expand(len(b),-1,-1)).squeeze(-1)
 return m.head(v).reshape(*shape,10)
def prepared(m,x):
 b=torch.tanh(m.layers[0](x)).reshape(-1,m.n,4);route=m.routes();h=b.gather(-1,route[0][None,:,None].expand(len(b),-1,-1)).squeeze(-1);return b,h,route
@torch.no_grad()
def exact(m,x,y):
 b,h,route=prepared(m,x);logits=downstream(m,h,route);base=F.cross_entropy(logits,y,reduction='none')
 if m.depth==1:alt=logits[:,None,None,:]+(b-h[:,:,None])[:,:,:,None]*m.head.weight.T[None,:,None,:]
 else:
  changed=h[:,None,None,:]+(b-h[:,:,None])[:,:,:,None]*torch.eye(m.n)[None,:,None,:];alt=downstream(m,changed,route)
 errors=F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).flatten(),reduction='none').reshape(alt.shape[:-1]);return errors.mean(0)-base.mean()
def analytic(m,x,y):
 with torch.no_grad():b,h,route=prepared(m,x)
 h=h.detach().requires_grad_();loss=F.cross_entropy(downstream(m,h,route),y,reduction='sum');g=torch.autograd.grad(loss,h)[0];score=(g[:,:,None]*(b-h.detach()[:,:,None])).mean(0);return score.detach(),g.detach()
def hadamard(size):
 h=torch.ones(1,1)
 while len(h)<size:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
 return h
@torch.no_grad()
def perturb(m,x,y,k,sigma,seed,noise='gaussian',feedback='per_image'):
 b,h,route=prepared(m,x);gen=torch.Generator().manual_seed(seed)
 if noise=='gaussian':z=torch.randn(len(x),k,m.n,generator=gen)
 else:
  assert k>=m.n and k&(k-1)==0;sign=torch.randint(0,2,(len(x),1,m.n),generator=gen).float()*2-1;z=hadamard(k)[None,:,:m.n]*sign
 plus=downstream(m,h[:,None,:]+sigma*z,route);minus=downstream(m,h[:,None,:]-sigma*z,route);target=y[:,None].expand(-1,k).flatten();lp=F.cross_entropy(plus.reshape(-1,10),target,reduction='none').reshape(len(x),k);lm=F.cross_entropy(minus.reshape(-1,10),target,reduction='none').reshape(len(x),k);difference=(lp-lm)/(2*sigma)
 if feedback=='batch':difference=difference.mean(0)[None].expand(len(x),-1)*len(x)
 g=(difference[:,:,None]*z).mean(1);score=(g[:,:,None]*(b-h[:,:,None])).mean(0);return score,g

def metrics(score,g,truth,gradient):
 winner=score.argmin(-1);oracle=truth.argmin(-1);regret=truth[torch.arange(len(truth)),winner]-truth.min(-1).values
 return {'best_branch_agreement':float((winner==oracle).float().mean()),'mean_exact_regret_ce':float(regret.mean()),'chosen_exact_change_ce':float(truth[torch.arange(len(truth)),winner].mean()),'gradient_cosine':float(F.cosine_similarity(g.flatten()[None],gradient.flatten()[None]).item()),'gradient_relative_l2':float((g-gradient).norm()/gradient.norm().clamp_min(1e-12))}
def conditions():
 out=[]
 for feedback in ['per_image','batch']:
  for k in [1,4,16,64]:
   for sigma in [.01,.1,.5]:out.append({'kind':'perturb','noise':'gaussian','feedback':feedback,'k':k,'sigma':sigma})
 for sigma in [.01,.1,.5]:out.append({'kind':'perturb','noise':'orthogonal','feedback':'per_image','k':32,'sigma':sigma})
 return out

def balanced(x,y,seed,batch=100):
 gen=torch.Generator().manual_seed(seed);ids=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:batch//10]] for c in range(10)]);return x[ids],y[ids]
@torch.no_grad()
def gate_losses(m,responses,y,route):
 # Responses B,N,4; route K,N. Each K draws a complete jointly perturbed network route.
 h=responses[:,None].expand(-1,len(route),-1,-1).gather(-1,route[None,:,:,None].expand(len(responses),-1,-1,1)).squeeze(-1);log=m.head(h);loss=F.cross_entropy(log.reshape(-1,10),y[:,None].expand(-1,len(route)).flatten(),reduction='none').reshape(len(y),len(route)).mean(0);return loss
@torch.no_grad()
def gate_acc(m,responses,y,route):return float((m.head(responses[:,torch.arange(m.n),route]).argmax(-1)==y).float().mean())
@torch.no_grad()
def recover(m,x,y,qx,qy,method,k=64,sigma=1.,lr=.1,seed=0,steps=40):
 assert m.depth==1;b,_,r=prepared(m,x);qb,_,_=prepared(m,qx);route=r[0].clone();gate=torch.zeros(m.n,4);gate.scatter_(1,route[:,None],1.);gate.requires_grad_();opt=torch.optim.Adam([gate],lr=lr);gen=torch.Generator().manual_seed(seed);trace=[];flips=[]
 initial=gate_acc(m,qb,qy,route)
 for t in range(steps):
  if method=='perturb':
   z=torch.randn(k,m.n,4,generator=gen);rp=(gate[None]+sigma*z).argmax(-1);rm=(gate[None]-sigma*z).argmax(-1);lp=gate_losses(m,b,y,rp);lm=gate_losses(m,b,y,rm);g=((lp-lm)[:,None,None]*z).mean(0)/(2*sigma);opt.zero_grad(set_to_none=True);gate.grad=g;opt.step();route=gate.argmax(-1);flips.append(float((rp!=gate.detach().argmax(-1)[None]).float().mean()))
  elif method=='exact_coordinate':
   # Greedy one-coordinate route replacement, 100 candidates per step at width25.
   trial=route[None].expand(m.n*4,-1).clone();trial[torch.arange(m.n*4),torch.arange(m.n).repeat_interleave(4)]=torch.arange(4).repeat(m.n);losses=gate_losses(m,b,y,trial);route=trial[losses.argmin()]
  elif method=='no_gate_update':pass
  else:raise ValueError(method)
  if t in [0,9,19,steps-1]:trace.append({'step':t+1,'query_accuracy':gate_acc(m,qb,qy,route),'fit_ce':float(gate_losses(m,b,y,route[None])[0])})
 return {'method':method,'k':k if method=='perturb' else None,'sigma':sigma if method=='perturb' else None,'lr':lr if method=='perturb' else None,'initial_query_accuracy':initial,'query_accuracy':gate_acc(m,qb,qy,route),'fit_ce':float(gate_losses(m,b,y,route[None])[0]),'trace':trace,'perturbed_route_flip_fraction':st.mean(flips) if flips else 0.,'final_route':route.tolist()}

def verify():
 m=arch.make(7,'branch2_proxy');x=torch.rand(6,784);y=torch.arange(6);truth=exact(m,x,y)
 with torch.no_grad():b,h,r=prepared(m,x)
 baseline=F.cross_entropy(m(x),y)
 for n,k in [(0,2),(23,1)]:
  changed=r.clone();changed[0,n]=k;expected=F.cross_entropy(m(x,routes=changed),y)-baseline;assert torch.allclose(truth[n,k],expected,atol=1e-6)
 # With a linear loss, a complete orthogonal direction set recovers all coordinates exactly.
 h=hadamard(32)[:,:25];g=torch.randn(25);est=((h@g)[:,None]*h).mean(0);assert torch.allclose(est,g,atol=1e-6)
 # Hard routing has zero useful ordinary derivative with respect to selection logits.
 gate=torch.randn(25,4,requires_grad=True);branch=torch.randn(25,4);loss=branch.gather(-1,gate.argmax(-1)[:,None]).sum()+0*gate.sum();assert torch.equal(torch.autograd.grad(loss,gate)[0],torch.zeros_like(gate))
 return {'exact_first_layer_scores_match_changed_full_forward':True,'orthogonal_linear_loss_gradient_exact':True,'hard_argmax_gate_derivative_zero':True}

def profile():
 torch.set_num_threads(2);raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);x,y=balanced(x,y,559,100);selected=json.loads((OUT/'selection.json').read_text());rows=[]
 for depth in [1,2]:
  m=load(532,depth,'forward');settings=[{'kind':'exact'},{'kind':'analytic'}]+selected[str(depth)]
  for s in settings:
   def fn():
    if s['kind']=='exact':return exact(m,x,y)
    if s['kind']=='analytic':return analytic(m,x,y)
    return perturb(m,x,y,s['k'],s['sigma'],559,s['noise'],s['feedback'])
   for _ in range(3):fn()
   blocks=[]
   for _ in range(5):
    t=time.perf_counter()
    for _ in range(10):fn()
    blocks.append((time.perf_counter()-t)*1000/10)
   rows.append({'depth':depth,**s,'median_ms':st.median(blocks)})
 save(OUT/'profile.json',{'threads':2,'batch':100,'scope':'Credit computation only, includes first-layer response computation and downstream scoring. No optimizer. Fixed checkpoint,3warmups,5x10calls,freshprocess. Exact first-layer counterfactuals only; analytic gradients only through downstream network. Perturbations vectorized over draws;2K downstream evaluations per image. CPU only.','rows':rows})

def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--profile',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 if a.verify:print(json.dumps(verify()));return
 if a.profile:profile();return
 OUT.mkdir(parents=True,exist_ok=False);save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text())
 save(OUT/'protocol.json',{'purpose':'Test branch node perturbation credit quality and hard-gate recoverability before continual-learning training. Not full Dust reproduction.','checkpoints':'Existing same-loss proxy models,one/twohiddenlayers;seed531forwarddevelopment,532-535bothordersconfirmation. Training unchanged, no checkpoints selected by new outcomes.','credit':'First hidden layer. Exact single-neuron alternative CE is reference. Analytic local linear estimate g*(branch-current) compared with Gaussian antithetic response-noise gradient estimates and orthogonal32direction estimates. Gaussian1/4/16/64pairs, sigma.01/.1/.5; perimage versus shared batch loss. Responses perturbed AFTERtanh: scores estimated in response space, no weight updates; would require local derivative to update branchweights. Downstream hardroutes held fixed.','selection':'For eachdepth/noisefamily/feedback/K independentlyselectsigma on531forwardtwo balanced100imageTRAINbatches,3noise repetitions, minimizing exactbestbranchregret. No officialtest labels or confirmationselection. Allcandidatesretained.','confirmation':'532-535,bothorders,one/twohiddenlayers,twofreshbalanced100imageTRAINbatches/noise seeds each. Fourindependentcheckpointseeds;orders/batchespaired. Accuracy refers to branch-choice agreement,not imageclassification. Gradientcosine measuredagainstexactlocalderivative;exactcounterfactualreference includesfinitechanges,analyticdoesnot.','hardgate':'Separate supervisedrecoverydiagnostic: frozen one-layer finalweights,learn25x4gate-logits from256oldestpairlabeledTRAINimages,40steps,Gaussian64antitheticpairs. Argmax route. Devsigma.2/.5/1,lr.03/.1/.3 on531forward,disjointoldpairTRAINquery. Fresh532-535bothordersqueryofficialtestoldestpair; oldTRAINexamplesare deliberately revisited ONLY for this diagnostic, not claimedreplayfreeCL. Exactgreedycoordinate and no-update controls; smallsigma.05 negativecontrol. Oraclehistoricalroute reference.','limitations':'No automaticinputretrieval, no fullgradientfreeweightlearning, no claim of retentionbenefit. Frozenweightgatelearningtests credit through discrete decisions only. Feedbacklabelsareprovided. Multiple directions add compute; onejointperturbation is notonepopulationmemberperbranch.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
 dev=[];selected={}
 for depth in [1,2]:
  m=load(531,depth,'forward');before={k:v.clone() for k,v in m.state_dict().items()}
  for batch in range(2):
   bx,by=balanced(x,y,550+batch);truth=exact(m,bx,by);_,gradient=analytic(m,bx,by)
   for s in conditions():
    for repeat in range(3):
     score,g=perturb(m,bx,by,s['k'],s['sigma'],55000+batch*100+repeat,s['noise'],s['feedback']);dev.append({'depth':depth,'batch':batch,'repeat':repeat,**s,**metrics(score,g,truth,gradient)})
  selected[str(depth)]=[]
  keys=sorted(set((r['noise'],r['feedback'],r['k']) for r in dev if r['depth']==depth))
  for noise,feedback,k in keys:
   options=[s for s in conditions() if (s['noise'],s['feedback'],s['k'])==(noise,feedback,k)];best=min(options,key=lambda s:st.mean(r['mean_exact_regret_ce'] for r in dev if r['depth']==depth and all(r[a]==s[a] for a in s)));selected[str(depth)].append(best)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());print('DEV_CREDIT',depth,selected[str(depth)],flush=True)
 save(OUT/'development_credit.json',dev);save(OUT/'selection.json',selected);rows=[]
 for seed in range(532,536):
  for order in ['forward','reverse']:
   for depth in [1,2]:
    m=load(seed,depth,order);before={k:v.clone() for k,v in m.state_dict().items()}
    for batch in range(2):
     bx,by=balanced(x,y,seed*10+batch);truth=exact(m,bx,by);score,g=analytic(m,bx,by);rows.append({'seed':seed,'order':order,'depth':depth,'batch':batch,'kind':'analytic',**metrics(score,g,truth,g)})
     for s in selected[str(depth)]:
      score,estimate=perturb(m,bx,by,s['k'],s['sigma'],seed*100+batch,s['noise'],s['feedback']);rows.append({'seed':seed,'order':order,'depth':depth,'batch':batch,**s,**metrics(score,estimate,truth,g)})
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());save(OUT/'credit_results.json',rows)
   print('CREDIT_DONE',seed,order,flush=True)
 # Gate recovery: choices trained from old labels; this is explicitly a supervised diagnostic.
 m=load(531,1,'forward');ids=torch.nonzero(torch.isin(y,torch.tensor(SPLITS[0]))).flatten();gen=torch.Generator().manual_seed(550);ids=ids[torch.randperm(len(ids),generator=gen)];fitids=ids[:256];qids=ids[256:768];gate_dev=[]
 for sigma in [.2,.5,1.]:
  for lr in [.03,.1,.3]:
   r=recover(m,x[fitids],y[fitids],x[qids],y[qids],'perturb',sigma=sigma,lr=lr,seed=550);gate_dev.append(r);print('DEV_GATE',sigma,lr,round(r['query_accuracy']*100,2),flush=True)
 best=max(gate_dev,key=lambda r:(r['query_accuracy'],-r['fit_ce']));setting={k:best[k] for k in ['sigma','lr','k']};save(OUT/'gate_development.json',gate_dev);save(OUT/'gate_selection.json',setting);gates=[]
 for seed in range(532,536):
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   m=load(seed,1,name);before={k:v.clone() for k,v in m.state_dict().items()};ids=torch.nonzero(torch.isin(y,torch.tensor(order[0]))).flatten();gen=torch.Generator().manual_seed(seed*100);ids=ids[torch.randperm(len(ids),generator=gen)[:256]];qids=torch.nonzero(torch.isin(ty,torch.tensor(order[0]))).flatten();ck=torch.load(CK/f'{seed}_{name}_branch1_proxy.pt',weights_only=True);historic=torch.tensor(ck['routes'])[0,0]
   with torch.no_grad():b,_,_=prepared(m,tx[qids]);oracle=gate_acc(m,b,ty[qids],historic)
   conditions_gate=[('no_gate_update',{}),('exact_coordinate',{}),('perturb',setting),('perturb',{**setting,'sigma':.05})]
   for method,s in conditions_gate:
    r=recover(m,x[ids],y[ids],tx[qids],ty[qids],method,seed=seed*100+7,**s);r.update(seed=seed,order=name,historical_oracle_accuracy=oracle,small_noise_control=(s.get('sigma')==.05));gates.append(r)
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());save(OUT/'gate_results.json',gates);print('GATE_DONE',seed,name,[(r['method'],r.get('small_noise_control'),round(r['query_accuracy']*100,2)) for r in gates[-4:]],flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==320 and len(gates)==32,'credit_records':len(rows),'gate_records':len(gates),'development_credit_records':len(dev),'no_backbone_weights_changed':True})
if __name__=='__main__':main()
