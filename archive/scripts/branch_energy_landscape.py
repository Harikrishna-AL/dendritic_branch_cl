"""Validation-first energy ranking, then conditional label-free gate descent.

No inference labels or historical-route bank in the autonomous candidate pool.
Masked predictors see only current examples, train on pixels, and never alter
the supervised backbone. Sufficient-statistic and forgetting-prone decoders
share the same linear hypothesis space.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts import controlled_branch_depth as backbone
from bdh_cl.scripts import online_proxy_branch_retrieval as old
from bdh_cl.scripts.overnight_multicue_pilot import compress, Direct
from bdh_cl.scripts.overnight_pmi_message_pilot import EDGES
from bdh_cl.data.mnist import load_raw,SPLITS

ROOT=Path('results/branch_energy_landscape_20261026')
METHODS=['confidence','masked_adam','masked_ridge','cue','ridge_cue','ridge_cue_pmi']
TARGET=torch.tensor([[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]])
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def tensors(raw):
 return torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255,torch.tensor(raw['train_labels'].copy(),dtype=torch.long)

def masks():
 result=torch.ones(4,1,28,28)
 for v in range(4):result[v,:,14*(v//2):14*(v//2+1),14*(v%2):14*(v%2+1)]=0
 return result.flatten(2).squeeze(1)
MASK=masks()

class Cue:
 def __init__(self):self.sums=np.zeros((25,4,16),np.float32);self.count=np.zeros((25,4),np.float32)
 def update(self,q,route):
  ix=np.arange(25);self.sums[ix,route]*=.995;self.sums[ix,route]+=q.sum(0);self.count[ix,route]*=.995;self.count[ix,route]+=len(q)
 def score(self,q):
  s=np.einsum('bd,nkd->bnk',q,old.norm(self.sums))
  return torch.tensor(np.where(self.count[None]>0,s,-1.))

class Predictor(nn.Module):
 def __init__(self):
  super().__init__();self.weight=nn.Parameter(torch.zeros(4,4,26))
 def forward(self,h):
  return torch.einsum('vbn,von->vbo',torch.cat([h,torch.ones(*h.shape[:-1],1)],-1),self.weight)

@torch.no_grad()
def observed(m,x):
 response=torch.tanh(m.layers[0](x)).reshape(-1,25,4)
 masked=(x[None]*MASK[:,None]).reshape(-1,784)
 hidden=torch.tanh(m.layers[0](masked)).reshape(4,len(x),25,4)
 pixels=F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),4).flatten(1)
 target=torch.stack([pixels[:,indices] for indices in TARGET])
 return response,hidden,target

def validation_ids(y,seed):
 rng=torch.Generator().manual_seed(seed)
 valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
 mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
 return mask,valid.reshape(10,1000)

def train(seed,x,y,out):
 m=backbone.make(seed,'branch1_proxy');opt=torch.optim.Adam(m.parameters(),lr=.003)
 decoder=Predictor();dop=torch.optim.Adam(decoder.parameters(),lr=.003)
 cov=np.broadcast_to(np.eye(26,dtype=np.float32),(4,26,26)).copy()
 cross=np.zeros((4,4,26),np.float32);cue=Cue();cooc=np.zeros((40,4,4),np.float32)
 direct=Direct(16);bank=[];steps=0;count=0;pixel_examples=0;start=time.perf_counter();decoder_seconds=0.;backbone_seconds=0.
 for stage,pair in enumerate(SPLITS):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):
    bx,by=x[ix],y[ix];route=m.routes()[0];q64=old.keys(bx)
    cue.update(compress(q64,16),route.numpy());direct.update(q64,by)
    cooc[np.arange(40),route.numpy()[EDGES[:,0]],route.numpy()[EDGES[:,1]]]+=len(bx)
    # Pixel-learning modules have no class-label or task argument.
    _,masked,target=observed(m,bx[:32])
    h=masked.gather(-1,route[None,None,:,None].expand(4,len(target[0]),-1,1)).squeeze(-1)
    features=torch.cat([h,torch.ones(4,len(h[0]),1)],-1).numpy()
    for v in range(4):cov[v]+=features[v].T@features[v];cross[v]+=target[v].numpy().T@features[v]
    t=time.perf_counter();dop.zero_grad(set_to_none=True);F.mse_loss(decoder(h),target).backward();dop.step();decoder_seconds+=time.perf_counter()-t
    t=time.perf_counter();backbone.step(m,opt,bx,by,'branch1_proxy');backbone_seconds+=time.perf_counter()-t
    steps+=1;count+=len(bx);pixel_examples+=len(h[0])
  bank.append(m.routes()[0].clone());print('STAGE',seed,stage+1,round(time.perf_counter()-start,1),flush=True)
 ridge=np.stack([np.linalg.solve(cov[v].astype(np.float64),cross[v].T.astype(np.float64)).T for v in range(4)]).astype(np.float32)
 state={'backbone':m.state_dict(),'adam_weight':decoder.weight.detach(),'ridge_weight':torch.tensor(ridge),
        'cue_sums':torch.tensor(cue.sums),'cue_count':torch.tensor(cue.count),'cooc':torch.tensor(cooc),
        'bank_diagnostic_only':torch.stack(bank)}
 torch.save(state,out/f'{seed}.pt')
 np.savez_compressed(out/f'{seed}_statistics.npz',cov=cov,cross=cross,direct_cov=direct.cov,direct_cross=direct.cross)
 return state,{'steps':steps,'backbone_presentations':count,'pixel_current_presentations':pixel_examples,
               'all_training_seconds':time.perf_counter()-start,'decoder_optimizer_seconds':decoder_seconds,
               'backbone_optimizer_seconds':backbone_seconds,'ridge_training_state_bytes':cov.nbytes+cross.nbytes,
               'adam_weights_bytes':decoder.weight.numel()*4,'cue_bytes':cue.sums.nbytes+cue.count.nbytes,'cooc_bytes':cooc.nbytes}

def context(state,x):
 m=backbone.make(0,'branch1_proxy');m.load_state_dict(state['backbone'])
 with torch.no_grad():
  clean=[];masked=[];target=[]
  for bx in x.split(256):
   c,z,t=observed(m,bx);clean.append(c);masked.append(z);target.append(t)
  clean=torch.cat(clean);masked=torch.cat(masked,1);target=torch.cat(target,1)
  q=torch.tensor(compress(np.concatenate([old.keys(bx) for bx in x.split(256)]),16))
  sums=state['cue_sums'];norm=sums/sums.norm(dim=-1,keepdim=True).clamp_min(1e-8)
  unary=torch.einsum('id,nkd->ink',q,norm)
  unary=torch.where(state['cue_count'][None]>0,unary,-1.)
  joint=state['cooc']+1;joint=joint/joint.sum((1,2),keepdim=True)
  pmi=(joint/(joint.sum(2,keepdim=True)*joint.sum(1,keepdim=True))).log()
 return {'m':m,'clean':clean,'masked':masked,'target':target,'unary':unary,'pmi':pmi,
         'adam':state['adam_weight'],'ridge':state['ridge_weight']}

def candidates(ctx,seed):
 n=len(ctx['clean']);rng=torch.Generator().manual_seed(seed)
 current=ctx['m'].routes()[0][None].expand(n,-1)
 local=ctx['unary'].argmax(-1)
 configs=[F.one_hot(current,4).float(),torch.full((n,25,4),.25),F.one_hot(local,4).float(),F.one_hot(current,4).float().softmax(-1)]
 for _ in range(24):configs.append(F.one_hot(torch.randint(4,(n,25),generator=rng),4).float())
 for probability in [.1,.3,.6,.9]:
  for _ in range(6):
   altered=torch.where(torch.rand(n,25,generator=rng)<probability,torch.randint(4,(n,25),generator=rng),current)
   configs.append(F.one_hot(altered,4).float())
 for _ in range(12):configs.append(torch.randn(n,25,4,generator=rng).softmax(-1))
 return torch.stack(configs,1)

def raw_energies(ctx,g):
 h=torch.einsum('vink,icnk->vicn',ctx['masked'],g)
 h=torch.cat([h,torch.ones(*h.shape[:-1],1)],-1)
 result={}
 for name in ['adam','ridge']:
  prediction=torch.einsum('vicn,von->vico',h,ctx[name])
  result[name]=(prediction-ctx['target'][:,:,None]).square().mean((0,3))
 result['cue']=-torch.einsum('ink,icnk->ic',ctx['unary'],g)/25
 pair=torch.zeros(g.shape[:2])
 for edge,(i,j) in enumerate(EDGES):pair-=torch.einsum('ick,kl,icl->ic',g[:,:,i],ctx['pmi'][edge],g[:,:,j])/40
 result['pmi']=pair
 logits=ctx['m'].head(torch.einsum('ink,icnk->icn',ctx['clean'],g))
 probability=logits.softmax(-1)
 result['confidence']=-(probability*probability.clamp_min(1e-8).log()).sum(-1)
 result['gate_entropy']=-(g*g.clamp_min(1e-8).log()).sum(-1).mean(-1)
 return result,logits

def standardized(raw,reference=None):
 if reference is None:reference={k:(raw[k].mean(1,keepdim=True),raw[k].std(1,keepdim=True).clamp_min(1e-5)) for k in ['adam','ridge','cue','pmi','confidence']}
 z={k:(raw[k]-center)/scale for k,(center,scale) in reference.items()}
 entropy=.05*raw['gate_entropy']
 energy={'confidence':z['confidence']+entropy,'masked_adam':z['adam']+entropy,'masked_ridge':z['ridge']+entropy,
         'cue':z['cue']+entropy,'ridge_cue':z['ridge']+z['cue']+entropy,'ridge_cue_pmi':z['ridge']+z['cue']+.25*z['pmi']+entropy}
 return energy,reference

def auc(correct,energy):
 result=[]
 for c,e in zip(correct,energy):
  if c.any() and (~c).any():
   delta=e[~c][None,:]-e[c][:,None]
   result.append(float(((delta>0)+.5*(delta==0)).mean()))
 return float(np.mean(result)) if result else None,len(result)

@torch.no_grad()
def ranking(state,x,y,seed):
 ctx=context(state,x);g=candidates(ctx,seed);raw,logits=raw_energies(ctx,g);energies,reference=standardized(raw)
 correct=(logits.argmax(-1)==y[:,None]).numpy();loss=-logits.log_softmax(-1).gather(-1,y[:,None,None].expand(-1,len(g[0]),1)).squeeze(-1).numpy()
 rows=[]
 for method,energy in energies.items():
  ee=energy.numpy();score,nauc=auc(correct,ee)
  erank=rankdata(ee,axis=1);lrank=rankdata(loss,axis=1)
  erank-=erank.mean(1,keepdims=True);lrank-=lrank.mean(1,keepdims=True)
  corr=(erank*lrank).sum(1)/np.maximum(np.linalg.norm(erank,axis=1)*np.linalg.norm(lrank,axis=1),1e-8)
  chosen=energy.argmin(1);pred=logits[torch.arange(len(y)),chosen].argmax(-1)
  rows.append({'method':method,'correctness_auc':score,'informative_images':nauc,'mean_spearman_energy_CE':float(corr.mean()),'selected_accuracy':float((pred==y).float().mean())})
 baseline={name:float(correct[:,i].mean()) for i,name in enumerate(['final_hard','uniform_soft','cue_onepass','final_soft'])}
 baseline['candidate_coverage_label_assisted_diagnostic']=float(correct.any(1).mean())
 # Distinct assisted diagnostic: all five externally saved routes are tried.
 bank=state['bank_diagnostic_only'];history=F.one_hot(bank,4).float()[None].expand(len(y),-1,-1,-1)
 rr,ll=raw_energies(ctx,history);ee,_=standardized(rr,reference);diagnostic={}
 for method,energy in ee.items():
  pred=ll[torch.arange(len(y)),energy.argmin(1)].argmax(-1)
  diagnostic[method]=float((pred==y).float().mean())
 true=torch.empty_like(y)
 for i,pair in enumerate(SPLITS):true[torch.isin(y,torch.tensor(pair))]=i
 baseline['historical_oracle_label_assisted']=float((ll[torch.arange(len(y)),true].argmax(-1)==y).float().mean())
 return {'seed':seed,'rows':rows,'baseline':baseline,'bank_assisted_energy_selection_diagnostic':diagnostic,'autonomous_candidate_count':g.shape[1]}

def project_simplex(v):
 u=v.sort(-1,descending=True).values;css=u.cumsum(-1)-1
 count=(u-css/torch.arange(1,5)>0).sum(-1).clamp_min(1)
 threshold=css.gather(-1,(count-1)[...,None]).squeeze(-1)/count
 return (v-threshold[...,None]).clamp_min(0)

def descent(state,x,method,initial):
 ctx=context(state,x)
 with torch.no_grad():
  pool=candidates(ctx,9230);raw,_=raw_energies(ctx,pool);_,reference=standardized(raw)
  route=ctx['m'].routes()[0][None].expand(len(x),-1) if initial=='final' else ctx['unary'].argmax(-1)
  g=torch.full((len(x),25,4),.25) if initial=='uniform' else F.one_hot(route,4).float().softmax(-1)
 def energy(gg):
  raw,_=raw_energies(ctx,gg[:,None]);return standardized(raw,reference)[0][method].squeeze(1)
 def prediction(gg,hard):
  if hard:gg=F.one_hot(gg.argmax(-1),4).float()
  return ctx['m'].head((ctx['clean']*gg).sum(-1)).argmax(-1).detach()
 curve=[];first=None;previous=None;start=time.perf_counter()
 for step in range(17):
  with torch.no_grad():
   value=energy(g)
   if first is None:first=value.clone()
   if previous is not None:assert bool((value<=previous+2e-5).all())
   if step in [0,1,4,8,16]:curve.append({'step':step,'energy_mean':float(value.mean()),'soft_prediction':prediction(g,False),'hard_prediction':prediction(g,True),'fraction_energy_decreased':float((value<first-1e-6).float().mean())})
   previous=value.clone()
  if step==16:break
  variable=g.detach().requires_grad_(True);gradient=torch.autograd.grad(energy(variable).sum(),variable)[0]
  with torch.no_grad():
   accepted=torch.zeros(len(g),dtype=torch.bool);next_g=g.clone();rate=torch.ones(len(g),1,1)
   for _ in range(12):
    proposal=project_simplex(g-rate*gradient);proposed=energy(proposal)
    take=(proposed<=value+1e-7)&(~accepted);next_g[take]=proposal[take];accepted|=take
    if bool(accepted.all()):break
    rate[~accepted]*=.5
   g=next_g
 return curve,time.perf_counter()-start

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['ranking','dynamics']);args=parser.parse_args()
 torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
 raw=load_raw('data_cache/mnist');x,y=tensors(raw)
 if args.mode=='ranking':
  assert not (ROOT/'ranking.json').exists(),'Refusing overwrite'
  (ROOT/'source.py').write_text(Path(__file__).read_text())
  save(ROOT/'protocol.json',{
   'purpose':'Validate input-conditioned energies BEFORE claiming energy descent retrieves skills.',
   'data':'Training arrays only: seeds830/831 stratified50kfit/10kheldout. Rank100heldout/class; separate dynamicsnext100/class. No official-test metrics or parameter/seed search.',
   'backbone':'Same25x4 proxybranch, CE,Adam.003,2epochs/currentdigitpair,batch128; labels onlycurrentbackbone and directbaseline. No model/taskboundary resets, replay or inference IDs.',
   'prediction':'Mask each14x14quadrant; samegateconfiguration across4maskedviews. Predict four raw4x4pooled pixels belonging to that quadrant from25selectedmaskedbranchresponses+bias. Four independent linear decoders416totalweights. Adam.003 vs full-rank ridge1.0 sufficientstats; current32examples/minibatch,pixels only. Sidecars do not alter backbone. No clean targetpixels in masked branch inputs.',
   'energies':METHODS,'composition':'Per-image z-scores of component energies over label-free64candidate pool; fixedscalesduringdescent. Composite ridge+cue; optional.25PMI. All add.05mean gateentropy to encourage discretegates; no coefficient tuning. Cue16Dmean .995activityassociation; grid40edges coactivationcounts/Laplace1.',
   'candidates':'64:finalhard,uniformsoft,cuehard,finalsoft;24randomhard;24currentperturbations(.1/.3/.6/.9);12randomsoft. NOhistoricalroutes. Samepool forallenergies. Historical5routes used only in separately named bank-assisted diagnostic.',
   'metrics':'Within-imagecorrect-vs-wrongenergyAUC,tieshalf;SpearmanenergyvsCE;energy-selectedaccuracy;coverageoracle diagnostic. Labels usedONLYafterenergies/predictions for analysis, never energyupdates.',
   'qualification':'A novel prediction/composite energy qualifies for dynamics if BOTHdevelopment seeds AUC>=.60 andselectedaccuracy>=finalhard+5pp. Composite additionallymustbeatdirectcueonepassby1pp BOTHseeds. Choose highestmeanselectedaccuracy, ties lower memory. Confidence/cue are baselines, notnovelqualifiers. Ifnonequalifiesstopbeforeofficialtests/descent.',
   'dynamics':'Onlyifqualified: onechosenenergy,disjoint100heldout/class,16projectedsimplexgradientsteps,per-example12halvingbacktrackingsteps,eta1;uniformprimary,finalsoft/cuesoftcontrols. Frozenbackbone/decoders,no labels in descent. Reportsoft/hardaccuracy after0/1/4/8/16 andenergydecrease. No convergence/correctness guarantee.',
   'confirmation_gate':'Freshconfirmationnotautomatic: requiredisjointvalidationuniformharddescentaccuracy>=cueonepass+1ppbothseeds andactualimprovementfromstep0; otherwise reportboundedfailure. Preserveallnegativepilots. No officialtesttuning.',
   'memory':'Cue6800B,PMI2560B,ridgestats12480B,inference416weights1664B(derivedcache countedseparately). Adam416weights1664Bplusmoments duringtraining. Masked4views addcompute;64candidate screeningalsoexpensive. Not memory/compute-equivalent toMLP,notbiologicallocalclaim.'})
  records=[]
  for seed in [830,831]:
   mask,valid=validation_ids(y,seed);state,timing=train(seed,x[mask],y[mask],ROOT)
   rankids=valid[:,:100].flatten();record=ranking(state,x[rankids],y[rankids],seed+30000);record['training_seed']=seed;record['timing']=timing
   records.append(record);save(ROOT/'ranking.json',records)
   print('RANK',seed,json.dumps(record),flush=True)
  qualified=[]
  for method in ['masked_adam','masked_ridge','ridge_cue','ridge_cue_pmi']:
   rr=[next(r for r in record['rows'] if r['method']==method) for record in records]
   gate=all(r['correctness_auc'] is not None and r['correctness_auc']>=.60 and r['selected_accuracy']>=record['baseline']['final_hard']+.05 for r,record in zip(rr,records))
   if method.startswith('ridge_cue'):gate=gate and all(r['selected_accuracy']>=record['baseline']['cue_onepass']+.01 for r,record in zip(rr,records))
   if gate:qualified.append({'method':method,'mean_accuracy':st.mean(r['selected_accuracy'] for r in rr)})
  chosen=max(qualified,key=lambda r:r['mean_accuracy'])['method'] if qualified else None
  save(ROOT/'qualification.json',{'qualified':qualified,'chosen':chosen,'decision':'run_disjoint_validation_dynamics' if chosen else 'stop_no_useful_novel_energy','no_official_test_metrics':True})
  print('QUALIFICATION',chosen,flush=True)
 else:
  assert not (ROOT/'dynamics.json').exists(),'Refusing overwrite'
  chosen=json.loads((ROOT/'qualification.json').read_text())['chosen'];assert chosen,'No energy passed the predeclared qualification'
  (ROOT/'dynamics_source.py').write_text(Path(__file__).read_text());records=[]
  for seed in [830,831]:
   _,valid=validation_ids(y,seed);ids=valid[:,100:200].flatten();state=torch.load(ROOT/f'{seed}.pt',weights_only=True)
   for initial in ['uniform','final','cue']:
    curve,seconds=descent(state,x[ids],chosen,initial)
    for row in curve:
     for field in ['soft_prediction','hard_prediction']:
      pred=row.pop(field);row[field.replace('prediction','accuracy')]=float((pred==y[ids]).float().mean())
      row[field.replace('prediction','pair_accuracy')]=[float((pred[torch.isin(y[ids],torch.tensor(pair))]==y[ids][torch.isin(y[ids],torch.tensor(pair))]).float().mean()) for pair in SPLITS]
    records.append({'seed':seed,'method':chosen,'initial':initial,'curve':curve,'seconds':seconds})
    save(ROOT/'dynamics.json',records);print('DYNAMICS',seed,initial,curve,flush=True)
  save(ROOT/'dynamics_completion.json',{'complete':len(records)==6,'official_test_used':False})

if __name__=='__main__':main()
