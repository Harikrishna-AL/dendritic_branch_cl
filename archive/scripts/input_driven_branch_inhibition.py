"""End-to-end input-driven branches/inhibition with ordinary global Adam.

No usefulness router, route bank, predictor, recurrent search or auxiliary input
memory. A genuine soma nonlinearity is retained and explicitly ablated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS

ROOT=Path('results/input_driven_inhibition_20261026')
ARMS=['branch_full','branch_no_context','branch_no_competition','branch_linear_soma',
      'flat_group_gated','flat_global_gated','mlp','dense2']
LRS=[.001,.003];SPARSITY=[0.,.01]

def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False))

class Model(nn.Module):
 def __init__(self,seed,arm):
  super().__init__();torch.manual_seed(seed);self.arm=arm
  if arm in ['mlp','dense2']:
   width=101 if arm=='mlp' else 91
   self.input=nn.Linear(784,width)
   if arm=='dense2':self.middle=nn.Linear(width,width)
   self.head=nn.Linear(width,10)
  else:
   self.input=nn.Linear(784,100,bias=False)
   self.threshold=nn.Parameter(torch.zeros(100))
   if arm.startswith('branch'):
    self.gain=nn.Parameter(torch.full((25,4),.5));self.soma_bias=nn.Parameter(torch.zeros(25))
    self.head=nn.Linear(25,10)
   else:self.head=nn.Linear(100,10)
   if arm!='branch_no_context':
    self.inhibition=nn.Linear(16,100)
    nn.init.normal_(self.inhibition.weight,std=.01);nn.init.constant_(self.inhibition.bias,-3.)

 def forward(self,x,record=False):
  if self.arm in ['mlp','dense2']:
   activity=F.relu(self.input(x));h=activity
   if self.arm=='dense2':h=torch.tanh(self.middle(h))
   inhibition=torch.zeros_like(activity);competition=inhibition
  else:
   drive=self.input(x)-self.threshold
   q=F.normalize(F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),4).flatten(1),dim=-1)
   inhibition=F.softplus(self.inhibition(q)) if hasattr(self,'inhibition') else torch.full_like(drive,F.softplus(torch.tensor(-3.)).item())
   positive=F.relu(drive)
   if self.arm=='flat_global_gated':competition=(positive.sum(-1,keepdim=True)-positive)/99
   else:
    groups=positive.reshape(-1,25,4)
    competition=((groups.sum(-1,keepdim=True)-groups)/3).flatten(1)
   if self.arm=='branch_no_competition':competition=torch.zeros_like(competition)
   activity=F.relu(drive-inhibition-.5*competition)
   if self.arm.startswith('branch'):
    h=(activity.reshape(-1,25,4)*self.gain).sum(-1)+self.soma_bias
    if self.arm!='branch_linear_soma':h=torch.tanh(h)
   else:h=activity
  logits=self.head(h)
  return (logits,activity,inhibition,competition) if record else logits

def evaluate(m,x,y):
 with torch.no_grad():
  logits=[];active=0.;amplitude=0.;count=0
  for bx in x.split(512):
   z,r,_,_=m(bx,True);logits.append(z);active+=float((r>0).sum());amplitude+=float(r.sum());count+=r.numel()
  z=torch.cat(logits);pred=z.argmax(-1)
  return {'accuracy':float((pred==y).float().mean()),'loss':float(F.cross_entropy(z,y)),
          'digit_accuracy':[float((pred[y==c]==y[y==c]).float().mean()) for c in range(10)],
          'active_fraction':active/count,'activation_mean':amplitude/count}

def fit(seed,arm,lr,sparsity,x,y,tx,ty,order,joint=False):
 m=Model(seed,arm);opt=torch.optim.Adam(m.parameters(),lr=lr,foreach=False)
 trajectory=[];start=time.perf_counter();training_seconds=0.;steps=0;presentations=0
 groups=[tuple(range(10))] if joint else order
 for stage,pair in enumerate(groups):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):
    tick=time.perf_counter();opt.zero_grad(set_to_none=True)
    log,activity,_,_=m(x[ix],True)
    loss=F.cross_entropy(log,y[ix])+sparsity*activity.mean()
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    training_seconds+=time.perf_counter()-tick;steps+=1;presentations+=len(ix)
  trajectory.append(evaluate(m,tx,ty))
 final=trajectory[-1]
 def pair_acc(stats,pair):return st.mean(stats['digit_accuracy'][c] for c in pair)
 acquisition=st.mean(pair_acc(trajectory[i],p) for i,p in enumerate(order)) if not joint else final['accuracy']
 forgetting=st.mean(max(pair_acc(trajectory[j],p) for j in range(i,5))-pair_acc(final,p) for i,p in enumerate(order[:-1])) if not joint else 0.
 param=sum(p.numel() for p in m.parameters())
 states=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
 result={'seed':seed,'arm':arm,'lr':lr,'sparsity':sparsity,'joint':joint,
         'order':'joint' if joint else ('forward' if order==SPLITS else 'reverse'),
         'accuracy':final['accuracy'],'acquisition':acquisition,'forgetting':forgetting,
         'old_accuracy':st.mean(pair_acc(final,p) for p in order[:-1]),'new_accuracy':pair_acc(final,order[-1]),
         'active_fraction':final['active_fraction'],'trajectory':trajectory,'parameters':param,
         'inference_bytes':param*4,'training_persistent_bytes':param*4+states,
         'training_seconds':training_seconds,'all_seconds':time.perf_counter()-start,'steps':steps,'presentations':presentations}
 return result,m

def verify():
 torch.set_num_threads(2);x=torch.rand(5,784);y=torch.arange(5)
 m=Model(850,'branch_full');z,r,i,c=m(x,True)
 reconstructed=torch.tanh((r.reshape(-1,25,4)*m.gain).sum(-1)+m.soma_bias)
 assert torch.allclose(z,m.head(reconstructed),atol=1e-7)
 F.cross_entropy(z,y).backward()
 gradients={k:float(p.grad.abs().sum()) for k,p in m.named_parameters()}
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 assert gradients['inhibition.weight']>0 and gradients['input.weight']>0 and gradients['gain']>0
 assert torch.allclose(m(x)[:1],m(x[:1]),atol=1e-6)
 linear=Model(850,'branch_linear_soma');_,r,_,_=linear(x,True)
 flat_weight=(linear.head.weight[:,:,None]*linear.gain[None]).flatten(1)
 flat_bias=linear.head.bias+F.linear(linear.soma_bias,linear.head.weight)
 assert torch.allclose(linear(x),F.linear(r,flat_weight,flat_bias),atol=1e-6)
 before={k:v.clone() for k,v in m.state_dict().items()};m(x)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 return {'end_to_end_gradients_finite':True,'input_inhibition_gain_gradients_nonzero':True,
         'prediction_independent_of_batch_composition':True,'soma_formula_matches':True,
         'linear_soma_flat_equivalence_checked':True,'inference_no_state_writes':True,
         'parameters':{arm:sum(p.numel() for p in Model(850,arm).parameters()) for arm in ARMS}}

def dataset():
 raw=load_raw('data_cache/mnist')
 def make(split):return torch.tensor(raw[split+'_images'].copy(),dtype=torch.float32).flatten(1)/255,torch.tensor(raw[split+'_labels'].copy(),dtype=torch.long)
 return make('train'),make('test')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['development','confirmation']);args=parser.parse_args()
 torch.set_num_threads(2);ROOT.mkdir(exist_ok=True,parents=True)
 (x,y),(tx,ty)=dataset()
 if args.mode=='development':
  assert not (ROOT/'development.json').exists(),'Refusing overwrite'
  (ROOT/'source.py').write_text(Path(__file__).read_text());save(ROOT/'verification.json',verify())
  save(ROOT/'protocol.json',{
   'question':'Can globally optimized input-dependent branch inhibition yield automatic class-incremental retention without external retrieval?',
   'architecture':'25neuronsx4branches. r=ReLU(Wx-threshold-softplus(Vq+c)-.5meanotherpositive_drives). q=normalized4x4pooledinput. h=tanh(sum_b gain_b*r_b+soma_bias); all10classhead. Multiplebranchescancontribute. No history/contextstate, only currentinput.',
   'optimizer':'OrdinaryglobalAdam,CE+lambda meanfirstlayeractivation,clip5,batch128,2epochs/currentpair. No freezing, projection, speciallocallearning, optimizerreset, taskID/boundaryinput, replay, routebank, recurrentsearch, cue memory or outputmask.',
   'arms':ARMS,'controls':'no_context removes learnedIbutretainsfixedsoftplus(-3)andlateralcompetition;no_competitionremoveslateralterm;linear_somaremovestanh. Flatgroup keepssamecontext+4unitcompetition butfree100unithead;flatglobalcompetesacross100units. MLP101ReLU;dense2 has91ReLU->91tanh. Ordinarynetworksalso receive sameL1candidate grid.',
   'development':'Seeds850/851,stratified50kfit/10kTRAINvalidation,forwardorder. Independentlyselecteacharm sequential byharmonicmean(acquisition,finalaccuracy) andjoint byaccuracy. LR.001/.003 andL1 0/.01,all4settingsretained. No officialtestmetrics selected settings.',
   'confirmation':'Fresh852-856,bothorders,full60k/10k. All8selectedarms, plusjointreferenceonceperseed/arm. Primary fullbranchvsMLP,flatgroup,flatglobal; no-context/no-competition/soma ablations. Do nottuneofficialtests orsearchseeds.',
   'metrics':'all10accuracy,current-pairacquisition,earlier4accuracy,newest,forgetting,activationdensity,parameters,Adamstate,inference/traininglatency. Acquisition usesall10outputs oncorrespondingdigitpair. Currentlabels onlytrainCE; evaluationlabels onlymetrics.',
   'fairness':'Near-matchedparameters (maincontrolswithin1%);no-contextablationhas1700fewerparams. Allhavesamepresentations/epochs,notexactFLOPs. Sharedclassifierdriftremainspossible. No biological-local-learning or noveltyclaim fromglobalbackprop.',
   'interpretation':'Allnegativearmsretained. Anacquisitiongap>2ppcomplicatesretentioninterpretation;jointaccuracydifferencesreported. Needadvantagesoverflatcontrolsandablationsforstructuralcredit,notjustoveroneweakMLP.'})
  rows=[]
  for seed in [850,851]:
   rng=torch.Generator().manual_seed(seed)
   valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
   mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
   for arm in ARMS:
    for joint in [False,True]:
     for lr in LRS:
      for sparsity in SPARSITY:
       row,_=fit(seed,arm,lr,sparsity,x[mask],y[mask],x[valid],y[valid],SPLITS,joint)
       rows.append(row);save(ROOT/'development.json',rows)
     print('DEV',seed,arm,'joint',joint,[(r['lr'],r['sparsity'],round(100*r['accuracy'],2),round(100*r['acquisition'],2)) for r in rows[-4:]],flush=True)
  selected=[]
  for arm in ARMS:
   for joint in [False,True]:
    options=[]
    for lr in LRS:
     for sparsity in SPARSITY:
      rr=[r for r in rows if (r['arm'],r['joint'],r['lr'],r['sparsity'])==(arm,joint,lr,sparsity)]
      accuracy=st.mean(r['accuracy'] for r in rr);acq=st.mean(r['acquisition'] for r in rr)
      score=accuracy if joint else 2*accuracy*acq/(accuracy+acq)
      options.append({'arm':arm,'joint':joint,'lr':lr,'sparsity':sparsity,'score':score,'development_accuracy':accuracy,'development_acquisition':acq})
    selected.append(max(options,key=lambda r:(r['score'],-r['sparsity'],-r['lr'])))
  save(ROOT/'selection.json',selected);print('SELECTED',selected,flush=True)
 else:
  assert not (ROOT/'confirmation.json').exists(),'Refusing overwrite'
  selection=json.loads((ROOT/'selection.json').read_text())
  save(ROOT/'selection_lock.json',{'sha256':hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest(),'settings':selection})
  (ROOT/'models').mkdir(exist_ok=True);rows=[]
  for seed in range(852,857):
   for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
    for s in selection:
     if s['joint'] and name=='reverse':continue
     row,m=fit(seed,s['arm'],s['lr'],s['sparsity'],x,y,tx,ty,order,s['joint'])
     rows.append(row);torch.save(m.state_dict(),ROOT/'models'/f"{seed}_{row['order']}_{s['arm']}.pt")
     save(ROOT/'confirmation.json',rows)
    print('CONFIRM',seed,name,[(r['arm'],r['joint'],round(100*r['accuracy'],2),round(100*r['acquisition'],2)) for r in rows if r['seed']==seed and r['order'] in [name,'joint']],flush=True)
  save(ROOT/'completion.json',{'complete':len(rows)==120,'runs':len(rows),'sequential':80,'joint':40})

if __name__=='__main__':main()
