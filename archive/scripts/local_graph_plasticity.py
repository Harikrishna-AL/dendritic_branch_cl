"""Protection, local teaching-gradient and allocation ablations; slowing-SGD control."""
import argparse,json,math,time,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_resource_benchmark import Branch,MLP,permutations,dump

class LocalVariant(Branch):
 def __init__(self,seed,s):
  super().__init__(seed,s);self.loss_rule=s.get('loss','mse');self.protection=s.get('protection','absolute');self.allocation=s.get('allocation','error');self.threshold=s.get('threshold',.4)
 def select(self):
  j=self.error.argmin(-1)
  if self.allocation=='novelty':
   unused=self.usage.reshape(self.n,self.k).argmin(-1)
   j=torch.where(self.error.min(-1).values>self.threshold,unused,j)
  return torch.arange(self.n)*self.k+j
 @torch.no_grad()
 def learn(self,x,y):
  z=self.data_features(x);response=torch.tanh(z@self.weight);target=self.codes[y].repeat_interleave(self.k,dim=1);active=self.select()
  residual=target[:,active]-response[:,active];reliable=(residual.abs()<.2).float();j=active%self.k
  self.links[torch.arange(self.n-1),j[:-1],j[1:]]+=(reliable[:,:-1]*reliable[:,1:]).sum(0)
  if self.protection=='off':rate=torch.full_like(self.usage[active],self.rate)
  else:
   use=self.usage[active]
   if self.protection=='relative':use=use-self.usage.reshape(self.n,self.k).min(-1).values
   rate=self.rate/(1+use/self.tau)
  # BCE with p=(1+tanh(a))/2 and target p*=(1+code)/2 has negative gradient code-response.
  direction=residual if self.loss_rule=='bce' else residual*(1-response[:,active].square())
  self.weight[:,active]+=(z.T@direction)/len(x)*rate[None]
  self.usage[active]+=reliable.sum(0)
  # Hold selection errors, reliable coactivity, codes and inference fixed to isolate update-gradient.
  self.error.copy_((response-target).square().mean(0).reshape(self.n,self.k))
  self.presentations+=len(x)

class SlowingMLP(MLP):
 def __init__(self,seed,s):super().__init__(seed,s);self.initial_lr=s['lr'];self.scale=s.get('decay',math.inf);self.presentations=0
 def learn(self,x,y):
  for g in self.opt.param_groups:g['lr']=self.initial_lr/(1+self.presentations/self.scale)
  super().learn(x,y);self.presentations+=len(x)

@torch.no_grad()
def evaluate(model,x,y,perms):
 return [float((torch.cat([model.predict(b[:,p]) for b in x.split(256)])==y).float().mean()) for p in perms]

def fit(seed,arm,s,x,y,mask,ps,ex,ey,checkpoints):
 model=LocalVariant(seed,s) if arm=='local' else SlowingMLP(seed,s);ids=torch.nonzero(mask,as_tuple=True)[0];rows=[];seconds=0.;acq=[]
 for domain,p in enumerate(ps):
  t=time.perf_counter()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):model.learn(x[ix][:,p],y[ix])
  seconds+=time.perf_counter()-t
  acq.append(evaluate(model,ex,ey,[p])[0])
  if domain+1 in checkpoints:
   a=evaluate(model,ex,ey,ps[:domain+1]);rows.append({'domains':domain+1,'accuracy':st.mean(a),'first_four':st.mean(a[:4]),'recent_four':st.mean(a[-4:]),'newest':a[-1],'old_accuracy':st.mean(a[:-1]),'domain_accuracy':a,'acquisition':acq.copy(),'training_seconds':seconds,'numeric_bytes':model.bytes()})
 return model,rows

def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.

def verify():
 torch.manual_seed(987);x=torch.rand(9,784);y=torch.arange(9)%10
 s={'rate':1.,'tau':10000.,'gated':True,'gamma':.1,'loss':'mse','protection':'absolute','allocation':'error'}
 b=Branch(6,s);v=LocalVariant(6,s)
 for _ in range(3):b.learn(x,y);v.learn(x,y)
 assert all(torch.equal(getattr(b,k),getattr(v,k)) for k in ['weight','codes','error','usage','links'])
 for loss in ['mse','bce']:
  m=LocalVariant(5,{**s,'loss':loss,'protection':'off'});z=m.data_features(x);w=m.weight.clone().requires_grad_();response=torch.tanh(z@w);active=m.select();t=m.codes[y].repeat_interleave(4,dim=1)[:,active]
  obj=.5*(response[:,active]-t).square().sum()/len(x) if loss=='mse' else torch.nn.functional.binary_cross_entropy((response[:,active]+1)/2,(t+1)/2,reduction='sum')/len(x)
  expected=w.detach()-torch.autograd.grad(obj,w)[0];m.learn(x,y);assert torch.allclose(m.weight,expected,atol=1e-7,rtol=1e-6)
 m=SlowingMLP(0,{'opt':'sgd','act':'relu','lr':.3,'decay':100000.});m.presentations=100000;m.learn(x,y);assert abs(m.opt.param_groups[0]['lr']-.15)<1e-10
 return {'baseline_variant_exactly_matches_original_training':True,'local_mse_and_bce_updates_match_autograd':True,'slowing_sgd_follows_presentation_based_schedule':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);dump(out/'verification.json',verify())
 source=json.loads(Path('results/local_graph_resource_benchmark_20261013/selection.json').read_text());rows=[];selections={}
 protocol={'architecture':'All local arms64neuronsx4branches,812480numericbytes, same fixedclasscodes, local coactivity, error trace and inference. No branch/input banks, replay, task IDs or resets. MLP256 SGD814120numericbytes plus a Python scalar presentation counter. Temporary storage excluded.', 'development':'New devseeds360/370;50kfit/10kheldout; fixedstratified2000validation subset;16permuted domains,2passes/currentdomain. Same seed/order across variants. No officialtest used for candidate selection.', 'local_ablations':['original','protection_off','bce_protected','bce_unprotected','novelty_allocation','relative_recycling'],'interpretation':'BCE changes only the local teaching-gradient; fixed random teaching codes are not removed. Allocation uses previous-minibatch local error threshold and existing branchusage. Relative recycling combines novelty allocation and damping relative to leastused branch per neuron; it is a package intervention, not a singlefactor isolation.', 'selection':'Select best local harmonic(meanall16domains,recent4accuracy), includingoriginal. Explicit success gate additionallyrequires newest+5pp, first4within5pp andallaveragewithin2pp oforiginalvalidation. MLPdecay100k/400kselected independently by meanall16accuracy; constantbaseline retained. All variants and failedgates archived.', 'confirmation':'Fresh361-363MNIST/371-373Fashion;32domains checkpoints4/8/16/32. Original, selectedlocal, constantMLP and selectedslowingMLP. Fixedhyperparameters; all60ktrain andfull10ktestatfinal32. Intermediatecurvesfixed2000testsubset; labelsonlymetrics at test. SlowingMLP lr=initial/(1+currentpresentations/decayscale), no taskboundary information.'}
 dump(out/'protocol.json',protocol);(out/'source.py').write_text(Path(__file__).read_text())
 for dataset,dev in [('mnist',360),('fashion_mnist',370)]:
  base=source[dataset+'_8']['branch']['chosen']['setting'];mlp=source[dataset+'_8']['sgd_mlp']['chosen']['setting']
  local={
   'original':{**base,'loss':'mse','protection':'absolute','allocation':'error'},
   'protection_off':{**base,'loss':'mse','protection':'off','allocation':'error'},
   'bce_protected':{**base,'loss':'bce','protection':'absolute','allocation':'error'},
   'bce_unprotected':{**base,'loss':'bce','protection':'off','allocation':'error'},
   'novelty_allocation':{**base,'loss':'mse','protection':'absolute','allocation':'novelty','threshold':.4},
   'relative_recycling':{**base,'loss':'mse','protection':'relative','allocation':'novelty','threshold':.4}}
  raw=load_raw('data_cache/'+dataset);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  rng=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;vid=val.reshape(10,1000)[:,:200].flatten();ps=permutations(dev,16);candidates=[]
  arms=[(name,'local',s) for name,s in local.items()]+[('mlp_constant','mlp',{**mlp,'decay':math.inf})]+[(f'mlp_decay_{int(d)}','mlp',{**mlp,'decay':d}) for d in [100000.,400000.]]
  for name,arm,s in arms:
   model,rs=fit(dev,arm,s,x,y,mask,ps,x[vid],y[vid],[4,8,16]);last=rs[-1];candidate={'name':name,'arm':arm,'setting':s,**last,'curve':rs,'score':harmonic(last['accuracy'],last['recent_four'])};candidates.append(candidate);dump(out/'development_progress.json',{'dataset':dataset,'candidates':candidates});print('dev',dataset,name,'all',round(last['accuracy']*100,2),'first4',round(last['first_four']*100,2),'new',round(last['newest']*100,2),flush=True)
  original=next(c for c in candidates if c['name']=='original');chosen=max([c for c in candidates if c['arm']=='local'],key=lambda c:c['score']);schedule=max([c for c in candidates if c['name'].startswith('mlp_decay')],key=lambda c:c['accuracy'])
  gate=chosen['newest']>=original['newest']+.05 and chosen['first_four']>=original['first_four']-.05 and chosen['accuracy']>=original['accuracy']-.02
  selections[dataset]={'local':chosen,'original':original,'mlp_schedule':schedule,'local_success_gate':gate,'candidates':candidates};dump(out/'selection.json',selections);print('selection',dataset,chosen['name'],'gate',gate,schedule['name'],flush=True)
  tests=[('original','local',local['original'])]
  if chosen['name']!='original':tests.append(('selected_local','local',chosen['setting']))
  tests+=[('mlp_constant','mlp',{**mlp,'decay':math.inf}),('mlp_slows','mlp',schedule['setting'])]
  tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)])
  for seed in range(dev+1,dev+4):
   ps=permutations(seed,32)
   for name,arm,s in tests:
    model,rs=fit(seed,arm,s,x,y,torch.ones(len(y),dtype=torch.bool),ps,tx[tid],ty[tid],[4,8,16,32]);full=evaluate(model,tx,ty,ps)
    for r in rs:rows.append({'dataset':dataset,'seed':seed,'name':name,'arm':arm,'setting':s,**r,'metric_scope':'stratified2000testsubset'})
    rows.append({'dataset':dataset,'seed':seed,'name':name,'arm':arm,'setting':s,'domains':32,'accuracy':st.mean(full),'first_four':st.mean(full[:4]),'recent_four':st.mean(full[-4:]),'newest':full[-1],'old_accuracy':st.mean(full[:-1]),'domain_accuracy':full,'numeric_bytes':model.bytes(),'metric_scope':'full10000test'})
    dump(out/'results.json',rows);print('test',dataset,seed,name,'all',round(st.mean(full)*100,2),'first4',round(st.mean(full[:4])*100,2),'new',round(full[-1]*100,2),flush=True)
 report(out,selections,rows)

def report(out,selections,rows):
 lines=['# Plasticity, allocation and local-gradient experiment','','Development16domains; independently selected local/scheduled controls confirmed on32domains with3freshpairedseeds. No testselection, taskIDs, replay or growingmodelcapacity. Localvariants keep same256responseunits andpersistentnumericstate.','', '| Development dataset | Variant | All accuracy | First4 | Newest |','|---|---|---:|---:|---:|']
 for ds,sel in selections.items():
  for c in sel['candidates']:lines.append(f'| {ds} | {c["name"]} | {100*c["accuracy"]:.2f}% | {100*c["first_four"]:.2f}% | {100*c["newest"]:.2f}% |')
 lines+=['','## Confirmation: full testset at32domains','','| Dataset | Model | All accuracy | First4 | Newest |','|---|---|---:|---:|---:|']
 for ds in selections:
  names=list(dict.fromkeys(r['name'] for r in rows if r['dataset']==ds))
  for name in names:
   rs=[r for r in rows if r['dataset']==ds and r['name']==name and r['metric_scope']=='full10000test'];m=lambda k:100*st.mean(r[k] for r in rs)
   lines.append(f'| {ds} | {name} | {m("accuracy"):.2f}% | {m("first_four"):.2f}% | {m("newest"):.2f}% |')
 lines+=['','BCE gradient comparison leaves fixed codes, selection errors, successful coactivity and inference unchanged. It does not test removal of coded supervision. Relative recycling changes both allocation and protection reference; interpret jointly. SlowingSGD is independently validation-selected; it may change both retention and acquisition. Full32-domain testnumbers are distinct from intermediate2000subsetcurves. No claim of biologicalmechanism, dendrite-specific superiority or conference readiness.'];(out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
