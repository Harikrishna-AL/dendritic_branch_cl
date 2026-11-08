"""Locked-settings independent confirmation of branch-local multimodal retrieval."""
import json,hashlib,statistics as st
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi,Direct
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/multicue_confirmation');ROOT=OUT.parent
SHAPES=[(16,1),(16,4),(32,1),(32,2),(64,1),(64,2),(64,4)]
SETTINGS=[]
class Study:
 def __init__(self):self.models={(s['dimension'],s['slots']):Multi(s['dimension'],s['slots'],s['decay']) for s in SETTINGS};self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.models.values():obj.update(q,route)
  self.local.update(q,route);t=np.zeros((len(q),25,4),np.float32);t[:,np.arange(25),route]=1;self.ridge.update(q,t.reshape(-1,100));self.accepted+=len(q)
def checksum(mem,direct):
 h=hashlib.sha256()
 for mm in mem.values():
  for obj in [*mm.models.values(),mm.local,mm.ridge]:
   for v in vars(obj).values():
    if isinstance(v,np.ndarray):h.update(v.tobytes())
 for obj in direct.values():h.update(obj.cov.tobytes());h.update(obj.cross.tobytes())
 return h.hexdigest()
@torch.no_grad()
def evaluate(m,mem,direct,bank,x,y,order):
 before={k:v.clone() for k,v in m.state_dict().items()};digest=checksum(mem,direct);q=np.concatenate([e.keys(b) for b in x.split(512)]);response=torch.cat([torch.tanh(m.layers[0](b)).reshape(-1,25,4) for b in x.split(512)]);yy=y.numpy()
 def metric(pred):
  pairs=[float((pred[np.isin(yy,p)]==yy[np.isin(yy,p)]).mean()) for p in order];return {'accuracy':float((pred==yy).mean()),'pair_accuracy':pairs,'old_accuracy':st.mean(pairs[:-1]),'new_accuracy':pairs[-1]}
 def predict(route):return m.head(response.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1).numpy()
 rows=[]
 for s in SETTINGS:
  obj=mem[s['stable']].models[(s['dimension'],s['slots'])];rows.append({'method':'multicue',**s,**metric(predict(obj.routes(q))),'extra_bytes':obj.size()})
 rows.append({'method':'original_local',**metric(predict(mem[1].local.routes(q,0))),'extra_bytes':26000});rows.append({'method':'global_ridge',**metric(predict((q@mem[1].ridge.weight(.1).T).reshape(-1,25,4).argmax(-1))),'extra_bytes':mem[1].ridge.size()})
 for d,obj in direct.items():rows.append({'method':'direct_ridge','dimension':d,**metric(obj.predict(q)),'extra_bytes':obj.cov.nbytes+obj.cross.nbytes})
 ordinary=metric(predict(np.broadcast_to(m.routes()[0].numpy(),(len(x),25)).copy()))
 # Label-based oracle constructed only after all automatic predictions above.
 true=np.zeros(len(y),np.int64)
 for i,p in enumerate(order):true[np.isin(yy,p)]=i
 oracle=metric(predict(bank.numpy()[true]));assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert checksum(mem,direct)==digest
 return rows,{'ordinary':ordinary,'oracle':oracle,'model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),'evaluation_state_unchanged':True,'bank_used_only_for_oracle':True}
def main():
 global SETTINGS
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();(OUT/'memories').mkdir();(OUT/'source.py').write_text(Path(__file__).read_text());dev=[json.loads((ROOT/name/'results.json').read_text())['rows'] for name in ['multicue_pilot','multicue_pilot_seed561']]
 for d,slots in SHAPES:
  candidates=[r for r in dev[0] if r.get('dimension')==d and r.get('slots')==slots];options=[]
  for a in candidates:
   b=next(r for r in dev[1] if all(r.get(k)==a.get(k) for k in ['method','dimension','slots','decay','stable']));options.append({'dimension':d,'slots':slots,'decay':a['decay'],'stable':a['stable'],'development_mean_accuracy':(a['accuracy']+b['accuracy'])/2})
  SETTINGS.append(max(options,key=lambda a:a['development_mean_accuracy']))
 e.save(OUT/'selection.json',SETTINGS);e.save(OUT/'protocol.json',{'scope':'Settings locked by meanvalidation seeds560/561 BEFORE officialtestconfirmation. Independent562-566bothorders,full60k/10k. Sevenpredeclaredcuecapacityshapes,decay/stabilitychosenper shape;allretained,notselectedbytest. Primary low-memory contrast D16S4 versusoriginalD64singleaverage,27200vs26000B. D32S2 alsofixedbudget26400B. D64S4highercapacity104000B. Backboneone-layer25x4,78760params+400Bstate,CE/proxyAdam.003,2currentepochs. Allassociationwritescurrentkeys/preupdatebranchactivityonly,noIDs/boundaries/labels/futurearchive/oldtaskreplay. Directclassifier alone usescurrentlabels initsmemory;NNsupervisedcurrentlabels. Alltenclassoutputs,noexternalbankinpredictions. Prototypesareexplicitinput-featurememory,notzeromemory. No claim ofuniquedendriticadvantage. e.load_raw returnsbothrawsplits,butdevelopmentdriversuseonlytrainingarrays;noofficialtestmetricsorlabelsselectsettings.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);original_step=e.backbone.step;e.Memory=Study;records=[]
 for seed in range(562,567):
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   direct={d:Direct(d) for d in [16,32,64]}
   def step(m,opt,bx,by,arm):
    q=e.keys(bx)
    for obj in direct.values():obj.update(q,by)
    return original_step(m,opt,bx,by,arm)
   e.backbone.step=step;m,mem,unused,bank,audit,timing=e.train(seed,x,y,order);rr,diag=evaluate(m,mem,direct,bank,tx,ty,order);records.append({'seed':seed,'order':name,'methods':rr,'diagnostics':diag,'write_audit':audit,'all_candidate_timing':timing});e.save(OUT/'results.json',records);torch.save({'state_dict':m.state_dict(),'oracle_bank_only':bank},OUT/'models'/f'{seed}_{name}.pt');arrays={}
   for stable,mm in mem.items():
    for label,obj in [(f'd{d}s{s}',v) for (d,s),v in mm.models.items()]+[('original_local',mm.local),('ridge',mm.ridge)]:
     for field,v in vars(obj).items():
      if isinstance(v,np.ndarray):arrays[f'stable{stable}_{label}_{field}']=v
   for d,obj in direct.items():arrays[f'direct{d}_cov']=obj.cov;arrays[f'direct{d}_cross']=obj.cross
   np.savez_compressed(OUT/'memories'/f'{seed}_{name}.npz',**arrays);print('RESULT',seed,name,[(r['method'],r.get('dimension'),r.get('slots'),round(100*r['accuracy'],2)) for r in rr],flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==10,'backbones':len(records),'automatic_readouts':sum(len(r['methods']) for r in records)})
if __name__=='__main__':main()
