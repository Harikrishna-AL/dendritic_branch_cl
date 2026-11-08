"""Exploratory no-archive decoding, with frozen primary settings and directRLS control."""
import json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.online_config_memory_benchmark import OUT,KEYS,train,keys,normalize,save,D
FOLLOW=OUT/'no_bank_followup'
def local_routes(m,q,gamma):
 unary=np.einsum('bd,nkd->bnk',q,normalize(m.sums));unary=np.where(m.count[None]>0,unary,-1e3)
 value=unary/.05;value-=value.max(-1,keepdims=True);p=np.exp(value);p/=p.sum(-1,keepdims=True)
 if gamma:
  right=m.links/np.maximum(m.links.sum(2,keepdims=True),1);left=m.links/np.maximum(m.links.sum(1,keepdims=True),1)
  for _ in range(3):
   message=.5*(np.einsum('bnl,nkl->bnk',np.roll(p,-1,axis=1),right)+np.einsum('bnk,nkl->bnl',np.roll(p,1,axis=1),np.roll(left,1,axis=0)));value=(unary+gamma*message)/.05;value-=value.max(-1,keepdims=True);p=np.exp(value);p/=p.sum(-1,keepdims=True)
 return p.argmax(-1)
def winners(mem,q,s):
 method=s['method']
 if method.startswith('local'):return local_routes(mem.local,q,s['gamma'])
 if method.startswith('central'):
  m=getattr(mem,method);used=m.count>0;return m.routes[used][(q@normalize(m.mean[used]).T).argmax(-1)].astype(np.int64)
 m=mem.rls;weight=np.linalg.solve(m.cov.astype(np.float64),m.cross.T.astype(np.float64)).T;return (q@weight.T).reshape(-1,24,4).argmax(-1)
@torch.no_grad()
def evaluate(model,initial,mem,bank,tx,ty,order,settings):
 qp={k:[] for k in KEYS};responses=[]
 for bx in tx.split(512):
  q=keys(bx,model,initial)
  for k in KEYS:qp[k].append(q[k])
  responses.append(torch.tanh(model.first(bx)).reshape(-1,24,4))
 q={k:np.concatenate(v) for k,v in qp.items()};response=torch.cat(responses);yy=ty.numpy();true=np.zeros(len(yy),np.int64)
 for i,p in enumerate(order):true[np.isin(yy,p)]=i
 result=[]
 for s in settings:
  chosen=winners(mem[(s['key'],s['stable'])],q[s['key']],s);hidden=response.gather(-1,torch.tensor(chosen.astype(np.int64))[:,:,None]).squeeze(-1);pred=model.head(hidden).argmax(-1).numpy();pair=[float((pred[np.isin(yy,p)]==yy[np.isin(yy,p)]).mean()) for p in order];result.append({**s,'accuracy':float((pred==yy).mean()),'pair_accuracy':pair,'old_accuracy':float(np.mean(pair[:-1])),'new_accuracy':pair[-1],'exact_historical_route_match_analysis':float((chosen==bank[true]).all(-1).mean())})
 return result
@torch.no_grad()
def direct_rls(seed,x,y,tx,ty,order,model,initial):
 cov=np.eye(D,dtype=np.float32);cross=np.zeros((10,D),np.float32)
 for stage,pair in enumerate(order):
  ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
  for epoch in range(2):
   gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
    q=keys(x[ix],model,initial)['fixed_pixels'];cov+=q.T@q
    for c in y[ix].unique():cross[int(c)]+=q[(y[ix]==c).numpy()].sum(0)
 weight=np.linalg.solve(cov.astype(np.float64),cross.T.astype(np.float64)).T;q=np.concatenate([keys(b,model,initial)['fixed_pixels'] for b in tx.split(512)]);pred=(q@weight.T).argmax(-1);yy=ty.numpy();pairs=[float((pred[np.isin(yy,p)]==yy[np.isin(yy,p)]).mean()) for p in order]
 return {'accuracy':float((pred==yy).mean()),'pair_accuracy':pairs,'old_accuracy':float(np.mean(pairs[:-1])),'new_accuracy':pairs[-1],'numeric_bytes':cov.nbytes+cross.nbytes}
def main():
 torch.set_num_threads(2);assert json.loads((OUT/'completion.json').read_text())['complete'];FOLLOW.mkdir(parents=True,exist_ok=False);(FOLLOW/'memories').mkdir();settings=json.loads((OUT/'selection.json').read_text());records=[]
 save(FOLLOW/'protocol.json',{'exploratory_followup':True,'question':'Does onlineassociationreadoutrequireexternallysavedfiveconfigurationbank?', 'settings':'Primarydevelopmentselectedsettingsunchanged;no retuningonnewdecode orofficialtests. Same6backbones/seedorders,notindependentconfirmation. Allconditionsreported.', 'inference':'No externalhistoricalroutesenterpredictions:localargmaxperneuronwith/withoutmessages;RLS96branchvaluesargmaxperneuron;centralnearestonlinecentroidreturnsitscontemporaneouslyobservedroute. Bankonlyhistoricalmatchdiagnostic. Memorywritesstillcurrentkeys/preupdatebrancheswithoutlabels/taskIDs/boundaries.', 'control':'DirectclassRLSusingidenticalfixed64pixelkeys,ridge1,currentdigitlabelonehot10. Same2currentpasses,order/seeds,nooldimage replay. C/Bstats18944bytes,noNNrequiredatdeployment. Moreexplicitlabelteachingthanbranchactivitytarget,validpredictivebaselinebutnotpurearchitectureablation.', 'audit':'Backbonetrainingrepeatedonlytoobtainonlineassociationstatesandnewdecode;mustexactlymatchprimarysavedbackbones. Saveallassociationarrays forreproducibility. Historicalbankexternalonlyoraclediagnostic,notusedbydecode.'})
 (FOLLOW/'source.py').write_text(Path(__file__).read_text());raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
 for seed in [511,512,513]:
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   model,initial,mem,bank,audit,seconds=train(seed,x,y,torch.ones(len(y),dtype=torch.bool),order);saved=torch.load(OUT/'models'/f'{seed}_{name}.pt',weights_only=True)['state'];assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items());before={k:v.clone() for k,v in model.state_dict().items()};rr=evaluate(model,initial,mem,bank,tx,ty,order,settings);control=direct_rls(seed,x,y,tx,ty,order,model,initial);assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
   arrays={}
   for (k,g),mm in mem.items():
    for member in ['local','central96','central144','rls']:
     for field,value in vars(getattr(mm,member)).items():
      if isinstance(value,np.ndarray):arrays[f'{k}_stable{g}_{member}_{field}']=value
   np.savez_compressed(FOLLOW/'memories'/f'{seed}_{name}.npz',**arrays);torch.save({'weight':initial[0],'bias':initial[1]},FOLLOW/'memories'/f'{seed}_{name}_initial_encoder.pt');records.append({'seed':seed,'order':name,'methods':rr,'direct_class_rls':control});save(FOLLOW/'results.json',records);print('NO_BANK',seed,name,[(z['key'],z['method'],round(z['accuracy']*100,2)) for z in rr],'direct',round(control['accuracy']*100,2),flush=True)
 save(FOLLOW/'completion.json',{'complete':len(records)==6,'readouts':sum(len(r['methods']) for r in records),'all_repeated_backbones_match_primary_exactly':True})
if __name__=='__main__':main()
