"""Locked one-shot confirmation: learned local compatibility versus shuffled/no messages."""
import hashlib,json,statistics as st,argparse
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_pmi_message_pilot import Study,decode
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/pmi_confirmation')
def arrays(mem,direct):
 data={}
 for stable,mm in mem.items():
  data[f'stable{stable}_cooc']=mm.cooc
  for label,obj in [(f'd{d}',v) for d,v in mm.models.items()]+[('local',mm.local),('ridge',mm.ridge)]:
   for field,v in vars(obj).items():
    if isinstance(v,np.ndarray):data[f'stable{stable}_{label}_{field}']=v
 data['direct_cov']=direct.cov;data['direct_cross']=direct.cross;return data
def checksum(data):return hashlib.sha256(b''.join(v.tobytes() for v in data.values())).hexdigest()
@torch.no_grad()
def evaluate(m,mem,direct,bank,x,y,order):
 before={k:v.clone() for k,v in m.state_dict().items()};digest=checksum(arrays(mem,direct));q=np.concatenate([e.keys(b) for b in x.split(512)]);b=torch.cat([torch.tanh(m.layers[0](z)).reshape(-1,25,4) for z in x.split(512)]);yy=y.numpy()
 def metric(pred):
  p=[float((pred[np.isin(yy,pair)]==yy[np.isin(yy,pair)]).mean()) for pair in order];return {'accuracy':float((pred==yy).mean()),'pair_accuracy':p,'old_accuracy':st.mean(p[:-1]),'new_accuracy':p[-1]}
 def classify(route):return m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1).numpy()
 rows=[]
 for d,obj in mem[1].models.items():
  for method,gamma,scramble in [('no_message',0.,False),('learned_pmi',.01,False),('shuffled_pmi',.01,True)]:rows.append({'method':method,'dimension':d,**metric(classify(decode(obj,q,mem[1].cooc,gamma,scramble))),'extra_bytes':obj.size()+(2560 if gamma else 0)})
 rows.append({'method':'original_local',**metric(classify(mem[1].local.routes(q,0))),'extra_bytes':26000});rows.append({'method':'global_ridge',**metric(classify((q@mem[1].ridge.weight(.1).T).reshape(-1,25,4).argmax(-1))),'extra_bytes':41984});rows.append({'method':'direct_ridge',**metric((q@direct.weight(1.).T).argmax(-1)),'extra_bytes':direct.size()});ordinary=metric(classify(np.broadcast_to(m.routes()[0].numpy(),(len(x),25)).copy()));truth=np.zeros(len(y),np.int64)
 for i,pair in enumerate(order):truth[np.isin(yy,pair)]=i
 oracle=metric(classify(bank.numpy()[truth]));assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert checksum(arrays(mem,direct))==digest
 return rows,{'ordinary':ordinary,'oracle':oracle,'model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),'evaluation_state_unchanged':True,'bank_not_used_in_automatic_predictions':True}
def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["mnist","fashion_mnist"],default="mnist");parser.add_argument("--seed-start",type=int,default=602);args=parser.parse_args()
 if args.dataset!="mnist":OUT=OUT.parent/"pmi_fashion_transfer"
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();(OUT/'memories').mkdir();(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'dataset':args.dataset,'seed_start':args.seed_start,'transfer_note':'Fashion uses all settings locked from MNIST, no Fashion tuning; source-MNIST confirmation602-606 is already complete.','primary':'D64four-cue learnedPMI minus no-message and minus shuffledPMI. D16four-cue retained negativecontrol. Original local/globalridge/directclassridge matchedbackbonecontrols.','selection':'Development600/601 only, gamma.01 locked: smallestpositivegamma within0.1ppofbestmeanvalidation; gamma.05bestmeanonly0.035pphigher andlessconsistent. This rulechosenafterdevelopment,BEFOREofficialtestconfirmation. No furthergammaorstep changesallowed.','confirmation':'Fivefreshseeds from seed_start,bothorders,full60ktraining/full10ktest. 2currentepochs,batch128,Adam.003,one-layer25x4 CE/proxybackbone. All90automaticreadoutsretained. Four-cue prototypesD16decay.995,D64decay1 fixedfromearlierdevelopment.','communication':'Fixed5x5lattice40edges,PMIfromcurrentpreupdatebranchcooc,unitLaplacesmoothing,3rounds,temp.05,damping.5,gamma.01. Shuffledcontrolpermutes40potentialswithfixed600seed. No taskIDs/classlabels/boundaries/futurebankinassociationwrites or predictions. Current labels trainNNanddirectclassifier only.','memory':'40x4x4FP32counts2560B extra; latticegenerateddeterministically,no learnedtopology. PrototypeD1627200B/D64104000B. All are explicit associativeinput/statisticalmemory,notnomemoryclaim. No replay or externalconfigurationbank. No unique dendritic attribution orconferenceclaim.'})
 raw=load_raw(f'data_cache/{args.dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);e.Memory=Study;records=[]
 for seed in range(args.seed_start,args.seed_start+5):
  for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
   m,mem,direct,bank,audit,timing=e.train(seed,x,y,order);rr,diag=evaluate(m,mem,direct,bank,tx,ty,order);records.append({'seed':seed,'order':name,'methods':rr,'diagnostics':diag,'write_audit':audit,'all_candidate_timing':timing});e.save(OUT/'results.json',records);torch.save({'state_dict':m.state_dict(),'oracle_bank_only':bank},OUT/'models'/f'{seed}_{name}.pt');np.savez_compressed(OUT/'memories'/f'{seed}_{name}.npz',**arrays(mem,direct));print('RESULT',seed,name,[(r['method'],r.get('dimension'),round(100*r['accuracy'],2)) for r in rr],flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==10,'backbones':len(records),'automatic_readouts':sum(len(r['methods']) for r in records)})
if __name__=='__main__':main()
