"""Unrestricted100-class oracle audit of original hard-gated branch design."""
import json,math,time,statistics as st
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.cifar100 import load
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier
OUT=Path('results/cifar100_oracle_routes_20261014')
def save(p,x):p.write_text(json.dumps(x,indent=2))
class CifarBranch(BranchClassifier):
 def __init__(self,arm):
  super().__init__(arm,inputs=3072,neurons=24,branches=4,gate='hard');self.head=nn.Linear(24,100);self.error.fill_(math.log(100))
def make(seed,arm):
 torch.manual_seed(seed)
 return nn.Sequential(nn.Linear(3072,93),nn.Tanh(),nn.Linear(93,100)) if arm=='mlp' else CifarBranch(arm)
def epoch(m,opt,x,y,seed):
 gen=torch.Generator().manual_seed(seed)
 for ix in torch.randperm(len(y),generator=gen).split(128):
  bx=x[ix];by=y[ix];opt.zero_grad(set_to_none=True)
  if isinstance(m,BranchClassifier):
   logits,candidate=m(bx,True);target=by[:,None,None].expand(candidate.shape[:-1]).flatten();error=F.cross_entropy(candidate.reshape(-1,100),target,reduction='none').reshape(candidate.shape[:-1]);loss=F.cross_entropy(logits,by)+.5*error.mean();local=error.detach().mean(0)
  else:loss=F.cross_entropy(m(bx),by)
  loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  if isinstance(m,BranchClassifier):m.observe(local)
@torch.no_grad()
def logits(m,x):return torch.cat([m(b) for b in x.split(512)])
@torch.no_grad()
def route_logits(m,x,routes):
 out=[]
 for bx in x.split(512):
  b=torch.tanh(m.first(bx)).reshape(-1,24,4);h=b[:,torch.arange(24)[None,:],routes];out.append(F.linear(h,m.head.weight,m.head.bias))
 return torch.cat(out)
def metrics(pred,y):return {'accuracy':float((pred==y).float().mean()),'task_accuracy':[float((pred[y//10==i]==y[y//10==i]).float().mean()) for i in range(10)]}
def verify():
 m=make(491,'local');x=torch.rand(7,3072);actual,candidate=m(x,True);b=torch.tanh(m.first(x)).reshape(-1,24,4);h=(b*m.gates()).sum(-1)
 for n,k in [(0,0),(9,2),(23,3)]:
  replaced=h.clone();replaced[:,n]=b[:,n,k];assert torch.allclose(candidate[:,n,k],m.head(replaced),atol=1e-6)
 assert actual.shape==(7,100)
 return {'counterfactuals_match_direct_neuron_replacement':True,'all_outputs_100_classes':True}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();save(OUT/'verification.json',verify());(OUT/'source.py').write_text(Path(__file__).read_text());rows=[]
 (train,valid,test,class_order)=load('data_cache/cifar100');x=torch.tensor(train[0].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(train[1].copy(),dtype=torch.long);vx=torch.tensor(valid[0].copy(),dtype=torch.float32).flatten(1)/255;vy=torch.tensor(valid[1].copy(),dtype=torch.long);tx=torch.tensor(test[0].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(test[1].copy(),dtype=torch.long)
 save(OUT/'protocol.json',{'question':'Does old-route recovery generalize fromSplitMNIST toSplitCIFAR100 inoriginalhardbranchmodel?', 'dataset':'OfficialCIFAR100archive,checksumvalidatedbyloader;45000fit/5000heldouttrain/10000officialtest. Seed2026fixedrandomclassorder;10successivetasksof10classes. ClassorderoriginalIDsrecorded.', 'class_order':class_order,'seeds':[491,492,493], 'architecture':'RGB3072pixels/255,24neurons4tanhbranches;one100-logithead. OriginalcounterfactualCEauxiliary.5,errorEMA.95,hardminimumerrorroute,Adam.001,clip5,batch128,noboundaryreset/replay/pretraining/augmentation. Full100classeseligiblealways,no outputmask.', 'arms':'Primary2passes/task:localbranches,uniformsamearchitecture/objective,matched93hiddenMLPplainCE. Secondarylocal8passes/task samearch/rule,notvalidation/testselected. Jointlocal2and8epochs capacities atsamegradientpresentationcountasrespective sequentialruns,notCL. Allnamedconditionspredeclared.', 'oracle':'Save24branchindicesaftereachtask;finalweightsandclassifierunchanged. Everyoldroutecomputesall100logitsforeverytestimage. Trueclassgroupselectscorrecthistoricalrouteonlyfororacleanalysis. NotautonomousorID-freeCLaccuracy.', 'controls':'Cyclicwrongrouteandunmaskedfinalroute. Uniformhasnocontrollablehistoricalroute;oracleidenticaltonormal. MLPfinalclassifierordinarybaseline;separateoracleknown-taskoutputmask labeledasassistedcontrol,notprimary100-classperformance.', 'acquisition':'Currenttaskaccuracyafteritslearning,unrestricted100-classpredictions,full1000currenttestimages;validationparallelscore. Oraclecurrentroutetestaccuracyattimeofacquisitioncomparedwithfinalrouteperformance. Pooracquisitionlimitsinterpretation.', 'limitations':'Smallfullyconnectedraw-pixelmodel,notCNN/pretrainedvisualencoder,notsotaCIFARbenchmark. Fixedhyperparametersnotindependentlytunedforthisdataset;3seedsoneclassorder. No claimthatfailureisolatesbranchmechanismifcapacity/acquisitionweak.'})
 for seed in [491,492,493]:
  for arm,epochs in [('local',2),('uniform',2),('mlp',2),('local',8)]:
   m=make(seed,arm);opt=torch.optim.Adam(m.parameters(),lr=.001);routes=[];acq=[];vacq=[];seconds=0.
   for task in range(10):
    ix=y//10==task;cx=x[ix];cy=y[ix];t=time.perf_counter()
    for e in range(epochs):epoch(m,opt,cx,cy,seed*100000+task*100+e)
    seconds+=time.perf_counter()-t
    with torch.no_grad():
     current=ty//10==task;acq.append(float((logits(m,tx[current]).argmax(-1)==ty[current]).float().mean()));vcurrent=vy//10==task;vacq.append(float((logits(m,vx[vcurrent]).argmax(-1)==vy[vcurrent]).float().mean()))
     if isinstance(m,BranchClassifier):routes.append(m.gates().argmax(-1).clone())
    print(seed,arm,epochs,'task',task+1,'acq',round(acq[-1]*100,2),flush=True)
   before={k:v.clone() for k,v in m.state_dict().items()};final=metrics(logits(m,tx).argmax(-1),ty);record={'seed':seed,'arm':arm,'epochs':epochs,'mode':'sequential','acquisition':acq,'validation_acquisition':vacq,'mean_acquisition':st.mean(acq),'final':final,'parameters':sum(p.numel() for p in m.parameters()),'model_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),'training_seconds':seconds}
   if arm=='local':
    rr=torch.stack(routes);all_logits=route_logits(m,tx,rr);assert all_logits.shape==(10000,10,100);pred=all_logits.argmax(-1);oracle=pred[torch.arange(len(ty)),ty//10];wrong=pred[torch.arange(len(ty)),(ty//10+1)%10];record.update(oracle=metrics(oracle,ty),wrong_route=metrics(wrong,ty),routes=rr.tolist(),route_bank_bytes=rr.numel()*rr.element_size(),route_by_task_accuracy=[[float((pred[ty//10==j,i]==ty[ty//10==j]).float().mean()) for j in range(10)] for i in range(10)])
    # Independentlyrestoreeachroute in originalforwardforcrosscheck.
    old=m.error.clone()
    for i in range(10):
     m.error.fill_(1);m.error.scatter_(1,rr[i,:,None],0);assert torch.equal(logits(m,tx).argmax(-1),pred[:,i])
    m.error.copy_(old)
   elif arm=='uniform':record.update(oracle=final,wrong_route=final,routes=[r.tolist() for r in routes])
   else:
    scores=logits(m,tx);mask=torch.arange(100)[None]//10!=ty[:,None]//10;record['oracle_task_output_mask']=metrics(scores.masked_fill(mask,-torch.inf).argmax(-1),ty)
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());torch.save({'state':m.state_dict(),'routes':routes},OUT/'models'/f'{seed}_{arm}_{epochs}epochs.pt');rows.append(record);save(OUT/'results.json',rows);print('FINAL',seed,arm,epochs,'automatic',round(final['accuracy']*100,2),'oracle',round(record.get('oracle',final)['accuracy']*100,2),flush=True)
  for epochs in [2,8]:
   m=make(seed,'local');opt=torch.optim.Adam(m.parameters(),lr=.001);t=time.perf_counter()
   for e in range(epochs):epoch(m,opt,x,y,seed*100000+9000+e)
   final=metrics(logits(m,tx).argmax(-1),ty);rows.append({'seed':seed,'arm':'local','epochs':epochs,'mode':'joint','final':final,'training_seconds':time.perf_counter()-t});save(OUT/'results.json',rows);print('JOINT',seed,epochs,round(final['accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==18,'runs':len(rows),'final_state_invariance_all_runs':True,'all_old_routes_crosschecked':True})
if __name__=='__main__':main()
