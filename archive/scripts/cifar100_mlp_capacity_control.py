"""Complete matched ordinaryMLP capacity/long-training controls for CIFARoracle pilot."""
import json,time,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.cifar100 import load
from bdh_cl.scripts.cifar100_oracle_routes import make,epoch,logits,metrics,save
OUT=Path('results/cifar100_mlp_capacity_control_20261014')
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);rows=[]
 train,valid,test,order=load('data_cache/cifar100');x=torch.tensor(train[0].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(train[1].copy(),dtype=torch.long);tx=torch.tensor(test[0].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(test[1].copy(),dtype=torch.long)
 save(OUT/'protocol.json',{'question':'MatchedparameterordinaryMLPjointcapacity and8-passsequentialreference fororiginalbranchCIFARoraclepilot.', 'model':'3072->93tanh->100,295189parametersvsbranch297508,about0.8%fewer. Samepixels/255,Adam.001,clip5,batch128,2or8learningpasses,45kfit/full10ktest,seeds491/492/493,onefixed2026classorder. No pretraining,CNN,augmentation,replay,orhyperparameterretuning.', 'matching':'Identicalnumbercurrentexamples/epochs,shuffle seeds,optimizer settings. MLP usesplainCE;branch usescounterfactualCEauxiliary. Differenthiddenoutputdim93vs24 despiteapproxparameterbudgetmatch;notisolatedarchitecture/objectiveablation.', 'oracle':'MLPcorrect-taskoutputmaskusestrueclassgrouponlyfortheexplicitassisteddiagnostic. NOTsameinterventionasbranchrouteoracle,whichdoesnotmask100outputs. PrimaryMLPpredictionalwaysall100classes. No equivalenthistoricalbranchconfigurationinplainMLP.', 'reference':'cifar100_oracle_routes_20261014 contains originalMLP2-passsequential3seeds,branch2/8sequential/joint,anduniform2pass. Thisfollowup addsMLP8sequentialandMLP2/8jointusingpairedsamecohort.'})
 (OUT/'source.py').write_text(Path(__file__).read_text())
 for seed in [491,492,493]:
  m=make(seed,'mlp');opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[];seconds=0.
  for task in range(10):
   mask=y//10==task;cx=x[mask];cy=y[mask];t=time.perf_counter()
   for e in range(8):epoch(m,opt,cx,cy,seed*100000+task*100+e)
   seconds+=time.perf_counter()-t
   with torch.no_grad():sel=ty//10==task;acq.append(float((logits(m,tx[sel]).argmax(-1)==ty[sel]).float().mean()))
  before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():
   scores=logits(m,tx);final=metrics(scores.argmax(-1),ty);mask=torch.arange(100)[None]//10!=ty[:,None]//10;assisted=metrics(scores.masked_fill(mask,-torch.inf).argmax(-1),ty)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());rows.append({'seed':seed,'arm':'mlp','mode':'sequential','epochs':8,'final':final,'oracle_task_output_mask':assisted,'acquisition':acq,'mean_acquisition':st.mean(acq),'training_seconds':seconds});save(OUT/'results.json',rows);print(seed,'sequential8','auto',round(final['accuracy']*100,2),'maskoracle',round(assisted['accuracy']*100,2),'acq',round(st.mean(acq)*100,2),flush=True)
  for epochs in [2,8]:
   m=make(seed,'mlp');opt=torch.optim.Adam(m.parameters(),lr=.001);t=time.perf_counter()
   for e in range(epochs):epoch(m,opt,x,y,seed*100000+9000+e)
   with torch.no_grad():final=metrics(logits(m,tx).argmax(-1),ty)
   rows.append({'seed':seed,'arm':'mlp','mode':'joint','epochs':epochs,'final':final,'training_seconds':time.perf_counter()-t});save(OUT/'results.json',rows);print(seed,'joint',epochs,round(final['accuracy']*100,2),flush=True)
 save(OUT/'completion.json',{'complete':len(rows)==9,'runs':len(rows)})
if __name__=='__main__':main()
