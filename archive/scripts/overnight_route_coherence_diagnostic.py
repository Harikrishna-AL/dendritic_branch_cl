"""Training-holdout routing diagnosis; bank projections are assisted diagnostics only."""
import json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi,compress
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/coherence_diagnostic')
class Study:
 def __init__(self):self.models={(16,4):Multi(16,4,.995),(64,4):Multi(64,4,1.)};self.accepted=0
 def update(self,q,route):
  for obj in self.models.values():obj.update(q,route)
  self.accepted+=len(q)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'scope':'Analysis-only onTRAINholdouts ofexistingdevelopment560/561. Noofficialtestornewhyperparameterselection. Samefrozenfinalweights; automaticlocalroutes versuslabel-oraclehistoricalroute. Nearest-Hammingandcue-scoreprojection useexternallyrecordedfivephase-endconfigs and are explicitlyBANK-ASSISTEDdiagnostics,notautonomous methods. No labels enterprojectionselection; labels onlyanalysis. Cue assignment memory never seesbank/boundaries. Diagnosis asks whether cross-neuronrouteincoherence alone limits retrieval. No improvementclaim basedonassistedconditions.'})
 raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);e.Memory=Study;records=[]
 for seed in [560,561]:
  g=torch.Generator().manual_seed(seed);valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=g)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;vx,vy=x[valid],y[valid];m,mem,direct,bank,audit,timing=e.train(seed,x[mask],y[mask],SPLITS)
  with torch.no_grad():
   q=np.concatenate([e.keys(b) for b in vx.split(512)]);b=torch.cat([torch.tanh(m.layers[0](z)).reshape(-1,25,4) for z in vx.split(512)]);yy=vy.numpy();truth=np.zeros(len(yy),np.int64)
   for i,pair in enumerate(SPLITS):truth[np.isin(yy,pair)]=i
   bank=bank.numpy()
   def classify(route):return m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1).numpy()
   oracle=classify(bank[truth]);rows=[]
   for (d,s),obj in mem[1].models.items():
    route=obj.routes(q);pred=classify(route);distance=(route[:,None]!=bank[None]).sum(-1);hammingchoice=distance.argmin(-1);cq=compress(q,d);score=np.einsum('bd,nksd->bnks',cq,e.norm(obj.sums));score=np.where(obj.count[None]>0,score,-1e3).max(-1);bankscores=np.stack([score[:,np.arange(25),pattern].sum(-1) for pattern in bank],-1);cuechoice=bankscores.argmax(-1);correct=pred==yy;agreement=(route==bank[truth]).mean(-1);bins=[]
    for lo,hi in [(0,0),(1,3),(4,6),(7,25)]:
     ids=(distance.min(-1)>=lo)&(distance.min(-1)<=hi);bins.append({'min_snapshot_distance':[lo,hi],'images':int(ids.sum()),'accuracy':float(correct[ids].mean()) if ids.any() else None})
    rows.append({'dimension':d,'slots':s,'automatic_accuracy':float(correct.mean()),'oracle_accuracy':float((oracle==yy).mean()),'assisted_hamming_projection_accuracy':float((classify(bank[hammingchoice])==yy).mean()),'assisted_cuescore_bank_accuracy':float((classify(bank[cuechoice])==yy).mean()),'branch_agreement_with_true_snapshot':float(agreement.mean()),'branch_agreement_correct_images':float(agreement[correct].mean()),'branch_agreement_wrong_images':float(agreement[~correct].mean()),'whole_snapshot_match':float((distance.min(-1)==0).mean()),'mean_nearest_snapshot_distance':float(distance.min(-1).mean()),'oracle_correct_auto_wrong':float(((oracle==yy)&~correct).mean()),'auto_correct_oracle_wrong':float((correct&(oracle!=yy)).mean()),'distance_bins':bins})
   records.append({'seed':seed,'validation_only':True,'rows':rows});e.save(OUT/'results.json',records);print('RESULT',seed,rows,flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==2,'diagnostic_conditions':4,'official_test_used':False})
if __name__=='__main__':main()
