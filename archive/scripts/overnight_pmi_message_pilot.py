"""Validation-only bounded test of local PMI compatibility on a fixed lattice."""
import json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.scripts.overnight_multicue_pilot import Multi,compress
from bdh_cl.data.mnist import load_raw,SPLITS
OUT=Path('results/overnight_branch_research_20261024/pmi_message_pilot');EDGES=np.array([(n,n+1) for n in range(25) if n%5<4]+[(n,n+5) for n in range(20)])
class Study:
 def __init__(self):self.models={16:Multi(16,4,.995),64:Multi(64,4,1.)};self.cooc=np.zeros((40,4,4),np.float32);self.local=e.Local(.995);self.ridge=e.Ridge(100);self.accepted=0
 def update(self,q,route):
  for obj in self.models.values():obj.update(q,route)
  self.cooc[np.arange(40),route[EDGES[:,0]],route[EDGES[:,1]]]+=len(q);self.local.update(q,route);target=np.zeros((len(q),25,4),np.float32);target[:,np.arange(25),route]=1;self.ridge.update(q,target.reshape(-1,100));self.accepted+=len(q)
def decode(obj,q,counts,gamma,scramble):
 q=compress(q,obj.d);unary=np.einsum('bd,nksd->bnks',q,e.norm(obj.sums));unary=np.where(obj.count[None]>0,unary,-1e3).max(-1)
 def softmax(v):
  v=v/.05;v-=v.max(-1,keepdims=True);p=np.exp(v);return p/p.sum(-1,keepdims=True)
 p=softmax(unary)
 if not gamma:return p.argmax(-1)
 joint=(counts+1)/(counts+1).sum((1,2),keepdims=True);pmi=np.log(joint/(joint.sum(2,keepdims=True)*joint.sum(1,keepdims=True)))
 if scramble:pmi=pmi[np.random.default_rng(600).permutation(40)]
 degree=np.bincount(EDGES.flatten(),minlength=25)
 for _ in range(3):
  msg=np.zeros_like(p)
  for edge,(i,j) in enumerate(EDGES):msg[:,i]+=p[:,j]@pmi[edge].T;msg[:,j]+=p[:,i]@pmi[edge]
  proposed=softmax(unary+gamma*msg/degree[None,:,None]);p=.5*p+.5*proposed
 return p.argmax(-1)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());e.save(OUT/'protocol.json',{'scope':'Fresh600/601development50kfit/10kheldoutTRAINvalidation,zeroofficialtestmetrics. Fixed5x5lattice40edges,currentpreupdateglobalroutecooccounts40x4x4float=2560B plusdeterministicedges. PMIwithLaplace1,3meanfieldrounds,damping.5,temp.05;gamma0/.01/.05/.2 bounded,plusfixededgepotentialpermutationcontrol. SameD16S4decay.995andD64S4decay1prototype. No taskIDs/boundaries/labels/futurearchive inroutingwrites/inference. Countadditionalpairwise memorystateexplicitly. Allcandidatesretained;small/inconsistentgainnotconfirmation-worthy.'})
 raw=load_raw('data_cache/mnist');allx=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;ally=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);e.Memory=Study;records=[]
 for seed in [600,601]:
  g=torch.Generator().manual_seed(seed);valid=torch.cat([torch.nonzero(ally==c).flatten()[torch.randperm(int((ally==c).sum()),generator=g)[:1000]] for c in range(10)]);mask=torch.ones(len(ally),dtype=torch.bool);mask[valid]=False;vx,vy=allx[valid],ally[valid];m,mem,direct,bank,audit,timing=e.train(seed,allx[mask],ally[mask],SPLITS)
  with torch.no_grad():
   q=np.concatenate([e.keys(b) for b in vx.split(512)]);b=torch.cat([torch.tanh(m.layers[0](z)).reshape(-1,25,4) for z in vx.split(512)])
   def acc(route):return float((m.head(b.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)==vy).float().mean())
   rows=[]
   for d,obj in mem[1].models.items():
    for gamma in [0.,.01,.05,.2]:
     for scramble in [False,True]:rows.append({'method':'pmi','dimension':d,'gamma':gamma,'scramble':scramble,'accuracy':acc(decode(obj,q,mem[1].cooc,gamma,scramble)),'extra_bytes':obj.size()+(2560 if gamma else 0)})
   rows.append({'method':'original_local','accuracy':acc(mem[1].local.routes(q,0))});rows.append({'method':'global_ridge','accuracy':acc((q@mem[1].ridge.weight(.1).T).reshape(-1,25,4).argmax(-1))});rows.append({'method':'direct_ridge','accuracy':float(((q@direct.weight(1.).T).argmax(-1)==vy.numpy()).mean())})
  records.append({'seed':seed,'validation_only':True,'rows':rows});e.save(OUT/'results.json',records);print('RESULT',seed,rows,flush=True)
 e.save(OUT/'completion.json',{'complete':len(records)==2,'development_readouts':38,'official_test_used':False})
if __name__=='__main__':main()
