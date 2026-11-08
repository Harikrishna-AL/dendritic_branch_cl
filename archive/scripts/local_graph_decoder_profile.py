"""Equivalent parallel max-plus reduction; timing diagnostic, no accuracy tuning."""
import json,time,statistics as st
from pathlib import Path
import torch
from bdh_cl.scripts.local_graph_resource_benchmark import Branch,BASE
from bdh_cl.scripts.local_graph_domains import branch_scores

@torch.no_grad()
def tree_scores(model,response,gamma):
 n,k=model.n,model.k
 value=-(response.reshape(-1,1,n,k)-model.codes[None,:,:,None]).square()
 count=model.links+1.;edge=(count/count.sum(-1,keepdim=True)).log()
 # Each edge matrix absorbs the unary at its destination.
 matrices=gamma*edge[None,None]+value[:,:,1:,None,:]
 while matrices.shape[2]>1:
  pairs=matrices.shape[2]//2
  left=matrices[:,:,:2*pairs:2];right=matrices[:,:,1:2*pairs:2]
  merged=(left[:,:,:,:,:,None]+right[:,:,:,None,:,:]).max(-2).values
  matrices=torch.cat([merged,matrices[:,:,-1:]],dim=2) if matrices.shape[2]%2 else merged
 return (value[:,:,0,:,None]+matrices[:,:,0]).flatten(-2).max(-1).values

def timing(fn,x):
 for _ in range(3):fn(x)
 vals=[]
 for _ in range(25):
  t=time.perf_counter();fn(x);vals.append((time.perf_counter()-t)*1000)
 return st.median(vals)

def main():
 torch.set_num_threads(2);out=Path('results/local_graph_decoder_profile_20261013');out.mkdir(exist_ok=False)
 checkpoint=torch.load('results/local_graph_domains_20261013/global_271_branch.pt',weights_only=False)
 model=Branch(271,BASE)
 for name in ['weight','codes','links','error','usage']:setattr(model,name,checkpoint[name])
 torch.manual_seed(901);rows=[];checks=[]
 for batch in [1,128]:
  x=torch.rand(batch,784);response=torch.tanh(model.data_features(x)@model.weight)
  a=branch_scores(model,response,.1);b=tree_scores(model,response,.1)
  assert torch.allclose(a,b,atol=2e-5,rtol=1e-5);assert torch.equal(a.argmax(-1),b.argmax(-1))
  checks.append({'batch':batch,'max_score_difference':float((a-b).abs().max()),'predictions_identical':True})
  sequential=lambda x:branch_scores(model,torch.tanh(model.data_features(x)@model.weight),.1).argmax(-1)
  parallel=lambda x:tree_scores(model,torch.tanh(model.data_features(x)@model.weight),.1).argmax(-1)
  rows.append({'batch':batch,'sequential_ms':timing(sequential,x),'tree_ms':timing(parallel,x)})
 (out/'results.json').write_text(json.dumps({'timing':rows,'equivalence_checks':checks,'limits':'Representative trained4branch model, random shaped inputs. No test-based hyperparameters or changed weights. Tree reduction trades more arithmetic and temporary storage for fewer sequential tensor operations. It is a timing diagnostic, not a new accuracy cohort. Floating point association changes; score equality checked within tolerance.'},indent=2))
 (out/'source.py').write_text(Path(__file__).read_text());print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
