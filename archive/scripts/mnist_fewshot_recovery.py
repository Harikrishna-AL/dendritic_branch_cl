"""Secondary few-shot recovery evaluation of frozen, sequentially trained MLPs."""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE
from bdh_cl.scripts.mnist_combined_selector import Model
CKPT=Path('results/mnist_combined_selector_20261105')

def coordinate_route(z,y,weight,bias,initial,sweeps=2):
 route=initial.clone();h=z.gather(2,route[None,:,None].expand(len(z),-1,-1)).squeeze(-1);logits=F.linear(h,weight,bias)
 for sweep in range(sweeps):
  for i in range(z.shape[1]):
   alt=logits[:,None,:]+(z[:,i,:]-h[:,i,None])[:,:,None]*weight[:,i][None,None,:]
   costs=F.cross_entropy(alt.reshape(-1,10),y[:,None].expand(-1,4).reshape(-1),reduction='none').reshape(len(y),4).mean(0)
   chosen=int(costs.argmin());logits=alt[:,chosen];h[:,i]=z[:,i,chosen];route[i]=chosen
 return route

def adapted_head(features,y,weight,bias):
 w=weight.clone().requires_grad_();b=bias.clone().requires_grad_();opt=torch.optim.Adam([w,b],lr=.01)
 for _ in range(20):
  opt.zero_grad();loss=F.cross_entropy(F.linear(features,w,b),y)+.01*(w-weight).square().mean();loss.backward();opt.step()
 return w.detach(),b.detach()

@torch.no_grad()
def cached(model,x):
 raw=torch.cat([model.responses(b,False) for b in x.split(256)]);z=raw.tanh();h=(z*raw.abs().softmax(-1)).sum(-1)
 return raw,z,h

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_fewshot_recovery_20261105'));p.add_argument('--repeats',type=int,default=10);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz');assert not np.intersect1d(split['train'],split['validation']).size
 def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
 sx,sy=ds(raw['train_images'][split['validation']],raw['train_labels'][split['validation']]);tx,ty=ds(raw['test_images'],raw['test_labels']);trainx,trainy=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']])
 contexts=torch.stack([trainx[trainy//2==pair].mean(0) for pair in range(5)])
 proto=dict(seeds=[101,102,103],orders=['forward','reverse'],shots_per_class=[1,2,5,10],support_repeats=a.repeats,support_pool='Original disjoint validation split, never training or test images',queries='Entire official MNIST test set; same query set across draws; repeats are not independent test datasets',scope='Recovery of previously learned classes, not novel-class few-shot generalization',frozen='All original network weights and traces frozen. Head-adaptation baseline fits a separate temporary head.',task_conditioned='Support contains two classes from one known episode; queries in that episode share those classes, but ten-class head is not masked',mixed='Support for all ten classes; query routing uses distance to five support task means, no query task identity',methods=dict(archive_error='Select among five archived routes by support CE, no direct label-to-route mapping',fresh_error='No route archive: initialize a single route from mean absolute support responses; two coordinate-descent sweeps minimizing support ten-class CE',prototype_router='Support-image mean chooses nearest stored training task mean, retrieving its saved route; requires training prototypes plus route archive',head_adapt='20 Adam steps LR .01, frozen image-only soft features, ten-class head, weak .01 mean-squared weight-change penalty'),limits='Support labels and task grouping are additional privileges; no comparisons with label-free CIL as if same protocol. Fixed adaptation recipes; no query-based tuning.')
 (a.out/'protocol.json').write_text(json.dumps(proto,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes());records=[];baselines=[]
 for seed in [101,102,103]:
  for first in [0,4]:
   torch.manual_seed(seed);m=Model('baseline',seed);saved=torch.load(CKPT/f'baseline_{seed}_{first}.pt',weights_only=True);m.load_state_dict(saved['state']);m.requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()};routes=saved['routes'].long()
   sr,sz,sh=cached(m,sx);_,qz,qh=cached(m,tx);w=m.head.weight.detach();b=m.head.bias.detach()
   with torch.no_grad():
    image=F.linear(qh,w,b).argmax(-1);bank=torch.stack([F.linear(qz.gather(2,r[None,:,None].expand(len(qz),-1,-1)).squeeze(-1),w,b) for r in routes],1)
    baselines.append(dict(seed=seed,first=first,image_accuracy=float((image==ty).float().mean()),oracle_accuracy=float((bank[torch.arange(len(ty)),ty//2].argmax(-1)==ty).float().mean())))
   for shots in [1,2,5,10]:
    for draw in range(a.repeats):
     # Nested shots use the same per-class permutation within each draw.
     gen=torch.Generator().manual_seed(seed*100000+first*1000+draw)
     class_ids=[torch.where(sy==c)[0] for c in range(10)];support=[ids[torch.randperm(len(ids),generator=gen)[:shots]] for ids in class_ids]
     means=[];predictions={name:torch.empty(5,len(ty),dtype=torch.long) for name in ['archive_error','fresh_error','prototype_router','head_adapt']};episode={name:[] for name in predictions};route_match=[];times={name:0. for name in predictions}
     for pair in range(5):
      ids=torch.cat(support[2*pair:2*pair+2]);ys=sy[ids];means.append(sx[ids].mean(0));mask=ty//2==pair
      with torch.no_grad():
       start=time.perf_counter();sb=torch.stack([F.linear(sz[ids].gather(2,r[None,:,None].expand(len(ids),-1,-1)).squeeze(-1),w,b) for r in routes],1);cost=F.cross_entropy(sb.reshape(-1,10),ys[:,None].expand(-1,5).reshape(-1),reduction='none').reshape(len(ids),5).mean(0);chosen=int(cost.argmin());times['archive_error']+=time.perf_counter()-start;predictions['archive_error'][pair]=bank[:,chosen].argmax(-1);route_match.append(int(chosen==pair))
       start=time.perf_counter();initial=sr[ids].abs().mean(0).argmax(-1);route=coordinate_route(sz[ids],ys,w,b,initial);times['fresh_error']+=time.perf_counter()-start;predictions['fresh_error'][pair]=F.linear(qz.gather(2,route[None,:,None].expand(len(qz),-1,-1)).squeeze(-1),w,b).argmax(-1)
       start=time.perf_counter();chosen=int((contexts-means[-1]).square().sum(-1).argmin());times['prototype_router']+=time.perf_counter()-start;predictions['prototype_router'][pair]=bank[:,chosen].argmax(-1)
      start=time.perf_counter();aw,ab=adapted_head(sh[ids],ys,w,b);times['head_adapt']+=time.perf_counter()-start
      with torch.no_grad():predictions['head_adapt'][pair]=F.linear(qh,aw,ab).argmax(-1)
      for name in predictions:episode[name].append(float((predictions[name][pair,mask]==ty[mask]).float().mean()))
     with torch.no_grad():
      means=torch.stack(means);assignment=(tx.square().sum(-1)[:,None]+means.square().sum(-1)[None]-2*tx@means.T).argmin(-1);mixed={name:float((pred[assignment,torch.arange(len(ty))]==ty).float().mean()) for name,pred in predictions.items()}
     records.append(dict(seed=seed,first=first,shots=shots,draw=draw,episode_accuracy={k:statistics.mean(v) for k,v in episode.items()},mixed_accuracy=mixed,mixed_router_accuracy=float((assignment==ty//2).float().mean()),archive_task_recovery=statistics.mean(route_match),adapt_seconds=times))
    (a.out/'results.json').write_text(json.dumps(records,indent=2));(a.out/'baselines.json').write_text(json.dumps(baselines,indent=2));rr=[r for r in records if r['seed']==seed and r['first']==first and r['shots']==shots];print(seed,first,shots,{name:round(100*statistics.mean(r['episode_accuracy'][name] for r in rr),2) for name in predictions},flush=True)
   assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
if __name__=='__main__':main()
