"""Fixed-capacity long streams: accuracy decay and acquisition/retention separation."""
import argparse,json,time,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_resource_benchmark import Branch,Central,MLP,permutations,dump
from bdh_cl.scripts.local_graph_grouping_benchmark import Grouped

@torch.no_grad()
def accuracy(model,x,y,p):
 pred=torch.cat([model.predict(b[:,p]) for b in x.split(256)])
 return float((pred==y).float().mean())

@torch.no_grad()
def checkpoint_accuracy(model,x,y,perms,subset):
 full=[];small=[]
 for p in perms:
  pred=torch.cat([model.predict(b[:,p]) for b in x.split(256)])
  correct=pred==y;full.append(float(correct.float().mean()));small.append(float(correct[subset].float().mean()))
 return full,small

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 source=json.loads(Path('results/local_graph_resource_benchmark_20261013/selection.json').read_text());group=json.loads(Path('results/local_graph_grouping_benchmark_20261013/selection.json').read_text())
 checkpoints=[4,8,16,32];rows=[]
 protocol={'datasets':['mnist','fashion_mnist'],'checkpoints':checkpoints,'seeds':{'mnist':[341,342,343],'fashion_mnist':[351,352,353]},'capacity':'Fixed throughout32domains: original64x4branch,32x8branch,SGD256MLP,central256units with8prototype slots. No added neurons, branches, prototypes or optimizer resets at checkpoints. Numeric bytes unchanged. No replay or task IDs supplied.','training':'Two current-domain passes, all60ktrain examples each. Domain1 identity then31random permutations; common seeded prefixes across every model.3,840,000current presentations/model. Same labels all10classes in each domain. Source images reused transformed, standard permuted-MNIST protocol.','selection':'All settings frozen from previously archived training-validation selections for8domains. No tuning at16/32, no officialtest feedback. This measures fixed-rule scaling and capacity overload, not best retuned performance at each length.','metrics':'At4/8/16/32evaluate full10ktest per seen domain, plus retained first4mean, recent4mean, oldest/newest. Acquisition eachdomain measured once immediately afterlearning using fixed stratified2000test examples. Forgetting since acquisition=max(0,initial-final) averaged over olddomains on same subset. Signed backward transfer also reported; distinct from earlier peak-to-final forgetting metric. Model training timers exclude evaluation.'}
 dump(out/'protocol.json',protocol);(out/'source.py').write_text(Path(__file__).read_text());settings={}
 for dataset in ['mnist','fashion_mnist']:
  s=source[dataset+'_8'];settings[dataset]={'branch4':s['branch']['chosen']['setting'],'branch8':group[dataset]['chosen']['setting'],'sgd_mlp':s['sgd_mlp']['chosen']['setting'],'central8':{**s['central']['chosen']['setting'],'slots':8}}
 dump(out/'settings.json',settings)
 for dataset in ['mnist','fashion_mnist']:
  raw=load_raw('data_cache/'+dataset);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  subset=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)]);smallx=tx[subset];smally=ty[subset]
  for seed in protocol['seeds'][dataset]:
   ps=permutations(seed,32)
   for arm in ['branch4','branch8','sgd_mlp','central8']:
    s=settings[dataset][arm];model={'branch4':Branch,'branch8':Grouped,'sgd_mlp':MLP,'central8':Central}[arm](seed,s);acquisition=[];training_seconds=0.;initial_bytes=model.deployed_bytes()
    for domain,p in enumerate(ps):
     t=time.perf_counter()
     for epoch in range(2):
      rng=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
      for ix in torch.randperm(len(y),generator=rng).split(128):model.learn(x[ix][:,p],y[ix])
     training_seconds+=time.perf_counter()-t
     acquisition.append(accuracy(model,smallx,smally,p));assert model.deployed_bytes()==initial_bytes
     count=domain+1
     if count in checkpoints:
      final,small=checkpoint_accuracy(model,tx,ty,ps[:count],subset)
      delta=[small[i]-acquisition[i] for i in range(count-1)]
      row={'dataset':dataset,'seed':seed,'arm':arm,'domains':count,'accuracy':st.mean(final),'domain_accuracy':final,'first_four_accuracy':st.mean(final[:4]),'recent_four_accuracy':st.mean(final[-4:]),'old_accuracy':st.mean(final[:-1]),'new_accuracy':final[-1],'acquisition_mean':st.mean(acquisition),'acquisition_by_domain':acquisition.copy(),'retention_subset':small,'forgetting_since_acquisition':st.mean(max(0.,-d) for d in delta),'backward_transfer':st.mean(delta),'training_seconds':training_seconds,'numeric_bytes':model.bytes(),'deployed_numeric_bytes':model.deployed_bytes()}
      rows.append(row);dump(out/'results.json',rows);print(dataset,seed,arm,count,'avg',round(row['accuracy']*100,2),'first4',round(row['first_four_accuracy']*100,2),'new',round(row['new_accuracy']*100,2),flush=True)
    # Only final state is archived; no saved phase configurations enter prediction.
    if arm in ['branch4','branch8']:
     torch.save({k:getattr(model,k) for k in ['weight','codes','error','usage','links']},out/f'{dataset}_{seed}_{arm}_final.pt')
    else:torch.save(model.net.state_dict() if arm=='sgd_mlp' else {'weight':model.weight,'codes':model.codes,'error':model.error,'usage':model.usage,'assemblies':model.assemblies,'counts':model.counts},out/f'{dataset}_{seed}_{arm}_final.pt')
 report(out,rows)

def report(out,rows):
 lines=['# Fixed-capacity scaling to32domains','','Three fresh paired seeds per dataset. Same fixed model capacity, settings and training exposure per domain; no task IDs or replay. Hyperparameters inherited from training-only8-domain validation and frozen. All accuracy columns below use full testsets unless marked acquisition/subset.','','| Dataset | Domains | Model | All-domain accuracy | First4 accuracy | Newest accuracy | Acquisition (subset) | Forgetting since acquisition (pp, subset) |','|---|---:|---|---:|---:|---:|---:|---:|']
 for ds in ['mnist','fashion_mnist']:
  for d in [4,8,16,32]:
   for arm in ['branch4','branch8','sgd_mlp','central8']:
    rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,d,arm)];m=lambda k:100*st.mean(r[k] for r in rs)
    lines.append(f'| {ds} | {d} | {arm} | {m("accuracy"):.2f}% | {m("first_four_accuracy"):.2f}% | {m("new_accuracy"):.2f}% | {m("acquisition_mean"):.2f}% | {m("forgetting_since_acquisition"):.2f} |')
 lines+=['','Retention is reported alongside acquisition and new-domain accuracy to expose learners that preserve old skills by failing to learn new domains. Forgetting here is clipped initial-acquisition-minus-final accuracy, not the earlier best-observed-minus-final metric. First4 preservation uses exactly the same initial domains across checkpoints, avoiding changes in the evaluated task mixture. Three seeds are paired observations, not independent samples per domain. Stored bytes remain fixed; no claim of equal FLOPs or retuned-best scaling.'];(out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
