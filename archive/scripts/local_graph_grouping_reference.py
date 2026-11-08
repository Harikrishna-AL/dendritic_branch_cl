"""Paired, validation-selected four-branch reference for the grouping study."""
import argparse,json,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_grouping_benchmark import train,permutations,acc,latency,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
 source=Path('results/local_graph_grouping_benchmark_20261013');selection=json.loads((source/'selection.json').read_text());reference=json.loads((source/'results.json').read_text());rows=[]
 (out/'source.py').write_text(Path(__file__).read_text());dump(out/'protocol.json',{'comparison':'Separately validation-selected K4 against K8 on identical confirmation streams, seed321-323MNIST/331-333Fashion. Select K4 ONLY from already recorded training-validation candidates; no new hyperparameters selected using tests. Added as paired follow-up, not independent confirmation. Training2passes/domain; all other settings identical to grouping study.'})
 for ds,testbase in [('mnist',320),('fashion_mnist',330)]:
  chosen=max([r for r in selection[ds]['candidates'] if r['setting']['branches']==4],key=lambda r:r['accuracy']);s=chosen['setting']
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)])
  for seed in range(testbase+1,testbase+4):
   ps=permutations(seed,8);m,matrix,seconds,state=train(seed,'grouped',s,x,y,torch.ones(len(y),dtype=torch.bool),ps,(tx[tid],ty[tid]));final=acc(m,tx,ty,ps)
   r={'dataset':ds,'seed':seed,'arm':'tuned_four_branch','setting':s,'accuracy':st.mean(final),'old_accuracy':st.mean(final[:-1]),'acquisition':st.mean(matrix[d][d] for d in range(8)),'forgetting':st.mean(max(matrix[t][d] for t in range(d,8))-matrix[-1][d] for d in range(7)),'retention_matrix':matrix,'domain_accuracy':final,'train_seconds':seconds,'numeric_bytes':state,'latency':latency(m,tx)};rows.append(r);dump(out/'results.json',rows);print(ds,seed,round(r['accuracy']*100,2),flush=True)
 lines=['# Paired branch grouping follow-up','','| Dataset | Tuned four branches | Selected eight branches | Paired difference (pp) |','|---|---:|---:|---:|']
 for ds in ['mnist','fashion_mnist']:
  k4={r['seed']:r for r in rows if r['dataset']==ds};k8={r['seed']:r for r in reference if r['dataset']==ds and r['arm']=='grouped'}
  lines.append(f'| {ds} | {100*st.mean(r["accuracy"] for r in k4.values()):.2f}% | {100*st.mean(r["accuracy"] for r in k8.values()):.2f}% | {100*st.mean(k8[s]["accuracy"]-k4[s]["accuracy"] for s in k4):+.2f} |')
 lines+=['','Same streams and256input-response units. Grouping changes code length and active-unit count as well as branch count; this is not a pure mechanism isolation. Three paired seeds give limited uncertainty estimates. Full candidate selection is archived in the source study.'];(out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
