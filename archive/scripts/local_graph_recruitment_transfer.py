"""Fixed-rule cross-dataset transfer of successful novelty recruitment, no retuning."""
import json,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_resource_benchmark import permutations,dump
from bdh_cl.scripts.local_graph_plasticity import fit,evaluate

def main():
 out=Path('results/local_graph_recruitment_transfer_20261013');out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 source=json.loads(Path('results/local_graph_plasticity_20261013/selection.json').read_text());candidate=next(c for c in source['mnist']['candidates'] if c['name']=='novelty_allocation');setting=candidate['setting']
 dump(out/'protocol.json',{'purpose':'Test exact same error-threshold/leastusage recruitment rule successful on Fashion on MNIST; no new hyperparameters. Rule already included in MNIST trainingvalidation, though it did not win jointscore. This is an explicitly adaptive cross-dataset followup motivated by Fashion confirmation, not independently preregistered validationselection.', 'settings':setting,'seeds':[361,362,363],'domains':32,'protocol':'Same primary confirmation streams, sample order, training2passes/domain and fixedmodelsize812480bytes. Paired followup, no extra currentdata exposure, imagesarchive/taskIDs/replay. Full10ktest/domain final. Intermediatecurves2000testsubset. Do not merge paired followup into independentseed count.'})
 (out/'source.py').write_text(Path(__file__).read_text());raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long);tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)]);rows=[]
 for seed in [361,362,363]:
  ps=permutations(seed,32);m,rs=fit(seed,'local',setting,x,y,torch.ones(len(y),dtype=torch.bool),ps,tx[tid],ty[tid],[4,8,16,32]);a=evaluate(m,tx,ty,ps)
  for r in rs:rows.append({'dataset':'mnist','seed':seed,'name':'novelty_transfer','arm':'local','setting':setting,**r,'metric_scope':'stratified2000testsubset'})
  rows.append({'dataset':'mnist','seed':seed,'name':'novelty_transfer','arm':'local','setting':setting,'domains':32,'accuracy':st.mean(a),'first_four':st.mean(a[:4]),'recent_four':st.mean(a[-4:]),'newest':a[-1],'old_accuracy':st.mean(a[:-1]),'domain_accuracy':a,'numeric_bytes':m.bytes(),'metric_scope':'full10000test'});dump(out/'results.json',rows);print('test',seed,'all',round(st.mean(a)*100,2),'first4',round(st.mean(a[:4])*100,2),'new',round(a[-1]*100,2),flush=True)
 rs=[r for r in rows if r['metric_scope']=='full10000test'];f=lambda k:100*st.mean(r[k] for r in rs)
 (out/'REPORT.md').write_text(f'# Fixed local recruitment transfer to MNIST\n\nThree paired seeds from primary cohort; adaptive transfer followup, not an independent replication. No retuning.\n\nOverall {f("accuracy"):.2f}%; first4 {f("first_four"):.2f}%; newest {f("newest"):.2f}%.\n')
if __name__=='__main__':main()
