"""Stronger slowing-SGD sweep and validation-matched acquisition control."""
import argparse,json,math,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_resource_benchmark import permutations,dump
from bdh_cl.scripts.local_graph_plasticity import fit,evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 base=json.loads(Path('results/local_graph_plasticity_20261013/selection.json').read_text());selections={};rows=[]
 dump(out/'protocol.json',{'purpose':'Previously tested100k/400kdecay schedules still acquire much more accurately than branches. Add1000/3000/10000/30000presentations scales and select slowingMLP with recent4validationaccuracy closest to originalbranch and independentlyselected localvariant. A <=5pp validationgap is required to call it approximately acquisitionmatched. Also retain highest all-domain-validation MLPschedule across entire sweep.', 'development':'Same360/370train-validation split and16domains as primary plasticity study; model selection uses training-validation only. No new tests used for selection. All new schedules predefined.', 'confirmation':'Three samepairedseeds361-363/371-373 as primary study;32domains, full10ktestfinal. Paired followup, not independent replication. Extra schedules do not alter branch methods. Same256unit MLP, no replay/taskIDs or boundaryLRresets. Selection at16maynot match acquisition at32; report actualremaininggap.'})
 (out/'source.py').write_text(Path(__file__).read_text())
 for dataset,dev in [('mnist',360),('fashion_mnist',370)]:
  orig=base[dataset]['original'];target=orig['recent_four'];constant=next(c for c in base[dataset]['candidates'] if c['name']=='mlp_constant')['setting'];candidates=[c for c in base[dataset]['candidates'] if c['arm']=='mlp']
  raw=load_raw('data_cache/'+dataset);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  rng=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)]);mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;vid=val.reshape(10,1000)[:,:200].flatten();ps=permutations(dev,16)
  for decay in [1000.,3000.,10000.,30000.]:
   s={**constant,'decay':decay};m,rs=fit(dev,'mlp',s,x,y,mask,ps,x[vid],y[vid],[4,8,16]);candidates.append({'name':f'mlp_decay_{int(decay)}','arm':'mlp','setting':s,**rs[-1],'curve':rs});print('dev',dataset,int(decay),'all',round(rs[-1]['accuracy']*100,2),'recent4',round(rs[-1]['recent_four']*100,2),'target',round(target*100,2),flush=True)
  match=min(candidates,key=lambda c:abs(c['recent_four']-target));best=max(candidates,key=lambda c:c['accuracy']);gap=abs(match['recent_four']-target)
  improved_target=base[dataset]['local']['recent_four'];improved_match=min(candidates,key=lambda c:abs(c['recent_four']-improved_target));improved_gap=abs(improved_match['recent_four']-improved_target)
  selections[dataset]={'target_recent4':target,'matched':match,'validation_gap':gap,'within5pp':gap<=.05,'improved_target_recent4':improved_target,'improved_match':improved_match,'improved_gap':improved_gap,'improved_within5pp':improved_gap<=.05,'best_all_accuracy':best,'candidates':candidates};dump(out/'selection.json',selections)
  tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)])
  tests=[('acquisition_match_original',match)]
  if improved_match['setting']!=match['setting']:tests.append(('acquisition_match_selected',improved_match))
  if best['setting']['decay'] not in [100000.,400000.,math.inf] and all(best['setting']!=c['setting'] for _,c in tests):tests.append(('best_extended_schedule',best))
  for seed in range(dev+1,dev+4):
   ps=permutations(seed,32)
   for name,c in tests:
    m,rs=fit(seed,'mlp',c['setting'],x,y,torch.ones(len(y),dtype=torch.bool),ps,tx[tid],ty[tid],[4,8,16,32]);a=evaluate(m,tx,ty,ps)
    for r in rs:rows.append({'dataset':dataset,'seed':seed,'name':name,'setting':c['setting'],**r,'metric_scope':'stratified2000testsubset'})
    rows.append({'dataset':dataset,'seed':seed,'name':name,'setting':c['setting'],'domains':32,'accuracy':st.mean(a),'first_four':st.mean(a[:4]),'recent_four':st.mean(a[-4:]),'newest':a[-1],'domain_accuracy':a,'metric_scope':'full10000test'});dump(out/'results.json',rows);print('test',dataset,seed,name,'all',round(st.mean(a)*100,2),'first4',round(st.mean(a[:4])*100,2),'new',round(a[-1]*100,2),flush=True)
 lines=['# Acquisition-matched slowing-MLP follow-up','','Validation matches recent-four domain accuracy at16domains; actual32-domain differences must be reported, not assumed away. Followup reuses primary confirmationseeds and is paired rather than independent.','', '| Dataset | Target | Selected decay | Validation recent4 gap (pp) | Within5pp |','|---|---|---:|---:|---|']
 for ds,s in selections.items():
  lines.append(f'| {ds} | original | {s["matched"]["setting"]["decay"]} | {100*s["validation_gap"]:.2f} | {s["within5pp"]} |')
  lines.append(f'| {ds} | selected local | {s["improved_match"]["setting"]["decay"]} | {100*s["improved_gap"]:.2f} | {s["improved_within5pp"]} |')
 lines+=['','| Final32domains | Model | All accuracy | First4 | Recent4 | Newest |','|---|---|---:|---:|---:|---:|']
 for ds in selections:
  for name in sorted(set(r['name'] for r in rows if r['dataset']==ds)):
   rs=[r for r in rows if r['dataset']==ds and r['name']==name and r['metric_scope']=='full10000test'];f=lambda k:100*st.mean(r[k] for r in rs)
   lines.append(f'| {ds} | {name} | {f("accuracy"):.2f}% | {f("first_four"):.2f}% | {f("recent_four"):.2f}% | {f("newest"):.2f}% |')
 (out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
