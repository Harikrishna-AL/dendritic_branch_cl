"""Compile every completed, accepted local experiment from this conversation."""
import csv,json,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
records=[];stages=[]

def add(experiment,architecture,method,seed,rule,history,average,memory,seconds=None,parameters=None,model_bytes=None,inference_bytes=None,peak_rss=None):
    # history entries have accuracy, oracle, routing and task_accuracy.
    final=history[-1]
    record={'experiment':experiment,'architecture':architecture,'method':method,'seed':seed,'readout':rule,
      'incremental_average_percent':100*average,'final_accuracy_percent':100*final['accuracy'],
      'oracle_accuracy_percent':100*final['oracle'],'routing_accuracy_percent':100*final['routing'],
      'acquisition_percent':100*st.mean(h['task_accuracy'][-1] for h in history),
      'forgetting_pp':100*st.mean(max(h['task_accuracy'][j] for h in history[j:-1])-final['task_accuracy'][j] for j in range(9)),
      'continuation_bytes':memory,'model_bytes':model_bytes,'primary_inference_tensor_bytes':inference_bytes,
      'parameters':parameters,'runtime_seconds':seconds,'peak_process_RSS_bytes':peak_rss}
    records.append(record)
    for i,h in enumerate(history):
        row={k:record[k] for k in ['experiment','architecture','method','seed','readout']}
        row.update(stage=i+1,accuracy_percent=100*h['accuracy'],oracle_accuracy_percent=100*h['oracle'],routing_accuracy_percent=100*h['routing'])
        row.update({f'task{j+1}_accuracy_percent':100*a for j,a in enumerate(h['task_accuracy'])});stages.append(row)

cnn=ROOT/'results/cnn_branch_comparison_20261027'
assert json.loads((cnn/'completion.json').read_text())['all_successful']
for f in sorted(cnn.glob('*/result.json')):
    r=json.loads(f.read_text())
    for rule in ['joint','prototype_cosine']:
        hh=[{'accuracy':h[rule]['accuracy'],'oracle':h[rule]['oracle_task_diagnostic'],'routing':h[rule]['routing_accuracy'],'task_accuracy':h[rule]['task_accuracy']} for h in r['history']]
        add('matched_CNN',r['size'],r['arm'],r['seed'],rule,hh,r['average_incremental'][rule],r['continuation_bytes'],r['seconds'],r['parameters'],r['model_bytes'],r['primary_inference_tensor_bytes'],r['peak_process_RSS_bytes'])
raw=ROOT/'results/dcnet_branch_pilot_fixednorm_20261026'
for r in json.loads((raw/'results.json').read_text()):
    for rule in ['head_probability','cosine']:
        hh=[{'accuracy':h[rule]['accuracy'],'oracle':h['oracle_task_diagnostic'],'routing':h[rule]['routing_accuracy'],'task_accuracy':h[rule]['task_accuracy']} for h in r['history']]
        add('raw_pixel_pilot','raw_128x10',r['arm'],r['seed'],rule,hh,r['average_incremental'][rule],r['persistent_bytes'],r['seconds'])
rawdc=ROOT/'results/dcnet_raw_short_20261027'
for r in json.loads((rawdc/'results.json').read_text()):
    hh=[{'accuracy':h['accuracy'],'oracle':h['oracle_task_diagnostic'],'routing':h['routing_accuracy'],'task_accuracy':[st.mean(h['per_class_accuracy'][10*j:10*(j+1)]) for j in range(i+1)]} for i,h in enumerate(r['history'])]
    add('raw_pixel_DCNet_variant','raw_128x10','dcnet_hat',r['seed'],'joint',hh,r['average_incremental'],r['continuation_bytes'],r['seconds'],model_bytes=r['model_bytes'])

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
write_csv(OUT/'per_seed_results.csv',records);write_csv(OUT/'all_stage_results.csv',stages)
(OUT/'all_results.json').write_text(json.dumps({'per_seed':records,'stages':stages},indent=2))

def mean_sd(group,key):
    v=[r[key] for r in group];return f'{st.mean(v):.2f} ± {st.stdev(v):.2f}'

def table(group):
    lines=['| Architecture | Method | Readout | Incremental avg (%) | Final (%) | Oracle (%) | Routing (%) | Continuation MB |',
      '|---|---|---|---:|---:|---:|---:|---:|']
    groups={}
    for r in group:groups.setdefault((r['architecture'],r['method'],r['readout']),[]).append(r)
    for key,rr in groups.items():
        lines.append('| '+' | '.join(key)+f' | {mean_sd(rr,"incremental_average_percent")} | {mean_sd(rr,"final_accuracy_percent")} | {mean_sd(rr,"oracle_accuracy_percent")} | {mean_sd(rr,"routing_accuracy_percent")} | {st.mean(r["continuation_bytes"] for r in rr)/1e6:.3f} |')
    return lines

lines=['# Complete local continual-learning results','',
'All 12 matched CNN runs and nine accepted raw-pixel runs completed. Three initialization/shuffle seeds (701–703), one fixed class order. Every final score evaluates all 100 CIFAR classes without a supplied inference task ID. No replay. ± is sample standard deviation across three seeds, not a confidence interval. Cluster jobs are outside this report.','',
'## Matched CNN comparison','',
'20 representation + 10 classifier epochs per task, batch 64. CNN16 channels 16–32–64; CNN32 channels 32–64–128. Three stride-2 convolutions, ReLU and global pooling. CNNs train from scratch with shared HAT protection; they are held fixed during classifier fitting. Both arms use DCNet IOE/DAC, augmentation, optimizer, projection and readout. Shared initialization and training random-stream seeds are matched. This isolates disjoint branch configurations versus learned HAT branch gating within that training framework. It is not the earlier raw-pixel pilot or published ResNet DCNet.','']
lines+=table([r for r in records if r['experiment']=='matched_CNN'])
lines+=['','## Acquisition and forgetting','',
'Acquisition averages current-task accuracy immediately after learning each task, with all then-seen classes competing. Forgetting averages best earlier accuracy minus final accuracy over the first nine tasks. These are different measurements from the final oracle diagnostic.','',
'| CNN | Method | Readout | Acquisition (%) | Forgetting (pp) |','|---|---|---|---:|---:|']
for architecture in ['cnn16','cnn32']:
    for method in ['branch_config','dcnet_hat']:
        for rule in ['joint','prototype_cosine']:
            rr=[r for r in records if r['experiment']=='matched_CNN' and r['architecture']==architecture and r['method']==method and r['readout']==rule]
            lines.append(f'| {architecture} | {method} | {rule} | {mean_sd(rr,"acquisition_percent")} | {mean_sd(rr,"forgetting_pp")} |')
lines+=['','## Resources for the CNN comparison','',
'Continuation state includes model tensors, cumulative/update masks, retained class directions and concentration summaries. Inference tensor bytes are a computed subset for the primary joint readout, not measured GPU memory or a minimal deployment footprint. Peak RSS includes the dataset, Python, allocator and runtime; it is not VRAM. MB/GB use decimal units.','',
'| CNN | Method | Parameters | Continuation MB | Primary inference tensors MB | Mean run minutes | Maximum process RSS GB |','|---|---|---:|---:|---:|---:|---:|']
for architecture in ['cnn16','cnn32']:
    for method in ['branch_config','dcnet_hat']:
        rr=[r for r in records if r['experiment']=='matched_CNN' and r['architecture']==architecture and r['method']==method and r['readout']=='joint']
        q=rr[0];lines.append(f'| {architecture} | {method} | {q["parameters"]:,} | {q["continuation_bytes"]/1e6:.3f} | {q["primary_inference_tensor_bytes"]/1e6:.3f} | {st.mean(r["runtime_seconds"] for r in rr)/60:.2f} | {max(r["peak_process_RSS_bytes"] for r in rr)/1e9:.3f} |')
lines+=['','Total worker wall time for the 12 sequential CNN runs: 3.46 hours. Full contiguous update masks were needed for correct MPS protection; mask reconstruction/compression could reduce memory in another implementation.','',
'## Every CNN seed','',
'| CNN | Method | Seed | Readout | Incremental avg (%) | Final (%) | Oracle (%) |','|---|---|---:|---|---:|---:|---:|']
for r in records:
    if r['experiment']=='matched_CNN':lines.append(f'| {r["architecture"]} | {r["method"]} | {r["seed"]} | {r["readout"]} | {r["incremental_average_percent"]:.2f} | {r["final_accuracy_percent"]:.2f} | {r["oracle_accuracy_percent"]:.2f} |')
lines+=['','## Earlier raw-pixel experiments — separate protocols','',
'Our raw-pixel pilot used five current-task epochs, fixed orthogonal class targets, Adam, no image augmentation, disjoint hard branch allocation and either probability or cosine competition. IOE-only is its compaction-loss ablation. The raw-pixel DCNet variant used five representation plus five classifier epochs, native DCNet losses/augmentation and learned HAT gates. Both have a 3072→128×10 tanh feature layer, but their full training/readout recipes differ. These are not resource-matched and cannot isolate the effect of adding a CNN.','']
lines+=table([r for r in records if r['experiment']!='matched_CNN'])
lines+=['','The exploratory run using full-dataset centering is superseded and excluded. The ResNet experiment was explicitly stopped before producing a complete score and is not a result.','',
'## Interpretation','',
'- In the matched CNN experiment, DCNet-HAT wins the primary joint readout on both sizes and every seed.',
'- With prototype cosine, CNN16 is tied; CNN32 gives branch configurations a mean +0.19 percentage points. That advantage occurs in only one of the three paired CNN32 seeds, so it is not a consistent win.',
'- Larger CNNs improve cosine scores and oracle diagnostics, but automatic task/head selection remains weak: roughly 15–16% routing accuracy against 10% chance. Both representation quality and cross-head calibration/retrieval need improvement.',
'- The continuation-memory saving is 39,680 bytes (approximately 0.040 MB). There is no demonstrated substantial memory or speed advantage in this implementation.',
'- Tiny-CNN HAT gates become heavily occupied as tasks accumulate. Per-task encoder weight movement and channel occupancy are recorded in each run’s cnn_updates.json.',
'- These results do not establish superiority over published DCNet or state of the art. Training budgets, architecture and one class order limit generalisation.','',
'## Files','',
'- [Per-seed metrics CSV](per_seed_results.csv): every completed model/readout and resource measurement.',
'- [All stage results CSV](all_stage_results.csv): 390 stage/readout records, including per-task accuracies.',
'- [All results JSON](all_results.json).',
'- [CNN protocol and live report](../cnn_branch_comparison_20261027/REPORT.md).',
'- [Raw branch pilot](../dcnet_branch_pilot_20261026/REPORT.md).',
'- [Raw-pixel DCNet comparison](../dcnet_raw_baseline_20261027/REPORT.md).']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f'Compiled {len(records)} per-seed/readout rows and {len(stages)} stage rows.')
