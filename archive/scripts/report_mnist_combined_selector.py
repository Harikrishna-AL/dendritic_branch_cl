import json,statistics,math
from pathlib import Path
p=Path('results/mnist_combined_selector_20261105');rs=json.loads((p/'results.json').read_text());lines=['# Combined image-only selector experiment','', 'Sparse 168-neuron four-branch MLP; 196 fixed random pixel inputs per branch. Three seeds × forward/reverse orders; two epochs per task; no replay, prototypes, pretraining or task identity at inference. Known boundaries during training. Hyperparameters are fixed in protocol.json.','', '| Training / inference | Validation accuracy | Validation oracle | Model MB | Parameters | Forgetting pp | Training s |','|---|---:|---:|---:|---:|---:|---:|']
for arm in ['baseline','magnitude_joint','selector_hard','selector_soft','selector_soft_slow']:
 rows=[r for r in rs if r['arm']==arm]
 if len(rows)!=6:continue
 for mode in (['primary','mean','magnitude_soft'] if arm=='baseline' else ['primary']):
  vals=[statistics.mean(r['stages'][-1][mode]['accuracy']*100 for r in rows if r['seed']==seed) for seed in [101,102,103]]
  oracle=statistics.mean(r['stages'][-1]['oracle']['accuracy']*100 for r in rows)
  forgetting=statistics.mean(statistics.mean(max(s[mode]['pairs'][pair] for s in r['stages'][i:])-r['stages'][-1][mode]['pairs'][pair] for i,pair in enumerate(r['order'][:-1]))*100 for r in rows)
  lines.append(f"| {arm} / {mode} | {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}% | {oracle:.2f}% | {rows[0]['model_bytes']/1e6:.3f} | {rows[0]['parameters']:,} | {forgetting:.2f} | {statistics.mean(r['training_seconds'] for r in rows):.1f} |")
if (p/'selection.json').exists():
 selected=json.loads((p/'selection.json').read_text());tests=json.loads((p/'test_results.json').read_text());lines+=['','## Validation-selected official test comparison','',f"Selected: {selected['arm']} / {selected['mode']}; mean validation {100*selected['validation_accuracy']:.2f}%.",'']
 for arm,mode in [('baseline','primary'),(selected['arm'],selected['mode'])]:
  rows=[r for r in tests if r['arm']==arm];vals=[statistics.mean(r['scores'][mode]['accuracy']*100 for r in rows if r['seed']==s) for s in [101,102,103]];lines.append(f'- {arm} / {mode}: {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}% test accuracy.')
 if selected['arm']!='baseline' or selected['mode']!='primary':
  d=[]
  for s in [101,102,103]:
   diffs=[]
   for r in [r for r in tests if r['arm']==selected['arm'] and r['seed']==s]:
    b=next(t for t in tests if t['arm']=='baseline' and t['seed']==s and t['order']==r['order']);diffs.append((r['scores'][selected['mode']]['accuracy']-b['scores']['primary']['accuracy'])*100)
   d.append(statistics.mean(diffs))
  mean=statistics.mean(d);half=4.30265273*statistics.stdev(d)/math.sqrt(3);lines.append(f'- Paired seed-level difference: {mean:+.2f} pp; descriptive 95% t interval [{mean-half:+.2f}, {mean+half:+.2f}] pp.')
lines+=['','Accuracy ± SD uses three seed-level means over task orders. Model memory includes compact branch weights and connection indices, excludes optimizer, temporary activations and diagnostic archives. Training time includes intermediate validation. Oracle uses true task identity to choose saved branch routes and is not image-only.','', 'Baseline primary chooses greatest absolute raw response but forwards its tanh value. Mean and magnitude_soft are frozen post-training alternatives; magnitude_soft uses softmax(abs(raw)) weights. magnitude_joint adds image-only hard-winner CE to trace CE + auxiliary CE. Learned selectors add a 784→32→672 network, image-only CE, and soft-target supervision from per-image relative counterfactual branch errors. Hard selection uses straight-through softmax gradients. Soft selection uses weighted tanh branches. Slow variant reduces branch/head LR from .0005 to .0001 after task one while maintaining selector LR .0005. No importance-based weight protection was implemented.','', 'Selector targets assess one-neuron substitutions relative to the trace route, not optimal simultaneous routes. Selectors can forget. These are fixed-recipe comparisons, not separately tuned methods. Selection is based on aggregate validation accuracy, and official test results are reported for selected configuration and baseline only.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
