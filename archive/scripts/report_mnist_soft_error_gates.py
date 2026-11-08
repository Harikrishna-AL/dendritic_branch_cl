import json,statistics,math
from pathlib import Path
p=Path('results/mnist_soft_error_gates_20261105');rs=json.loads((p/'results.json').read_text());lines=['# Soft error-trace gate experiment','', 'Split MNIST, 168 neurons × four branches, 196 fixed random pixels per branch. Three seeds × two task orders, two epochs per task, same optimizer and auxiliary loss. Reported image-only inference uses softmax(abs(raw response)) to combine signed tanh branches.','', '| Training gate | Validation image-only acc | Single-branch oracle | Mixture oracle | Gate max weight | Normalized entropy | Train s |','|---|---:|---:|---:|---:|---:|---:|']
for arm in ['hard','soft_0.1','soft_0.01','soft_0.001','soft_0.0001','soft_0.00001','anneal','uniform']:
 rows=[r for r in rs if r['arm']==arm]
 if len(rows)!=6:continue
 vals=[statistics.mean(r['stages'][-1]['magnitude_soft']['accuracy']*100 for r in rows if r['seed']==s) for s in [101,102,103]];oracle=statistics.mean(r['stages'][-1]['oracle']['accuracy']*100 for r in rows);mixture=statistics.mean(r['stages'][-1]['oracle_mixture']['accuracy']*100 for r in rows);maximum=statistics.mean(g['mean_max_weight'] for r in rows for g in r['gate_stats']);entropy=statistics.mean(g['normalized_entropy'] for r in rows for g in r['gate_stats']);seconds=statistics.mean(r['training_seconds'] for r in rows)
 lines.append(f'| {arm} | {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}% | {oracle:.2f}% | {mixture:.2f}% | {maximum:.3f} | {entropy:.3f} | {seconds:.1f} |')
if (p/'test_results.json').exists():
 selected=json.loads((p/'selection.json').read_text())['arm'];tests=json.loads((p/'test_results.json').read_text());lines+=['',f'Validation-selected training gate: **{selected}**.','', 'Official test evaluation (selected configuration and hard baseline only):','']
 for arm in sorted(set(['hard',selected])):
  rows=[r for r in tests if r['arm']==arm];vals=[statistics.mean(r['scores']['magnitude_soft']['accuracy']*100 for r in rows if r['seed']==s) for s in [101,102,103]];lines.append(f'- {arm}: {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}% image-only test accuracy.')
 if selected!='hard':
  d=[]
  for s in [101,102,103]:
   delta=[]
   for r in [r for r in tests if r['arm']==selected and r['seed']==s]:
    b=next(b for b in tests if b['arm']=='hard' and b['seed']==s and b['order']==r['order']);delta.append((r['scores']['magnitude_soft']['accuracy']-b['scores']['magnitude_soft']['accuracy'])*100)
   d.append(statistics.mean(delta))
  mean=statistics.mean(d);half=4.30265273*statistics.stdev(d)/math.sqrt(3);lines.append(f'- Paired seed-level gain {mean:+.2f} pp; descriptive 95% t interval [{mean-half:+.2f}, {mean+half:+.2f}] pp.')
lines+=['','Training gate = softmax(-EMA branch error / temperature), detached from gradient. EMA=.95. Fixed temperatures .1, .01, .001, .0001, .00001; annealing exponentially .1→.001 over each task, restarting at known task boundaries without resetting EMA or optimizer. Uniform control fixes .25 per branch. Every auxiliary counterfactual replaces exactly one neuron’s current mixture with one branch.','', 'Both oracle diagnostics require true task identity: single-branch oracle retrieves the saved minimum-error branch; mixture oracle retrieves the saved task-specific soft weights. Neither is image-only. Gate statistics average the training steps within each stage and then runs/stages. Entropy 0 = hard selection, 1 = uniform mixture.','', 'All models have 134,074 parameters and 1.593 MB persistent tensor state including sparse indices. No replay, prototypes, pretraining, or task identity at image-only inference. No test-driven temperature tuning. Accuracy ± SD uses three seed-level averages over two orders. Fixed recipe; results do not rule out other gate scales, penalties or training schedules.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
