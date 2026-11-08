"""Build plots and a transparent interpretation of the fixed-capacity long stream."""
import json,statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
rows=json.loads((root/'results/local_graph_long_stream_20261013/results.json').read_text());assert len(rows)==96
arms=['branch4','branch8','sgd_mlp','central8'];labels=['64 neurons × 4 branches','32 neurons × 8 branches','MLP 256 / SGD','Central memory / 8 slots'];colors=['#3977b8','#167f79','#d67b27','#8b58a2'];counts=[4,8,16,32]
fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for i,ds in enumerate(['mnist','fashion_mnist']):
 for j,(key,title) in enumerate([('accuracy','All domains learned so far'),('first_four_accuracy','Same first four domains'),('new_accuracy','Newest domain')]):
  ax=axs[i,j]
  for arm,label,color in zip(arms,labels,colors):
   groups=[[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,d,arm)] for d in counts]
   means=np.array([100*st.mean(r[key] for r in rs) for rs in groups]);sd=np.array([100*st.stdev(r[key] for r in rs) for rs in groups])
   ax.plot(counts,means,marker='o',color=color,label=label);ax.fill_between(counts,means-sd,means+sd,color=color,alpha=.12)
  ax.set_xscale('log',base=2);ax.set_xticks(counts,[str(x) for x in counts]);ax.set_ylim(0,100);ax.grid(alpha=.2);ax.set_xlabel('Number of sequential domains');ax.set_title(title)
  if j==0:ax.set_ylabel(('MNIST' if ds=='mnist' else 'Fashion MNIST')+' accuracy (%)')
axs[0,0].legend(fontsize=8,loc='lower left')
fig.suptitle('Fixed model sizes through 32 domains: separate retention from continued acquisition\nThree fresh paired seeds per dataset; shading is one sample standard deviation',fontsize=13)
fig.savefig(out/'scaling.png',dpi=180);fig.savefig(out/'scaling.svg');plt.close(fig)
lines=['# Does local branch memory degrade more slowly as domains accumulate?','','This experiment extends the fixed-capacity comparison to 4, 8, 16 and 32 sequential pixel mappings on MNIST and Fashion MNIST. Reported means use three fresh paired seeds per dataset. See all-domain accuracy, retention of exactly the same first four domains, and newest-domain learning separately; none alone establishes a superior continual learner.','',
'## Protocol','',
'- All ten classes appear in every domain. Domain 1 is ordinary images; later domains use fixed random permutations. Every model sees identical domain prefixes and sample order for its seed.','- Two passes over all 60,000 current-domain training examples; no replay of earlier-domain images, task IDs, boundary notifications, optimizer resets or extra model capacity. Source images reappear under new transformations as in standard permuted-MNIST.','- All settings are frozen from the prior training-validation selections for eight domains. There is no tuning on the current official test results or separate optimization at 16/32 domains. This measures scaling of these fixed rules, not the best achievable performance of each architecture at every length.','- Final checkpoint accuracy uses all 10,000 test images for every seen domain. Acquisition is measured immediately after each domain using the same fixed stratified 2,000-example test subset. Labels are used only to calculate metrics at test time, never to select routes or update the model.','- Architectures stay fixed: 64×4 branches (812,480 bytes), 32×8 branches (815,104 bytes), SGD MLP784→256→10 (814,120 bytes), and 256-unit central activity memory with eight prototype slots (824,352 bytes). Numeric state excludes gradients and temporary working buffers. Central capacity is eight throughout, so its prototype capacity is also overloaded by long streams; a larger capacity at a rebalanced parameter budget would be a separate control.','',f'![Scaling curves]({out}/scaling.png)','',
'## Full results','',
'| Dataset | Domains | Model | All-domain accuracy | First-four accuracy | Newest-domain accuracy |','|---|---:|---|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 for d in counts:
  for arm,label in zip(arms,labels):
   rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,d,arm)];m=lambda k:100*st.mean(r[k] for r in rs)
   lines.append(f'| {ds} | {d} | {label} | {m("accuracy"):.2f}% | {m("first_four_accuracy"):.2f}% | {m("new_accuracy"):.2f}% |')
lines+=['','## Degradation from 4 to 32 domains','','A positive drop means worse accuracy. First-four drop is the clearest same-input retention comparison; all-domain drop also reflects an expanding evaluation set and reduced acquisition of later mappings.','',
'| Dataset | Model | All-domain drop (pp) | First-four drop (pp) | First-four accuracy retained at32 relative to4 |','|---|---|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 for arm,label in zip(arms,labels):
  early={r['seed']:r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,4,arm)};late={r['seed']:r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,32,arm)}
  drop=lambda k:100*st.mean(early[s][k]-late[s][k] for s in early)
  ratio=100*st.mean(late[s]['first_four_accuracy']/early[s]['first_four_accuracy'] for s in early)
  lines.append(f'| {ds} | {label} | {drop("accuracy"):.2f} | {drop("first_four_accuracy"):.2f} | {ratio:.1f}% |')
lines+=['','## Learning and forgetting on the same subset','','Forgetting since acquisition is mean max(0, accuracy immediately after learning an old domain minus its final accuracy). Signed backward transfer is mean final-minus-initial accuracy for old domains. These differ from the earlier best-observed-minus-final metric, and should not be mixed numerically.','',
'| Dataset at32 domains | Model | Mean acquisition | Latest acquisition | Forgetting since acquisition (pp) | Signed backward transfer (pp) |','|---|---|---:|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 for arm,label in zip(arms,labels):
  rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,32,arm)];m=lambda k:100*st.mean(r[k] for r in rs)
  lines.append(f'| {ds} | {label} | {m("acquisition_mean"):.2f}% | {100*st.mean(r["acquisition_by_domain"][-1] for r in rs):.2f}% | {m("forgetting_since_acquisition"):.2f} | {m("backward_transfer"):.2f} |')
lines+=['','## Interpretation boundaries','','Preserving the first four domains while learning the newest poorly is a stability-plasticity trade-off, not a complete solution. A branch advantage in initial-domain retention does not automatically imply higher average accuracy or a better memory/computation trade-off. The models differ in fixed coded teaching versus learned output heads and in local usage damping versus fixed-rate SGD; these tests do not isolate branch structure from those learning-rule differences. A plasticity-matched ordinary control or stronger regularized MLP is needed before attributing retention uniquely to dendrites.','',
'Three seeds give limited uncertainty; domains within a seed are correlated observations, not independent replicates. No top-conference or biological-realism claim follows from these small benchmarks.','',
'## Reproduction','','Script: `bdh_cl/scripts/local_graph_long_stream.py`. Results: `results/local_graph_long_stream_20261013/`. Archived protocol, frozen settings, driver source, per-seed/per-domain metrics and final model states. Historical configurations are not stored or used by the learner. Final checkpoints are offline analysis artifacts.','',
'Python: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`; two CPU threads, cached real MNIST and Fashion MNIST.','']
(out/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[lines.index('## Degradation from 4 to 32 domains'):lines.index('## Learning and forgetting on the same subset')]))
