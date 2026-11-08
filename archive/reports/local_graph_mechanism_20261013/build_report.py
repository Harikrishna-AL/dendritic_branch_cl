import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
r=json.loads((ROOT/'results/local_graph_mechanism_factorial_20261013/results.json').read_text())
c=json.loads((ROOT/'results/local_graph_recent_trace_control_20261013/results.json').read_text())
assert len(r)==84 and len(c)==24
for row in r+c:
 assert len(row['domain_accuracy'])==16
 assert abs(st.mean(row['domain_accuracy'])-row['accuracy'])<1e-9

def get(ds,comp='grouped',prot='absolute',mode='communicating',trace='recent'):
 data=c if trace=='cumulative' else r
 a=[x for x in data if x['dataset']==ds and x['protection']==prot and x['readout']==mode and (trace=='cumulative' or x['competition']==comp)]
 assert len(a)==3
 return sorted(a,key=lambda x:x['seed'])
def mean(a,k='accuracy'):return 100*st.mean(x[k] for x in a)
def delta(a,b,k='accuracy'):return [100*(x[k]-y[k]) for x,y in zip(a,b)]
labels={'mnist':'Permuted MNIST','fashion_mnist':'Permuted Fashion MNIST'}
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,layout='constrained')
for ax,ds in zip(axes,labels):
 names=['Grouped\nprotected','Global\nprotected','Grouped\nunprotected','Global\nunprotected']
 for j,(comp,prot) in enumerate([('grouped','absolute'),('global','absolute'),('grouped','off'),('global','off')]):
  for off,mode,color in [(-.18,'grouped_no_communication','#a4b5c4'),(.18,'communicating','#236fa3')]:
   a=get(ds,comp,prot,mode);v=mean(a)
   ax.bar(j+off,v,.34,color=color,label=('No communication' if off<0 else 'Communication') if j==0 else None)
   ax.scatter([j+off]*3,[x['accuracy']*100 for x in a],s=12,color='#152634',zorder=3)
 ax.set_xticks(range(4),names);ax.set_title(labels[ds]);ax.set_ylim(0,85);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Final accuracy across all 16 domains (%)');axes[1].legend(frameon=False)
fig.savefig(out/'mechanism_ablation.png',dpi=180);fig.savefig(out/'mechanism_ablation.svg');plt.close(fig)
lines=['# What do branch competition, communication and recent memory contribute?', '',
'Completed on 13 October 2026. These experiments identify contributions inside the current model. They do **not** establish a general advantage over ordinary MLPs or a uniquely biological dendritic mechanism.', '',
'## Main findings', '',
'- Grouped competition outperformed global competition with the same input weights, teaching codes, active-unit count and decoder.',
'- Learned neighbor compatibility improved retrieval at unchanged final input weights. This improvement also occurred with plasticity protection switched off, so protection alone cannot explain it.',
'- A dense network with the same local coded teaching rule matched or exceeded protected branches on overall accuracy. Branches retained somewhat higher early-domain accuracy under protection, but had lower newest-domain accuracy.',
'- Recent error memory mostly changed the stability–plasticity trade-off. It did not consistently outperform cumulative error memory when protection was enabled.', '',
'## Experiment and interpretation', '',
'Each dataset has 16 domains: the original images followed by 15 pixel permutations. All domains use the same ten classes. Each current domain receives two passes over 60,000 training images. Final evaluation uses all 10,000 test images per domain. Three fresh paired seeds per dataset use identical initial input weights, permutations and minibatch orders across arms. Training and inference receive no task IDs; training receives class labels for the local teaching signal. No replay, example reservoir, input/configuration archive or task-boundary reset is used.', '',
'The model has 64 groups of four tanh responses (256 input-response units). A fixed class code gives every group a desired response during training. Grouped competition trains one branch per group. If the best recent error exceeds 0.4, it recruits the least-used branch in that group. Global competition selects 64 units across the full layer instead; it keeps the same structured decoder to isolate **training competition**. Neighbor coactivity records which adjacent branches were reliable together. The communicating decoder scores class hypotheses jointly with compatible branch routes. This is an exact compatibility decoder, not an experimentally established biological communication process.', '',
'Protection reduces each selected unit’s learning rate as reliable usage accumulates: rate = 1 / (1 + usage / 10,000). Turning protection off leaves the other mechanisms intact. Turning communication off changes only inference on the same final weights. Settings were fixed before scoring this cohort.', '',
'All-domain accuracy averages final performance on all 16 domains. First-four accuracy measures final performance on the earliest four domains; it is a retained-accuracy measure, not by itself a forgetting score. Newest accuracy measures acquisition of the last domain.', '',
'![Controlled competition and communication ablation](mechanism_ablation.png)', '',
'Bars are seed means; dots are individual seeds. Each communication comparison reuses the same final trained weights.', '',
'## Paired communication effects', '',
'| Dataset | Protection | Grouped communication gain | Global communication gain | Grouping × communication interaction |',
'|---|---|---:|---:|---:|']
for ds in labels:
 for prot in ['absolute','off']:
  dg=delta(get(ds,'grouped',prot),get(ds,'grouped',prot,'grouped_no_communication'))
  du=delta(get(ds,'global',prot),get(ds,'global',prot,'grouped_no_communication'))
  interaction=[x-y for x,y in zip(dg,du)]
  lines.append(f'| {labels[ds]} | {prot} | {st.mean(dg):.2f} ± {st.stdev(dg):.2f} pp | {st.mean(du):.2f} pp | {st.mean(interaction):.2f} pp |')
lines+=['', 'The ± term is sample standard deviation across three paired differences, not a confidence interval. Every grouped communication gain was positive across all three seeds in both protection settings and datasets. This supports a functional interaction between grouped learning and learned neighbor compatibility within this model. Three seeds and two related image benchmarks are insufficient for a broad statistical or novelty claim.', '',
'## Complete factorial results', '', '| Dataset | Competition | Protection | Readout | All domains | First four | Newest |', '|---|---|---|---|---:|---:|---:|']
for ds in labels:
 for comp in ['grouped','global','dense']:
  for prot in ['absolute','off']:
   for mode in (['dense_fixed_readout'] if comp=='dense' else ['communicating','grouped_no_communication','unstructured_top64']):
    a=get(ds,comp,prot,mode)
    lines.append(f'| {labels[ds]} | {comp} | {prot} | {mode} | {mean(a):.2f}% | {mean(a,"first_four"):.2f}% | {mean(a,"newest"):.2f}% |')
lines+=['',
'Grouped/global arms each store 812,480 bytes of numeric persistent state. The dense coded control stores 808,448 bytes, about 0.5% less, with the same input weights and teaching codes but no graph. It updates and reads all 256 units rather than 64 selected units. Its learning rate is 0.25 and protection scale 40,000 to compensate for four times the update exposure; these are nominal adjustments, not independently optimized settings. Its compute is larger per training step. This control is an ordinary hidden layer with a fixed coded linear output, **not** the strongest independently tuned learned-head MLP. The separate slowing-SGD MLP benchmark remains necessary.', '',
'## Recent versus cumulative error memory', '',
'The recent trace stores the previous minibatch’s local errors. The cumulative control uses the same arrays and averages errors from all previous presentations: E_new = (1 − α) E_old + α E_batch, with α = batch_size / (past_presentations + batch_size). No historical images are revisited. Historical errors were measured under earlier weights, so this is a cumulative historical-error control, not an exact current loss on past data. These follow-ups reuse the factorial seeds and are not an independent confirmation cohort. Both use the communicating decoder below.', '',
'| Dataset | Protection | Error trace | All domains | First four | Newest |', '|---|---|---|---:|---:|---:|']
for ds in labels:
 for prot in ['absolute','off']:
  for trace in ['recent','cumulative']:
   a=get(ds,prot=prot,trace=trace)
   lines.append(f'| {labels[ds]} | {prot} | {trace} | {mean(a):.2f}% | {mean(a,"first_four"):.2f}% | {mean(a,"newest"):.2f}% |')
lines+=['',
'Without protection, recent memory preserves substantially higher first-four accuracy but learns the newest domain less well. With protection, recent/cumulative overall accuracy is similar, and cumulative memory slightly improves Fashion overall accuracy. This does **not** demonstrate that short-term memory is necessary for continual learning. The trace affects training selection; inference here uses final weights and accumulated coactivity, rather than an adaptive short-term test-time state.', '',
'## What can be claimed about an ordinary MLP?', '',
'A defensible scoped statement is: **On our sequential permuted-MNIST and permuted-Fashion-MNIST streams, the branch model retained higher accuracy on the earliest domains than a similarly sized MLP trained with constant-rate SGD. This advantage was not consistent against validation-selected slowing-SGD MLPs.**', '',
'In the preceding 32-domain study, novelty branches versus constant-rate MLP retained 58.53% versus 27.09% on the earliest four MNIST domains, and 60.80% versus 29.22% on Fashion. However, a validation-selected slowing MLP achieved 63.37% and 64.72%, respectively, and also higher all-domain and newest-domain means. That study used a separate paired cohort; its numbers must not be treated as direct comparisons with the present 16-domain experiments. See [plasticity and ordinary controls](../local_graph_plasticity_20261013/REPORT.md).', '',
'## Attribution limits and next discriminating test', '',
'An ordinary 256-unit hidden layer plus the **same** grouping, gates, local memory and decoder computes the same function as this branch implementation. We verified that equivalence numerically. That augmented implementation is not a plain MLP, but it prevents attributing the effect to the anatomical name “dendrite.” The supported contribution is the computational organization and rules.', '',
'Global competition can give uneven coverage of the fixed teaching codes and fewer reliable adjacent pairs. Grouped competition guarantees one active response for every code coordinate. Thus code coverage, not only branch specialization, can explain its advantage. Dense controls also change active count and output aggregation; they are meaningful controls, not a pure single-factor comparison.', '',
'The next useful test is to vary neighbor topology and preserve teaching-code coverage independently of branch grouping, then compare against a separately tuned sparse ordinary learner at matched storage and acquisition. Learned versus shuffled compatibility should be tested on these same final weights. Such tests can distinguish useful compatibility structure from a generic sparsity or fixed-code effect. The present work already establishes a within-model communication benefit but not a general dendritic anti-forgetting advantage.', '',
'## Reproducibility and verification', '',
'- [Factorial driver](../../bdh_cl/scripts/local_graph_mechanism_factorial.py), [protocol](../../results/local_graph_mechanism_factorial_20261013/protocol.json), [84 result rows](../../results/local_graph_mechanism_factorial_20261013/results.json). Thirty-six trained models; multiple inference readouts are paired evaluations, not independent runs.',
'- [Cumulative driver](../../bdh_cl/scripts/local_graph_recent_trace_control.py), [protocol](../../results/local_graph_recent_trace_control_20261013/protocol.json), [24 result rows](../../results/local_graph_recent_trace_control_20261013/results.json). Twelve additional trained models, with two readouts each.',
'- Verified grouped training matches the prior novelty implementation; zero-communication decoding matches the reference decoder; ordinary hidden-layer routing is numerically equivalent; dense coded predictions equal the fixed linear output; cumulative and recent models match exactly on the first batch; all reported averages agree with their 16 domain values.',
'- Persistent-state budgets exclude temporary tensors, gradients and peak working memory. This study does not establish a runtime or equal-FLOP advantage.', '']
(out/'REPORT.md').write_text('\n'.join(lines))
progress=ROOT/'results/conference_research_progress_20261013/RESEARCH_PROGRESS.md'
p=progress.read_text();marker='## Controlled competition, communication and recent-memory ablations\n'
if marker not in p:
 anchor='## New protection/allocation experiment and stronger ordinary controls'
 note='''## Controlled competition, communication and recent-memory ablations

Completed 36 trained factorial models plus 12 cumulative-trace follow-ups on 16 domains, three fresh paired seeds per dataset. Grouped communication gains are 3.09 pp (MNIST) and 1.62 pp (Fashion) under protection, and 4.05/2.73 pp without protection; all grouped gains are positive in each seed. Grouped competition also outperforms global competition with identical structured inference. This supports functional grouped-learning/compatibility effects beyond protection alone.

A dense coded local learner reaches 68.12%/64.88% protected all-domain accuracy versus branches 66.49%/64.70%. Recent error memory chiefly improves early retention without protection at the cost of new acquisition; under protection cumulative memory matches or slightly exceeds overall accuracy. Ordinary units with the same routing are numerically equivalent to the branch implementation. Uneven teaching-code coverage remains a grouping confound. No uniquely dendritic effect or broad MLP superiority is established.

Full methods, paired gains, all controls and trace results: `../../reports/local_graph_mechanism_20261013/REPORT.md`.

'''
 progress.write_text(p.replace(anchor,note+anchor,1))
print('Verified 108 evaluation rows; generated report and figure.')
