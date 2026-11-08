import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;base=ROOT/'results/sequential_prototype_representation_20261013'
r=json.loads((base/'results.json').read_text());d=json.loads((base/'refit_diagnostic.json').read_text());c=json.loads((base/'completion.json').read_text());assert c['complete'] and c['all_storage_equal'] and len(r)==len(d)==48
labels={'mnist':'Split MNIST','fashion_mnist':'Split Fashion MNIST'};names={'dense':'Dense prototype learner','grouped':'Grouped branch updates','global_sparse':'Global sparse updates','frozen':'Frozen random features'};colors={'dense':'#b15745','grouped':'#236e9e','global_sparse':'#9372aa','frozen':'#477d4e'}
def get(data,ds,arm):
 a=[x for x in data if x['dataset']==ds and x['arm']==arm];assert len(a)==6;return a
def seeds(a,func):return [st.mean(func(x) for x in a if x['seed']==s) for s in sorted(set(x['seed'] for x in a))]
def avg(a,func):return 100*st.mean(seeds(a,func))
for x in r:
 assert len(x['history'])==5 and x['numeric_bytes']==814160
 for i,h in enumerate(x['history']):assert len(h['final']['pair_accuracy'])==i+1
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
for ax,ds in zip(axes,labels):
 for arm in names:
  a=get(r,ds,arm);values=[100*st.mean(x['history'][i]['final']['accuracy'] for x in a) for i in range(5)]
  ax.plot([2,4,6,8,10],values,marker='o',color=colors[arm],label=names[arm])
 ax.set_title(labels[ds]);ax.set_xlabel('Number of seen classes');ax.set_xticks([2,4,6,8,10]);ax.set_ylim(0,100);ax.grid(alpha=.2)
axes[0].set_ylabel('Accuracy across all seen classes (%)');axes[1].legend(frameon=False,fontsize=8)
fig.savefig(OUT/'sequential_accuracy.png',dpi=180);fig.savefig(OUT/'sequential_accuracy.svg');plt.close(fig)
lines=['# Sequential prototype-distance representation test','',
'Completed on 13 October 2026. **The joint-trained objective transfers poorly to sequential learning with old prototypes unchanged. All three learned variants suffer severe forgetting; frozen features remain substantially stronger.** This tests the prototype-gradient version without protection or communication, not the earlier complete local branch-memory model.','',
'## Protocol','',
'Five disjoint pairs of MNIST/Fashion MNIST classes arrive sequentially. Both forward pair order (0/1 → 2/3 → 4/5 → 6/7 → 8/9) and reversed order are evaluated. Each current pair receives ten learning epochs, matching the joint study’s total 600,000 gradient presentations per full sequence. At every test, all classes observed so far compete; old classes remain eligible. No correct task, pair or test label enters prediction.','',
'All arms use the same L2-normalized inputs, 256 tanh responses, nearest-class prototype classifier, paired initialization and current minibatch order. Dense updates all response units; grouped updates the largest-gradient unit in each of 64 four-unit groups; global sparse updates the largest-gradient 64 units globally. Frozen features never change. All arms have exactly 814,160 bytes of numeric persistent state, including class prototypes/counts. Gradients are computed for the full layer before sparse masking; no sparse-compute or biologically local-learning claim is made.','',
'Rates and temperatures are taken unchanged from independent joint-validation selection. There is no sequential retuning or test-based selection. Three seeds per dataset reuse the joint cohort to enable a paired follow-up; this is not an independent replication. The two orders are paired repeats, not six independent seeds. All 60,000 training images and 10,000 test images are used.','',
'Only current-class prototypes are bootstrapped before learning, updated online from current labeled minibatches, and refitted after each current epoch. Old prototypes and their counts remain exactly unchanged after their classes leave. The loss compares each current example against **all seen classes**, including old prototypes. Training-stage boundaries schedule the current fitting passes; this is task-ID-free inference but not a boundary-free streaming protocol. No old images, replay reservoir or historical configuration bank is used by the primary learner.','',
'## Primary replay-free result','',
'| Dataset | Model | Final ten-class accuracy | Old eight classes | Newest pair | Final accuracy seed SD |','|---|---|---:|---:|---:|---:|']
for ds in labels:
 for arm in names:
  a=get(r,ds,arm);v=seeds(a,lambda x:x['accuracy'])
  lines.append(f'| {labels[ds]} | {names[arm]} | {100*st.mean(v):.2f}% | {avg(a,lambda x:x["history"][-1]["final"]["old_accuracy"]):.2f}% | {avg(a,lambda x:x["history"][-1]["final"]["new_accuracy"]):.2f}% | {100*st.stdev(v):.2f} pp |')
lines+=['', 'Seed SD is computed after averaging the two orders within each seed. Old/new accuracies are pair averages; all-class accuracy is weighted by official test class sizes. Near 20% overall accuracy can therefore coexist with near-perfect newest-pair classification and near-zero old-class accuracy.','',
'![Sequential accuracy](sequential_accuracy.png)','',
'## Representation drift and backward transfer','',
'Test class mean responses were recorded at acquisition and after later learning, only for analysis. No test-derived statistic enters learning or prototypes. Drift is mean squared movement of old-class mean responses. It is a proxy, not a complete information-retention metric. Backward transfer is final minus acquisition accuracy, averaged across the first four pairs; this also includes competition from newly introduced classes.','',
'| Dataset | Model | Old-class mean drift (MSE) | Backward transfer |','|---|---|---:|---:|']
for ds in labels:
 for arm in names:
  a=get(r,ds,arm)
  drift=st.mean(seeds(a,lambda x:st.mean(v for cls,v in x['history'][-1]['classmean_drift'].items() if int(cls) not in x['pairs'][-1])))
  lines.append(f'| {labels[ds]} | {names[arm]} | {drift:.6f} | {avg(a,lambda x:x["backward_transfer"]):.2f} pp |')
lines+=['',
'## Offline prototype-refit diagnostic — uses old training data','',
'For analysis only, the final feature weights were held unchanged and all ten class prototypes were rebuilt using all labeled training images, including historical classes. No gradients were applied and no test labels selected a route. This asks whether useful class separation remains in the final responses despite stale old prototypes. **These numbers are not valid replay-free continual-learning performance:** the diagnostic accesses old training data. Original saved model states and primary results remain unchanged.','',
'| Dataset | Model | Primary accuracy | Offline all-class prototype refit | Old eight classes after refit |','|---|---|---:|---:|---:|']
for ds in labels:
 for arm in names:
  a=get(d,ds,arm)
  lines.append(f'| {labels[ds]} | {names[arm]} | {avg(a,lambda x:x["original"]["accuracy"]):.2f}% | {avg(a,lambda x:x["offline_allclass_refit"]["accuracy"]):.2f}% | {avg(a,lambda x:x["offline_allclass_refit"]["old_accuracy"]):.2f}% |')
lines+=['',
'A recovery after refitting means final features still contain class information accessible to the same nearest-mean classifier when fresh class summaries are supplied. It does not prove the old representation was preserved, that prototypes can be updated autonomously without old data, or that branches uniquely retained information. Dense recovery must be considered alongside branch recovery.','',
'## What this means','',
'Joint prototype-loss learning successfully acquired useful features, but without stability mechanisms it did not preserve accessible old-class performance during sequential learning. Updating fewer response units, even one per group, was insufficient under the fixed transferred settings. This rules out a simple claim that prototype-aligned learning plus grouping alone solves forgetting. It does not rule out the original hypothesis involving branch-specific memory, selective protection and communication, which are absent here.','',
'The next discriminating intervention is protection under the same prototype objective, with matched dense/global sparse protection controls and explicit old/new accuracy reporting. Any prototype correction or frozen-plus-learnable residual mechanism should have a corresponding ordinary control. The current offline refit diagnostic is evidence about remaining features, not an autonomous solution.','',
'## Reproducibility and verification','',
'- [Sequential driver](../../bdh_cl/scripts/sequential_prototype_representation.py), [protocol](../../results/sequential_prototype_representation_20261013/protocol.json), [48 primary models](../../results/sequential_prototype_representation_20261013/results.json), [final model files](../../results/sequential_prototype_representation_20261013/models).','- [Refit diagnostic driver](../../bdh_cl/scripts/sequential_prototype_refit_diagnostic.py), [diagnostic protocol](../../results/sequential_prototype_representation_20261013/refit_protocol.json), [48 diagnostic evaluations](../../results/sequential_prototype_representation_20261013/refit_diagnostic.json).','- Checked old prototype/count invariance at every stage, old-class eligibility, unseen-class exclusion, exact agreement with joint updates when all classes are observed, frozen-weight invariance, saved-primary prediction reproduction, unchanged feature weights during diagnostic refitting, complete counts and equal persistent state.','- Stored histories/test means are reporting artifacts, not model memory or learner input. Each learned epoch adds a current-data-only prototype-fitting pass; bootstrap adds another. Temporary gradient/working memory and evaluation artifacts are excluded from model storage.','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
print('48 primary and48 diagnostic models verified; report generated.')
for ds in labels:
 for arm in names:
  aa=get(r,ds,arm);bb=get(d,ds,arm);print(ds,arm,'primary',round(avg(aa,lambda x:x['accuracy']),2),'refit',round(avg(bb,lambda x:x['offline_allclass_refit']['accuracy']),2))
