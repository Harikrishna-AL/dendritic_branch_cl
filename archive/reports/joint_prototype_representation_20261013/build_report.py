import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
base=ROOT/'results/joint_prototype_representation_20261013';ref=ROOT/'results/joint_fixed_code_reference_20261013'
r=json.loads((base/'results.json').read_text());s=json.loads((base/'selection.json').read_text());a=json.loads((ref/'results.json').read_text())
assert len(r)==24 and len(a)==12 and json.loads((base/'completion.json').read_text())['complete'] and json.loads((ref/'completion.json').read_text())['complete']
assert all(x['numeric_bytes']==814160 for x in r)
labels={'mnist':'MNIST','fashion_mnist':'Fashion MNIST'}
colors={'dense':'#b15745','grouped':'#236e9e','global_sparse':'#9372aa','frozen':'#477d4e'}
readable={'dense':'Dense prototype learning','grouped':'Grouped branch updates','global_sparse':'Global sparse updates','frozen':'Frozen random features'}
def get(data,ds,arm):
 v=sorted([x for x in data if x['dataset']==ds and x['arm']==arm],key=lambda x:x['seed']);assert len(v)==3;return v
def mean(v):return 100*st.mean(x['accuracy'] for x in v)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
for ax,ds in zip(axes,labels):
 for arm in ['dense','grouped','global_sparse','frozen']:
  v=get(r,ds,arm)
  if arm=='frozen':ax.axhline(mean(v),color=colors[arm],linestyle='--',label=readable[arm])
  else:
   xx=[h['epoch'] for h in v[0]['history']];yy=[100*st.mean(x['history'][i]['accuracy'] for x in v) for i in range(len(xx))]
   ax.plot(xx,yy,marker='o',color=colors[arm],label=readable[arm])
 ax.set_title(labels[ds]);ax.set_xlabel('Joint training epochs');ax.set_xticks([0,2,5,10]);ax.set_ylim(60,100);ax.grid(alpha=.2)
axes[0].set_ylabel('Ten-class test accuracy (%)');axes[1].legend(frameon=False,fontsize=8)
fig.savefig(OUT/'joint_accuracy.png',dpi=180);fig.savefig(OUT/'joint_accuracy.svg');plt.close(fig)
lines=['# Joint prototype-distance representation learning','',
'Completed on 13 October 2026. This is an **acquisition test with all ten classes available together**, not continual learning. It tests whether gradient-based prototype learning can improve on the initial random representation.','',
'## Main result','',
'| Dataset | Frozen random features | Grouped branch updates | Dense updates | Global sparse updates |','|---|---:|---:|---:|---:|']
for ds in labels:lines.append(f'| {labels[ds]} | {mean(get(r,ds,"frozen")):.2f}% | {mean(get(r,ds,"grouped")):.2f}% | {mean(get(r,ds,"dense")):.2f}% | {mean(get(r,ds,"global_sparse")):.2f}% |')
lines+=['', 'All cells average three fresh paired seeds; every learned model trains for ten epochs. All arms store exactly 814,160 bytes of numeric persistent state. Class prototypes are explicit summary memory and included. The primary classifier is identical across arms.','',
'![Joint-training accuracy](joint_accuracy.png)','',
'## Experiment','',
'Cached real MNIST and Fashion MNIST each supply 60,000 training images and 10,000 official test images. All ten classes are shuffled together. Images are normalized to unit length and augmented with a bias input. Each model has 256 tanh responses, with initial input weights sampled at standard deviation 0.02. Paired models share initial weights and minibatch order. There is no pretrained encoder, replay, task ID or learned output head.','',
'Each class has a 256-dimensional mean response prototype μ_c. The training objective is cross-entropy over distance scores:','',
'```text\nh(x) = tanh([normalize(x), 1] W)\ns_c(x) = -||h(x) - μ_c||² / T\np(c | x) = softmax(s(x))_c\nL = -log p(y | x)\n```','',
'Prototypes are treated as detached targets when differentiating the loss. They are initialized from training-class means before learning. After each weight update, the current minibatch updates its class prototypes with EMA coefficient 0.1. After each epoch, all prototypes are refitted from the training images under the current weights. Frozen models fit their prototypes once from the same training data. Test labels are used only for scoring. Unlike the earlier sequential experiment, all classes remain available, so historical prototype staleness can be removed.','',
'## What “branches” means in this test','',
'- **Dense:** update all 256 response units using the prototype-loss gradient.','- **Grouped:** organize units into 64 groups of four; in each minibatch update the unit with the largest input-weight gradient norm in each group. Exactly 64 units update.','- **Global sparse:** update the 64 largest-gradient units over the whole layer without one-per-group constraints.','- **Frozen:** keep the initial input weights unchanged; learn only class prototypes.','',
'This is a new gradient-based acquisition reference. Group structure restricts **learning**, while the prototype classifier reads the same full 256-dimensional response in all arms. It does not implement branch competition at inference, neighbor communication, usage protection, or the historical short-term error trace. Gradients are computed for all units before masking; this is neither a biological local-learning demonstration nor a sparse-gradient compute saving. It answers whether prototype-aligned learning is viable before trying a local approximation.','',
'## Validation and confirmation','',
'Each dataset uses one separate development seed (441/451), with 50,000 fitting images and 10,000 held-out training images, stratified at 1,000 per class. Each trainable arm independently selects among learning rates 0.03/0.3 and temperatures 0.03/0.3, based only on validation accuracy after ten epochs. Official test data never selects settings. Confirmation uses fresh seeds 442–444 and 452–454 with all 60,000 training images. Test checkpoints 0/2/5/10 were fixed in advance; no best-test-checkpoint selection. There are 24 development models, 18 learned confirmation models and six frozen references.','',
'| Dataset | Arm | Selected rate | Selected temperature | Final test accuracy ± seed SD | Gain over paired frozen model |','|---|---|---:|---:|---:|---:|']
for ds in labels:
 frozen=get(r,ds,'frozen')
 for arm in ['dense','grouped','global_sparse']:
  v=get(r,ds,arm);setting=s[ds][arm]['chosen'];g=[100*(x['accuracy']-y['accuracy']) for x,y in zip(v,frozen)]
  lines.append(f'| {labels[ds]} | {arm} | {setting["lr"]} | {setting["temperature"]} | {mean(v):.2f}% ± {100*st.stdev(x["accuracy"] for x in v):.2f} pp | {st.mean(g):.2f} pp |')
lines+=['', 'The uncertainty term is sample standard deviation across three seeds, not a confidence interval. The paired gains are descriptive; no broad statistical significance claim is made.','',
'## Previous fixed-code learner under joint training','',
'A separate diagnostic trains the preceding fixed-code local learner jointly on all ten classes using the same fresh seeds, initial input weights, data order and ten-epoch budget. It uses its earlier class-incremental validation settings without joint retuning. The final classifier is again full-response nearest-class prototypes refitted after each epoch. This is not a pure loss ablation: branch selection, local weight updates, usage protection and learning-rate choices also differ.','',
'| Dataset | Previous grouped fixed-code learner | Previous dense fixed-code learner |','|---|---:|---:|']
for ds in labels:lines.append(f'| {labels[ds]} | {mean(get(a,ds,"grouped")):.2f}% | {mean(get(a,ds,"dense")):.2f}% |')
lines+=['',
'This reference prevents attributing the difference between the earlier low sequential accuracy and the new joint accuracy solely to the prototype objective. Joint access to all classes removes interference and permits prototype refresh, both of which matter. A causal objective comparison would keep selection, protection, optimizer and tuning budget fixed while changing only the loss.','',
'## Interpretation and next step','',
'Exceeding the paired frozen baseline demonstrates useful representation learning in this joint setting. It does not demonstrate autonomous routing, anti-forgetting, or a uniquely dendritic advantage. Compare grouped versus global sparse and dense results before attributing gains to group structure; all can implement the same prototype objective.','',
'The next gate is to freeze the selected settings and run sequential class learning with old prototypes left unchanged, reporting both acquisition and retention. Protection and local feedback should then be introduced as separately tested changes. A frozen-plus-learnable residual design is another candidate, but must have a matched dense counterpart and an explicit memory budget.','',
'## Reproducibility and checks','',
'- [Main driver](../../bdh_cl/scripts/joint_prototype_representation.py), [protocol](../../results/joint_prototype_representation_20261013/protocol.json), [validation selection](../../results/joint_prototype_representation_20261013/selection.json), [confirmation results](../../results/joint_prototype_representation_20261013/results.json).','- [Fixed-code diagnostic driver](../../bdh_cl/scripts/joint_fixed_code_reference.py), [reference protocol](../../results/joint_fixed_code_reference_20261013/protocol.json), [reference results](../../results/joint_fixed_code_reference_20261013/results.json).','- Verified distance logits match explicit squared-distance softmax, dense updates match the loss gradient, sparse updates leave inactive weights untouched, grouped selection activates exactly one response per group, frozen weights remain unchanged, and all expected result counts and persistent budgets are complete.','- Each learned epoch has one gradient-training pass and one non-learning prototype-fitting pass. Bootstrap prototype fitting adds another training-data pass. Reported gradient training time and subsequent fitting time are separate; bootstrap/evaluation time is excluded. No runtime or peak-working-memory advantage is claimed.','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
print('Report and figure generated; all 36 confirmation/reference models verified.')
for ds in labels:
 for arm in ['frozen','grouped','dense','global_sparse']:print(ds,arm,round(mean(get(r,ds,arm)),2))
