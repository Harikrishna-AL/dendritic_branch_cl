import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
base=ROOT/'results/class_incremental_branch_prototypes_20261013';follow=ROOT/'results/class_incremental_prototype_refresh_20261013'
r=json.loads((base/'results.json').read_text());q=json.loads((follow/'results.json').read_text());sel=json.loads((base/'selection.json').read_text())
ordinary=ROOT/'results/class_incremental_mlp_prototypes_20261013'
bp=json.loads((ordinary/'results.json').read_text());assert len(bp)==12 and json.loads((ordinary/'verification.json').read_text())['complete_12_models']
assert len(r)==60 and len(q)==24 and json.loads((base/'completion.json').read_text())['complete'] and json.loads((follow/'verification.json').read_text())['complete_24_models']
for a in r+q+bp:
 assert len(a['history'])==5
 for i,h in enumerate(a['history']):
  for v in h['readouts'].values():assert len(v['pair_accuracy'])==i+1 and 0<=v['accuracy']<=1

def subset(data,ds,arm):
 a=[x for x in data if x['dataset']==ds and x['arm']==arm];assert len(a)==6;return a
def seed_values(a,func):return [st.mean(func(x) for x in a if x['seed']==s) for s in sorted(set(x['seed'] for x in a))]
def metric(a,mode='flat',key='accuracy'):return seed_values(a,lambda x:x['history'][-1]['readouts'][mode][key])
def number(a,mode='flat',key='accuracy'):return 100*st.mean(metric(a,mode,key))
def bwt(x,mode):
 last=x['history'][-1]['readouts'][mode]['pair_accuracy']
 return st.mean(last[i]-x['history'][i]['readouts'][mode]['pair_accuracy'][i] for i in range(4))
labels={'mnist':'Split MNIST','fashion_mnist':'Split Fashion MNIST'}
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
for ax,ds in zip(axes,labels):
 for data,arm,name,color,style in [(q,'branch_protected','Branches, fitted prototypes','#2674a0','-'),(q,'dense_protected','Dense, fitted prototypes','#bb5845','-'),(bp,'backprop_mlp','Backprop MLP, fitted prototypes','#9b7aaf','-'),(r,'frozen','Frozen random features','#498553','-'),(r,'branch_protected','Branches, online prototypes','#2674a0','--')]:
  a=subset(data,ds,arm);v=[100*st.mean(x['history'][i]['readouts']['flat']['accuracy'] for x in a) for i in range(5)]
  ax.plot([2,4,6,8,10],v,marker='o',color=color,linestyle=style,label=name)
 ax.set_title(labels[ds]);ax.set_xticks([2,4,6,8,10]);ax.set_xlabel('Number of seen classes');ax.grid(alpha=.2);ax.set_ylim(0,100)
axes[0].set_ylabel('Accuracy among all seen classes (%)');axes[1].legend(fontsize=8,frameon=False)
fig.savefig(OUT/'prototype_accuracy.png',dpi=180);fig.savefig(OUT/'prototype_accuracy.svg');plt.close(fig)
lines=['# Class-incremental branch learning with prototype classifiers','',
'Completed on 13 October 2026. **Branches outperform the dense local-coded learner in this experiment, but both are far below frozen random features. This is a relative advantage under a restricted learning rule, not a successful general continual-learning result or a broad MLP superiority claim.**','',
'## What was tested','',
'Five successive disjoint pairs of classes: 0/1, 2/3, 4/5, 6/7, 8/9, and the reversed pair order. Each model receives two passes over the current classes, using all 60,000 training images in total per pass across the full sequence. Inference compares against every class observed so far, reaching all ten classes at the end. No task identity, correct pair, test label, saved route configuration, or replay examples enter prediction. Prototypes explicitly store class summaries; this is not memory-free learning.','',
'The branch and dense learners use the same initial 785×256 input weights, normalized inputs, tanh responses, fixed supervised teaching codes, and local squared-error weight updates. Branches train one response in each of 64 groups of four and track neighboring reliable coactivity; dense learning updates all 256 responses. Protection slows updates as reliable usage accumulates. A frozen baseline keeps the same initial random response weights unchanged.','',
'**Primary classifier:** each class has a 256-dimensional mean response vector. Both learners use exactly the same squared Euclidean nearest-prototype rule over their full 256 responses. This makes the readout comparison direct; the branch advantage in this classifier concerns selective learning/protection, not communication at inference.','',
'**Secondary branch classifiers:** choose the best-matching branch per group, with or without learned neighbor compatibility. All class hypotheses are scored; the true class never selects a route. These secondary readouts use the same final input weights and prototypes as the flat classifier but are different classifiers from the dense primary.','',
'## Prototype construction and fair-control checks','',
'1. Online means average the post-update responses of current labeled training minibatches. Repeated epochs count as repeated observations. Old-class prototypes are never refreshed after their classes leave the stream.','2. A follow-up fits each current class prototype after its feature training finishes, using one additional non-learning pass over current training images. Earlier classes are never revisited. This uses the current-stage boundary to schedule fitting but supplies no task ID to prediction. It removes within-acquisition averaging drift, while allowing later feature drift to remain.','',
'Each architecture independently selected its protected learning rate and usage scale on a 50,000/10,000 training/validation split, using one development seed per dataset and forward class order. Candidate base rates were 0.25 and 1, with scales 1,000 and 10,000; dense rates were divided by four and scales multiplied by four to adjust nominal update exposure. Official test images were not used for selection. Fresh confirmation uses three seeds per dataset and both pair orders. Orders are paired repeats, not six independent seeds. Protection-off is an ablation at the protected model’s chosen settings, not independently tuned. The prototype-fitting follow-up reuses these settings and seeds without retuning.','',
'| Dataset | Branch selected rate / usage scale | Dense selected rate / usage scale |','|---|---|---|']
for ds in labels:
 a=sel[ds]['grouped']['chosen'];b=sel[ds]['dense']['chosen'];lines.append(f'| {labels[ds]} | {a["rate"]} / {a["tau"]:.0f} | {b["rate"]} / {b["tau"]:.0f} |')
lines+=['',
'Branches store 822,800 numeric bytes; dense and frozen controls store 818,768 bytes (about 0.5% less). This includes 10×256 FP32 prototypes and ten int64 counts, adding 10,320 bytes to each underlying model. Persistent budgets exclude temporary tensors and peak working memory. Dense updates four times as many response units; this is not an equal-compute test.','',
'![Prototype accuracy across the class stream](prototype_accuracy.png)','',
'Curves average three seeds and two orders. The classifier always considers all seen classes; declining accuracy can reflect both representation drift and competition from newly introduced classes.','',
'## Main comparison: identical nearest-prototype classifier','',
'| Dataset | Prototype construction | Model | Final ten-class accuracy | Old eight classes | Newest pair | Seed SD of final accuracy |','|---|---|---|---:|---:|---:|---:|']
for ds in labels:
 for data,construction,arms in [(r,'Online',['branch_protected','dense_protected','branch_unprotected','dense_unprotected','frozen']),(q,'Fitted after current training',['branch_protected','dense_protected']),(bp,'Fitted after current training',['backprop_mlp'])]:
  for arm in arms:
   a=subset(data,ds,arm);v=metric(a)
   lines.append(f'| {labels[ds]} | {construction} | {arm} | {number(a):.2f}% | {number(a,key="old_accuracy"):.2f}% | {number(a,key="new_accuracy"):.2f}% | {100*st.stdev(v):.2f} pp |')
lines+=['', 'Seed SD is computed after averaging the two orders within each seed. The frozen classifier uses online means; because its representation never changes, these means use one consistent coordinate system.','',
'## Secondary routing and communication ablations','',
'| Dataset | Prototype construction | Protection | Flat | Grouped, no communication | Grouped + communication |','|---|---|---|---:|---:|---:|']
for ds in labels:
 for data,construction,arms in [(r,'Online',['branch_protected','branch_unprotected']),(q,'Fitted',['branch_protected'])]:
  for arm in arms:
   a=subset(data,ds,arm);lines.append(f'| {labels[ds]} | {construction} | {arm} | {number(a):.2f}% | {number(a,"grouped"):.2f}% | {number(a,"communicating"):.2f}% |')
lines+=['',
'## Conventional backpropagation MLP control','',
'A conventional 256-unit tanh MLP learns its input weights and a ten-class linear head using cross-entropy and SGD without momentum. Its features are classified using exactly the same fitted nearest-class prototypes as the primary comparison. Input normalization, initial input weights, data/order, two learning passes, validation split and current-only prototype fitting are shared. Learning rates 0.03, 0.3 and 1 were selected independently using final validation prototype accuracy. All ten outputs participate in training; no current-pair mask is used. Its 824,440-byte persistent state is about 0.2% larger than branches. This is a naive trained MLP control, not a state-of-the-art continual-learning method or an exhaustive optimizer search.','',
'## Drift and retained performance','',
'We measured each test class’s mean response at acquisition and again after later learning, only for analysis. These test means never update the model or its prototypes. Class-mean drift is a partial representation diagnostic; it does not capture every change in the feature distribution. Backward transfer below is final pair accuracy minus its accuracy when learned, averaged across the first four pairs. New classes compete at later evaluation, so negative transfer is not solely a weight-overwrite measure.','',
'| Dataset | Prototype construction | Model | Backward transfer | Old-class mean response drift (MSE) |','|---|---|---|---:|---:|']
for ds in labels:
 for data,construction,arms in [(r,'Online',['branch_protected','dense_protected','frozen']),(q,'Fitted',['branch_protected','dense_protected']),(bp,'Fitted',['backprop_mlp'])]:
  for arm in arms:
   a=subset(data,ds,arm)
   drift=st.mean(st.mean(v for c,v in x['history'][-1]['test_class_mean_drift'].items() if int(c) not in x.get('pairs',[[8,9] if x['order']=='forward' else [0,1]])[-1]) for x in a)
   # Follow-up has no pairs field; newest pair is determined by order above.
   lines.append(f'| {labels[ds]} | {construction} | {arm} | {100*st.mean(seed_values(a,lambda x:bwt(x,"flat"))):.2f} pp | {drift:.6f} |')
lines+=['',
'## What this means for the hypothesis','',
'The class-incremental comparison does reveal a substantial branch-versus-dense difference under this local coded learning rule. Therefore similar domain-incremental results did not imply the two learners would behave identically here. But the dense control performs poorly, and the branch learner also damages a representation whose initial random version already supports much higher prototype accuracy. Relative superiority over this dense learner is insufficient evidence of a practical advantage.','',
'The primary flat readout does not use branch communication. Its differences concern selective grouped training, exposure, local protection and representation drift. Secondary communication results must be interpreted separately. Fixed teaching-code coverage and active-unit count are still confounds; a conventional fixed-rate SGD MLP was independently selected and tested as an additional control below. This is still not a comparison against all ordinary learning rules or established continual-learning methods.','',
'Fitting current prototypes after training cannot prevent later changes in old-class embeddings. A useful next direction is protecting those embeddings while improving new-class discrimination, followed by stronger protection schedules for the backpropagation baseline and a sparse dense control with matched code coverage. The present result is a diagnostic lead, not proof that dendritic short-term memory solves class-incremental learning.','',
'## Reproducibility','',
'- [Main driver](../../bdh_cl/scripts/class_incremental_branch_prototypes.py), [protocol](../../results/class_incremental_branch_prototypes_20261013/protocol.json), [validation selection](../../results/class_incremental_branch_prototypes_20261013/selection.json), [60 confirmation models](../../results/class_incremental_branch_prototypes_20261013/results.json).','- [Prototype-fitting driver](../../bdh_cl/scripts/class_incremental_prototype_refresh.py), [follow-up protocol](../../results/class_incremental_prototype_refresh_20261013/protocol.json), [24 paired follow-up models](../../results/class_incremental_prototype_refresh_20261013/results.json).','- [Conventional MLP driver](../../bdh_cl/scripts/class_incremental_mlp_prototypes.py), [protocol](../../results/class_incremental_mlp_prototypes_20261013/protocol.json), [selection](../../results/class_incremental_mlp_prototypes_20261013/selection.json), [12 paired confirmation models](../../results/class_incremental_mlp_prototypes_20261013/results.json).','- Verified explicit Euclidean predictions, old prototype invariance at every stage, prototype fitting leaves input weights unchanged, complete run counts, and five-stage result matrices. Sixteen local-rule and six ordinary-MLP development models precede confirmation.','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
print('Verified 60 local confirmation +24 prototype-fitting +12 conventional MLP models; report generated.')
for ds in labels:
 for arm in ['branch_protected','dense_protected']:
  print(ds,arm,'fitted',round(number(subset(q,ds,arm)),2))
