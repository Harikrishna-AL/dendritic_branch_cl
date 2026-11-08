"""Summarize all somatic-fix experiments without choosing settings from test data."""
import json
import math
import os
import statistics as st
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/bdh_somatic_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/bdh_somatic_xdg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/somatic_consensus_20261021'
HERE=Path(__file__).resolve().parent
adam=json.loads((DATA/'results.json').read_text())
sgd=json.loads((DATA/'sgd_results.json').read_text())
joint=json.loads((DATA/'joint_results.json').read_text())
audit=json.loads((DATA/'feedback_transition_audit.json').read_text())
assert (len(adam),len(sgd),len(joint),len(audit))==(160,60,20,640)
select={'adam':json.loads((DATA/'selection.json').read_text()),'sgd':json.loads((DATA/'sgd_selection.json').read_text())}
sources={'adam':adam,'sgd':sgd}
SEEDS=list(range(701,711))
TC=2.262157163
FAMILIES=['baseline','feedback','somatic_trace','somatic_trace_feedback','somatic_cue','somatic_cue_feedback','global64_baseline','global64_feedback']


def chosen(row,name,optimizer='adam'):
    ss=select[optimizer]['readouts'][row['arm']]
    options=[r for r in row['methods'] if r['family']==name]
    return next(r for r in options if any(all(r[k]==spec[k] for k in spec) for spec in ss))


def raw_setting(row,keep,feedback=False,rank='trace'):
    return next(r for r in row['methods'] if r['cue']=='local' and r['keep']==keep and r['rank']==rank and r['feedback']==feedback)


def describe(values):
    data=list(values.values());mu=st.mean(data);sd=st.stdev(data);half=TC*sd/math.sqrt(len(data))
    return {'mean':mu,'seed_sd':sd,'low':mu-half,'high':mu+half,'seed_values':values}


def values(optimizer,arm,family,field='accuracy'):
    return {seed:100*st.mean(chosen(r,family,optimizer)[field] for r in sources[optimizer] if r['arm']==arm and r['seed']==seed) for seed in SEEDS}


def paired(optimizer,arm,a,b):
    aa=values(optimizer,arm,a);bb=values(optimizer,arm,b)
    return describe({seed:aa[seed]-bb[seed] for seed in SEEDS})


def fmt(d):return f"{d['mean']:+.2f} [{d['low']:+.2f}, {d['high']:+.2f}]"


summary={};comparisons={}
for optimizer,records in sources.items():
    summary[optimizer]={}
    for arm in dict.fromkeys(r['arm'] for r in records):
        rr=[r for r in records if r['arm']==arm]
        methods={name:{field:describe(values(optimizer,arm,name,field)) for field in ['accuracy','old_accuracy','new_accuracy','first_pass_top2_true_class_coverage']} for name in FAMILIES}
        summary[optimizer][arm]={'methods':methods,
            'acquisition':100*st.mean(r['acquisition'] for r in rr),
            'historical':100*st.mean(r['diagnostics']['historical']['accuracy'] for r in rr),
            'readout_numeric_bytes':{name:chosen(rr[0],name,optimizer)['model_and_memory_bytes'] for name in FAMILIES},
            'chosen_settings':select[optimizer]['readouts'][arm]}
    arm='patch5_independent_keep24'
    comparisons[optimizer]={
        'feedback_minus_baseline':paired(optimizer,arm,'feedback','baseline'),
        'trace_suppression_minus_baseline':paired(optimizer,arm,'somatic_trace','baseline'),
        'trace_suppression_and_feedback_minus_baseline':paired(optimizer,arm,'somatic_trace_feedback','baseline'),
        'cue_suppression_minus_baseline':paired(optimizer,arm,'somatic_cue','baseline')}

transition={}
for family in ['feedback','somatic_trace_feedback','somatic_cue_feedback']:
    rr=[r for r in audit if r['arm']=='patch5_independent_keep24' and r['family']==family]
    transition[family]={key:100*st.mean(r[key] for r in rr) for key in ['before_accuracy','after_accuracy','wrong_to_correct','correct_to_wrong','net_change','first_top2_true_class_coverage','final_prediction_outside_initial_top2']}
direct=100*st.mean(r['direct'][0]['accuracy'] for r in adam)
jointmeans={arm:100*st.mean(r['joint_accuracy'] for r in joint if r['arm']==arm) for arm in dict.fromkeys(r['arm'] for r in joint)}
payload={'models':summary,'paired_comparisons':comparisons,'feedback_transitions_adam_patch5':transition,'direct_classifier_accuracy':direct,'joint_capacity':jointmeans,
         'uncertainty':'Paired seed t intervals df9; orders averaged within seed, unadjusted multiple comparisons.'}
(DATA/'summary.json').write_text(json.dumps(payload,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
families=['baseline','feedback','somatic_trace','somatic_trace_feedback']
labels=['All groups','All groups\n+ class feedback','Drop weak\ngroups','Drop weak groups\n+ class feedback']
colors=['#8898a7','#287f8f','#c49862','#6e9d8e']
for ax,opt in zip(axes,['adam','sgd']):
    for i,(family,color) in enumerate(zip(families,colors)):
        d=summary[opt]['patch5_independent_keep24']['methods'][family]['accuracy']
        ax.bar(i,d['mean'],color=color)
        for seed in SEEDS:ax.plot(i+(seed-705.5)*.018,d['seed_values'][seed],'.',color='#1c2732',ms=3,alpha=.7)
    ax.axhline(direct,color='#ad5d30',ls='--',label='Direct classifier')
    ax.set_ylim(0,100);ax.set_xticks(range(4),['All groups','Class\nfeedback','Drop groups','Drop +\nfeedback']);ax.set_title('Adam' if opt=='adam' else 'SGD without optimizer history')
    ax.legend(fontsize=8,loc='upper right')
axes[0].set_ylabel('Automatic ten-digit test accuracy (%)')
fig.suptitle('5×5 patch groups: suppression and class feedback\nTen seeds; both class orders; validation-selected suppression fraction',fontsize=12)
fig.tight_layout();fig.savefig(HERE/'fixes_accuracy.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.4,4.8))
arm='patch5_independent_keep24'
for family,color,label in [('baseline','#8898a7','All groups'),('feedback','#287f8f','Class feedback'),('somatic_trace','#c49862','Drop weak groups'),('somatic_trace_feedback','#6e9d8e','Drop + feedback')]:
    d=summary['adam'][arm]['methods'][family]
    x=labels.index({'baseline':'All groups','feedback':'All groups\n+ class feedback','somatic_trace':'Drop weak\ngroups','somatic_trace_feedback':'Drop weak groups\n+ class feedback'}[family])
    ax.bar(x-.17,d['old_accuracy']['mean'],.34,color=color,label=label)
    ax.bar(x+.17,d['new_accuracy']['mean'],.34,color=color,hatch='//',alpha=.65)
ax.set_xticks(range(4),['All groups','Feedback','Drop weak','Drop + feedback']);ax.set_ylim(0,100)
ax.set_ylabel('Accuracy (%)');ax.set_title('Earlier eight digits (solid) versus newest pair (hatched)')
fig.tight_layout();fig.savefig(HERE/'old_new_accuracy.png',dpi=180);plt.close(fig)

lines=['# Somatic suppression, class feedback, and compact patch groups',
 '', 'Completed 21 October 2026. These experiments test the three proposed fixes separately and together. They include 160 Adam sequential runs, 60 SGD sequential controls, and 20 joint-capacity references. All confirmation uses ten paired initialization seeds and both digit-pair orders; these are not 240 independent initializations.',
 '', '**Main finding:** class feedback improves the compact independent-output model from 47.91% to 51.65% (+3.74 percentage points; 95% paired-seed interval +2.56 to +4.93). Trace-ranked somatic suppression instead reduces it to 39.00%; suppression plus feedback reaches 40.54%. The direct classifier remains stronger at 85.53%.',
 '', 'Patch restriction meets the model-plus-local-memory budget at 18,520 bytes. Adam raises the full persistent state to 46,264 bytes. SGD without momentum meets the stricter 18,520-byte continuing-learning budget, but reaches only 45.99% with feedback. Thus the state-saving goal is achievable here; the joint state-and-accuracy advantage is not.',
 '', '## Definitions and exact implementation',
 '', '- **Group and branch:** 24 groups each have four learned tanh responses. Each group initially selects one branch. Independent output connections give each branch its own weight for each digit class, implementing W[class, group, branch]. Shared-output alternatives are explicit controls.',
 '- **Somatic thresholding:** after branch selection, retain either 17 of 24 groups (drop about 29%) or 12 of 24 (drop 50%). Suppressed groups output zero. The primary ranking is the selected branch\'s recent training usefulness, an EMA of negative loss-gradient times response. This score is global recent-task state, not usefulness measured on the current image.',
 '- **Cue-ranked suppression:** a separate secondary variant ranks groups by their current image-to-memory similarity. This is retrieval confidence, not supervised usefulness. It is kept separate from the proposed trace-ranked rule.',
 '- **Class feedback, called bAP consensus in the proposal:** Pass 1 produces ten logits. Its top two predicted classes each propose a route. For candidate class c, every active group selects the branch maximizing current response × W[c, group, branch]. Groups suppressed in Pass 1 remain suppressed. Each candidate route produces ten new logits. The route with the larger candidate-class log probability wins, and its unrestricted ten-class argmax is the final prediction. There is no true-label feedback, route bank, task ID, stored per-class configuration, or test-time learning. This is a mathematical two-pass feedback rule, not a biophysical bAP simulation.',
 '- **Patch-restricted group:** all four branches within a group read the same small image region, with different learned filters. Twenty-four overlapping 5×5 patches cover a central 20×25 rectangle. Each branch stores only 25 input weights. A 6×6 shared-output control is also included.',
 '- **Local retrieval memory:** before each neural update, associate each group\'s current winning branch with the current patch summary. The summary is normalized 3×3 average pooling plus a constant bias coordinate (10 numbers). Store per-branch sums and counts, not images. A group\'s winner is recorded before somatic suppression, even if its output will later be zeroed. No class label enters these association writes.',
 '- **Direct classifier:** ridge regression on a fixed 64-dimensional whole-image summary, accumulating statistics from current digit labels. This baseline needs no branch network or configuration selector.',
 '', '## Main compact-model result',
 '', 'The following table uses the 5×5 independent-output model trained with all 24 groups active. Suppression is then a fixed-weight inference intervention. The fraction dropped is chosen on validation separately for each reported family, never on test outcomes. Both Adam primary trace-based families chose 17 retained groups in development.',
 '', '![Fixes](fixes_accuracy.png)',
 '', '| Inference rule | Adam accuracy | Adam old eight digits | Adam newest pair | SGD accuracy |', '|---|---:|---:|---:|---:|']
names={'baseline':'All groups; no feedback','feedback':'All groups; class feedback','somatic_trace':'Suppress weak groups using usefulness trace','somatic_trace_feedback':'Trace suppression plus class feedback','somatic_cue':'Suppress groups using image-cue similarity','somatic_cue_feedback':'Cue suppression plus class feedback'}
for family,name in names.items():
    d=summary['adam'][arm]['methods'][family];ss=summary['sgd'][arm]['methods'][family]
    lines.append(f"| {name} | {d['accuracy']['mean']:.2f}% | {d['old_accuracy']['mean']:.2f}% | {d['new_accuracy']['mean']:.2f}% | {ss['accuracy']['mean']:.2f}% |")
lines += ['', f'The direct classifier reaches **{direct:.2f}%**. The compact branch pipeline remains substantially below this simpler control.',
 '', '![Old and new](old_new_accuracy.png)',
 '', '## Paired effects of each fix',
 '', 'Differences are percentage points versus all groups/no feedback, on identical final weights and local associative state. Intervals are 95% paired-seed t intervals with nine degrees of freedom; both orders are averaged within each seed. They are exploratory and not corrected for multiple comparisons.',
 '', '| Optimizer | Feedback alone | Trace suppression alone | Trace suppression + feedback | Cue suppression alone |', '|---|---:|---:|---:|---:|']
for opt,c in comparisons.items():
    lines.append('| '+opt+' | '+' | '.join(fmt(c[name]) for name in ['feedback_minus_baseline','trace_suppression_minus_baseline','trace_suppression_and_feedback_minus_baseline','cue_suppression_minus_baseline'])+' |')
lines += ['', '## Both requested suppression fractions',
 '', 'All predetermined fractions are shown below on the same Adam compact backbone trained with 24 groups. No fraction is selected from these test scores.',
 '', '| Groups retained | Ranking | No feedback | Class feedback |', '|---|---|---:|---:|']
for keep in [24,17,12]:
    for rank in (['trace'] if keep==24 else ['trace','cue']):
        rr=[r for r in adam if r['arm']=='patch5_independent_keep24']
        off=100*st.mean(raw_setting(r,keep,False,rank)['accuracy'] for r in rr)
        on=100*st.mean(raw_setting(r,keep,True,rank)['accuracy'] for r in rr)
        lines.append(f'| {keep} | {rank} | {off:.2f}% | {on:.2f}% |')
lines += ['', '## Did suppression need to be present during training?',
 '', 'These rows train with 24, 17, or 12 active groups and evaluate with the same retained count and trace ranking. This avoids interpreting a failed posthoc intervention as a failure of every suppression-trained model.',
 '', '| Adam training groups retained | Same-count inference, no feedback | Same-count inference, feedback | Current-pair acquisition | Historical recovery |', '|---|---:|---:|---:|---:|']
for keep in [24,17,12]:
    a=f'patch5_independent_keep{keep}';rr=[r for r in adam if r['arm']==a]
    off=100*st.mean(raw_setting(r,keep)['accuracy'] for r in rr)
    on=100*st.mean(raw_setting(r,keep,True)['accuracy'] for r in rr)
    d=summary['adam'][a]
    lines.append(f"| {keep} | {off:.2f}% | {on:.2f}% | {d['acquisition']:.2f}% | {d['historical']:.2f}% |")
lines += ['', 'Historical recovery uses the true digit pair to restore the actual branch-and-group mask saved after learning that pair, at final fixed weights. It is a task-assisted diagnostic and never enters automatic inference.',
 '', '## Every neural architecture and memory budget',
 '', 'Baseline and feedback columns retain all groups at inference; training retention is named explicitly. The shared-head controls cannot assign different output connections to branches within a group. Their feedback can still distinguish branches through different responses multiplied by a shared class weight.',
 '', '| Adam architecture | Baseline | Class feedback | Acquisition | Historical recovery | Model + local memory bytes |', '|---|---:|---:|---:|---:|---:|']
for a,d in summary['adam'].items():
    lines.append(f"| {a.replace('_',' ')} | {d['methods']['baseline']['accuracy']['mean']:.2f}% | {d['methods']['feedback']['accuracy']['mean']:.2f}% | {d['acquisition']:.2f}% | {d['historical']:.2f}% | {d['readout_numeric_bytes']['baseline']:,} |")
lines += ['', '**Optimizer state changes the memory conclusion.** The compact model uses fewer bytes than the direct classifier only when comparing model-plus-retrieval state, or when using SGD without momentum. Adam retains additional moment arrays needed to continue its training trajectory.',
 '', '| Compact system | Model + retrieval | Optimizer history | Total persistent numeric state |', '|---|---:|---:|---:|',
 '| 5×5 independent outputs, Adam | 18,520 B | 27,744 B | **46,264 B** |',
 '| 5×5 independent outputs, SGD without momentum | 18,520 B | 0 B | **18,520 B** |',
 '| Direct 64-feature ridge classifier | 18,944 B | 0 B | **18,944 B** |',
 '', 'Counts include actual input/output weights and biases, usefulness traces, local cue sums/counts, and patch coordinates. Gradients, temporary indexing/feedback routes, solve workspace, and Python metadata are excluded for all methods. No cached derived coefficients are stored. A deployment-only frozen direct classifier could discard its continual-learning statistics and keep much less state; the 18,944-byte comparison retains the statistics needed for further online updates.',
 '', 'The SGD supplement uses the same cohort and training stream, selecting one common rate from 0.03/0.1/0.3 on development validation before its own confirmation. It was added specifically to resolve the optimizer-state budget issue; it is not independent replication of the Adam experiment.',
 '', '## What does class feedback correct or destroy?',
 '', 'These changes are measured on identical test images and frozen weights. Labels enter only after both predictions are complete. Values are percentages of all test examples, not conditional rates.',
 '', '| Adam compact configuration | Wrong → correct | Correct → wrong | Net gain | True class in first-pass top two |', '|---|---:|---:|---:|---:|']
for family,d in transition.items():
    lines.append(f"| {family.replace('_',' ')} | {d['wrong_to_correct']:.2f}% | {d['correct_to_wrong']:.2f}% | {d['net_change']:+.2f} pp | {d['first_top2_true_class_coverage']:.2f}% |")
lines += ['', 'First-pass top-two coverage diagnoses the quality of the proposed class hypotheses. It is not a formal accuracy ceiling because all ten outputs remain eligible after feedback. The audit also records predictions outside those first two classes and asserts exact reproduction of the primary feedback predictions.',
 '', '## Larger-cue and joint-capacity controls',
 '', 'The same trained models are also tested with per-branch 64-dimensional whole-image prototype cues. This is a larger-memory control, not the compact pipeline: for the independent-output 5×5 model it requires 39,256 model-plus-memory bytes before optimizer state.',
 '', '| Model trained with 24 groups | Local cue baseline | Whole-image 64-feature cue baseline | Whole-image cue + feedback |', '|---|---:|---:|---:|']
for a in ['full_independent_keep24','patch5_independent_keep24']:
    d=summary['adam'][a]['methods']
    lines.append(f"| {a.replace('_',' ')} | {d['baseline']['accuracy']['mean']:.2f}% | {d['global64_baseline']['accuracy']['mean']:.2f}% | {d['global64_feedback']['accuracy']['mean']:.2f}% |")
lines += ['', f"Joint training, with all classes mixed and the same total example presentations, reaches **{jointmeans['full_independent_keep24']:.2f}%** for the full-image network and **{jointmeans['patch5_independent_keep24']:.2f}%** for the compact 5×5 network. Each joint arm independently chooses its learning rate on development validation. These are acquisition/capacity references, not continual-learning results.",
 '', '## Protocol and limitations',
 '', '- Real MNIST, pixels/255, five class pairs, two passes per pair, batch 128, gradient clipping 5, continuous optimizer, and usefulness EMA decay 0.95. No replay, pretraining, task-ID input, boundary reset, or historical route bank in automatic inference.',
 '', '- Development initialization 700 uses a stratified 50,000/10,000 fitting/validation split. Eight Adam backbones each test rates 0.001/0.003. One common rate is selected by mean automatic local-memory accuracy evaluated at each training retention count. Inference retention 17/12 is selected separately within each trace/cue/feedback family on validation. All predetermined test settings are retained, not just chosen readouts. Joint references have four separate development runs. The SGD supplement has nine development runs.',
 '', f"- Frozen Adam learning rate: {select['adam']['lr']}; frozen SGD learning rate: {select['sgd']['lr']}; direct classifier ridge: {select['adam']['direct_ridge']}. Confirmation uses initializations 701–710, both class orders, full 60,000 training and 10,000 test images. All Adam/SGD paired comparisons share initialization and sample order where shapes match. Joint references use ten seeds and have no class order.",
 '', '- Class feedback is an inference-only intervention; it is not trained end-to-end through the revised routes. The rule can increase a candidate class\'s own score without making it correct. There is no learned confidence guard or fallback to the first-pass route. Conclusions apply to this explicit literal implementation, not every feedback rule.',
 '', '- Trace-ranked suppression uses recent training usefulness, which can favor the newest pair. Worse old accuracy alongside better newest-pair accuracy is consistent with this bias; the present ablations do not fully identify the causal source. Cue similarity is a different signal and is reported separately.',
 '', '- The compact patch and full-image models differ in effective input capacity and local cue access. A patch-versus-full comparison is a complete-design comparison, not an isolated spatial-mask intervention. The lower joint accuracy of the compact model must remain visible.',
 '', '- All evaluation calls checksum model and memory before and after. The 640-case feedback audit reproduces the exact primary predictions and separates repaired errors from newly introduced errors. Shared-output expansion, exact active counts, local memory writes, class-contribution maximization, and numerical-state budgets are verified.',
 '', '- These are new initializations of the same repeatedly studied MNIST dataset, not a new dataset. Ten paired seeds reduce initialization uncertainty but do not establish broad generalization. Secondary intervals are not corrected for multiple comparisons.',
 '', '## Reproducibility',
 '', '- [Main driver](../../bdh_cl/scripts/somatic_consensus.py), [SGD budget control](../../bdh_cl/scripts/somatic_consensus_sgd.py), [feedback audit](../../bdh_cl/scripts/audit_somatic_consensus.py).',
 '- [Primary protocol](../../results/somatic_consensus_20261021/protocol.json), [primary frozen selection](../../results/somatic_consensus_20261021/selection.json), [SGD protocol](../../results/somatic_consensus_20261021/sgd_protocol.json), [SGD selection](../../results/somatic_consensus_20261021/sgd_selection.json).',
 '- [Adam results](../../results/somatic_consensus_20261021/results.json), [SGD results](../../results/somatic_consensus_20261021/sgd_results.json), [joint references](../../results/somatic_consensus_20261021/joint_results.json), [summary and paired intervals](../../results/somatic_consensus_20261021/summary.json).',
 '- [Feedback transitions](../../results/somatic_consensus_20261021/feedback_transition_audit.json), [state budget audit](../../results/somatic_consensus_20261021/state_budget_audit.json), [verification](../../results/somatic_consensus_20261021/verification.json).',
 '- Complete checkpoints, every validation candidate, primary and supplemental completion manifests, logs, and archived source are retained in the results directory.',
 '']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print('Wrote report, plots and summary.')
