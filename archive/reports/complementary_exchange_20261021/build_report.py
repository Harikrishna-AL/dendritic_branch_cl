"""Summarize prespecified factorial confirmation; no test-selected settings."""
import json
import math
import os
import statistics as st
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/bdh_complementary_plot_cache')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/bdh_complementary_xdg_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/complementary_exchange_20261021'
HERE=Path(__file__).resolve().parent
rows=json.loads((DATA/'results.json').read_text())
assert len(rows)==240
selection=json.loads((DATA/'selection.json').read_text())
STRUCTURES=['grouped_shared','global_shared','global_independent']
LABELS={'grouped_shared':'Branch groups; shared output weights',
        'global_shared':'Ordinary global selection; same shared output weights',
        'global_independent':'Ordinary global selection; independent output weights'}
SEEDS=list(range(601,611))
CRITICAL=2.262157163


def metric(row,kind='automatic',field='accuracy'):
    if kind in ['historical','latest']:
        return row['diagnostics'][kind][field]
    if kind=='acquisition':return row['acquisition']
    method={'automatic':'ridge','local':'local_field','direct':'direct'}[kind]
    return next(m[field] for m in row['methods'] if m['method']==method)


def values(arm,kind='automatic',field='accuracy'):
    return {seed:100*st.mean(metric(r,kind,field) for r in rows if r['arm']==arm and r['seed']==seed) for seed in SEEDS}


def describe(d):
    data=list(d.values());mu=st.mean(data);sd=st.stdev(data);half=CRITICAL*sd/math.sqrt(len(data))
    return {'mean':mu,'seed_sd':sd,'low':mu-half,'high':mu+half,'seed_values':d}


def subtract(a,b):return {seed:a[seed]-b[seed] for seed in SEEDS}


cells={}
interactions={}
contrasts={}
for structure in STRUCTURES:
    for view in ['complementary','redundant']:
        for comm in ['self','neighbor']:
            arm=f'{structure}__{view}__{comm}'
            cells[arm]={
                'automatic':describe(values(arm)),
                'old':describe(values(arm,field='old_accuracy')),
                'new':describe(values(arm,field='new_accuracy')),
                'historical':describe(values(arm,'historical')),
                'historical_old':describe(values(arm,'historical','old_accuracy')),
                'acquisition':describe(values(arm,'acquisition')),
                'local':describe(values(arm,'local')),
                'latest':describe(values(arm,'latest')),
                'direct':describe(values(arm,'direct'))}
    gains={}
    for view in ['complementary','redundant']:
        gains[view]=subtract(values(f'{structure}__{view}__neighbor'),values(f'{structure}__{view}__self'))
    interactions[structure]={'complementary_gain':describe(gains['complementary']),
                             'redundant_gain':describe(gains['redundant']),
                             'interaction':describe(subtract(gains['complementary'],gains['redundant']))}
for view in ['complementary','redundant']:
    for comm in ['self','neighbor']:
        for control in ['global_shared','global_independent']:
            key=f'branch_minus_{control}__{view}__{comm}'
            contrasts[key]=describe(subtract(values(f'grouped_shared__{view}__{comm}'),values(f'{control}__{view}__{comm}')))
for control in ['global_shared','global_independent']:
    a=interactions['grouped_shared']['interaction']['seed_values']
    b=interactions[control]['interaction']['seed_values']
    contrasts['branch_interaction_minus_'+control]=describe(subtract(a,b))

intervention=[]
if (DATA/'inference_intervention.json').exists():
    intervention=json.loads((DATA/'inference_intervention.json').read_text())
assert len(intervention)==120
inference={}
for structure in STRUCTURES:
    for view in ['complementary','redundant']:
        arm=f'{structure}__{view}__neighbor'
        dd={seed:100*st.mean(r['learned_neighbor_automatic']-r['same_weights_self_only_automatic']
                            for r in intervention if r['arm']==arm and r['seed']==seed) for seed in SEEDS}
        inference[arm]=describe(dd)
summary={'cells':cells,'communication_effects':interactions,'architecture_contrasts':contrasts,
         'frozen_neighbor_dependence':inference,'units':'percentage points / accuracy percentages',
         'uncertainty':'Order-averaged paired seed t intervals, df9; unadjusted for multiple comparisons.'}
(DATA/'summary.json').write_text(json.dumps(summary,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15.5,5.1),sharey=True)
titles=['Branch groups','Ordinary selection\nshared output weights','Ordinary selection\nindependent output weights']
for ax,structure,title in zip(axes,STRUCTURES,titles):
    for i,view in enumerate(['complementary','redundant']):
        for offset,comm,color,label in [(-.18,'self','#8695a5','Self signal only'),(.18,'neighbor','#287f8f','Neighbor signals')]:
            d=cells[f'{structure}__{view}__{comm}']['automatic']
            ax.bar(i+offset,d['mean'],.36,color=color,label=label if i==0 else None)
            for seed in SEEDS:
                jitter=(seed-605.5)*.009
                ax.plot(i+offset+jitter,d['seed_values'][seed],'.',color='#1c2732',ms=3,alpha=.65)
    direct=cells[f'{structure}__complementary__self']['direct']['mean']
    ax.axhline(direct,color='#ad5d30',ls='--',label='Direct image classifier')
    ax.set_xticks([0,1],['Different quarters\nwithin each group','Same quarter\nwithin each group'])
    ax.set_title(title)
    ax.set_ylim(60,90)
axes[0].set_ylabel('Automatic ten-digit test accuracy (%)')
axes[-1].legend(loc='lower right',fontsize=8)
fig.suptitle('Does complementary information make sensory communication more useful?\nTen fresh seeds; forward/reverse orders averaged within seed',fontsize=12)
fig.tight_layout();fig.savefig(HERE/'factorial_accuracy.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,4.4))
names=[]
for i,s in enumerate(STRUCTURES):
    d=interactions[s]['interaction'];ax.errorbar(d['mean'],i,xerr=[[d['mean']-d['low']],[d['high']-d['mean']]],fmt='o',color='#287f8f',capsize=4)
    names.append({'grouped_shared':'Branch groups','global_shared':'Ordinary, shared outputs','global_independent':'Ordinary, independent outputs'}[s])
ax.axvline(0,color='#666',ls='--');ax.set_yticks(range(3),names);ax.invert_yaxis()
ax.set_xlabel('Extra communication benefit with different views (percentage points)')
ax.set_title('(Neighbor − self) with complementary views\nminus (neighbor − self) with redundant views; 95% paired-seed intervals')
fig.tight_layout();fig.savefig(HERE/'interaction.png',dpi=180);plt.close(fig)


def fmt(d):return f"{d['mean']:+.2f} [{d['low']:+.2f}, {d['high']:+.2f}]"


lines=[
 '# Complementary sensory fields and communication: controlled confirmation',
 '', 'Completed 21 October 2026. This study tests whether communication helps more when alternatives see different image regions than when they repeatedly see the same region. It uses ten fresh initialization seeds, both digit-pair orders, common validation-selected settings, and an exactly matched ordinary-selection control.',
 '', '**Main finding: the complementary-information hypothesis is not confirmed for this one-step exchange rule.** Branch automatic accuracy is 78.75% with self signals and 78.79% with neighbor signals under complementary views. Its extra communication benefit over redundant views is +0.22 percentage points (95% interval -1.64 to +2.07). Both ordinary-selection controls also have interaction intervals spanning zero. This does not establish exact equivalence or rule out every practically relevant effect.',
 '', 'The exactly matched ordinary-selection control reaches 80.23% without exchange and 79.40% with exchange under complementary views. Independent-output ordinary selection reaches 81.30% and 81.34%. There is no demonstrated automatic branch advantage. Old-digit branch accuracy changes from 76.63% to 76.25% with exchange, while newest-digit accuracy improves; the small overall change is not evidence of improved old-skill retention.',
 '', 'At fixed final weights, replacing neighbor signals by self signals changes complementary branch accuracy by only about 0.05 percentage points. The trained models have very little practical dependence on this exchanged evidence. The direct classifier remains stronger at 85.53%.',
 '', '## What was controlled',
 '', '- All models have 96 learned sensory responses and select exactly 24 of them per prediction. Each response reads exactly 196 pixels using 196 actual weights; no masked-out unused input parameters are allocated.',
 '- In a **complementary group**, the four responses see different image quarters. In a **redundant group**, all four see the same quarter, but have different learned filters. Across the full network, every quarter supplies exactly 24 responses in both conditions. Thus global coverage and input-weight budgets match.',
 '- **Neighbor exchange** lets a response mix signals from two other responses in its four-response group. **Self-only** uses its own signal in the same computation. Both have the same 96 trainable message gains and computation depth. Self-only means no exchange between responses, not absence of the extra transform.',
 '- **Branch grouping** selects one response within every group and shares output connections within that group. **Ordinary global selection with shared outputs** selects any 24 of 96 responses, retaining precisely the same output-weight sharing and parameter count. This isolates the restriction on selection. **Ordinary independent outputs** also selects globally, but every response has separate output connections; it has 720 additional parameters.',
 '- Shared-output networks have 19,258 learned parameters; independent-output networks have 19,978. All four view/communication conditions within each structure share the same initialization and parameter count.',
 '', '## How automatic selection works',
 '', 'A fixed 64-dimensional summary of the current image is associated with the currently active response mask during training. An online ridge-regression learner stores covariance and cross-statistics, without storing old examples. At evaluation it predicts response scores from the image; the network selects its 24 active responses from those scores. No task identity or historical mask archive enters automatic predictions.',
 '', 'Labels supervise the neural prediction loss and a separate direct-classifier control. They do not supervise the image-to-mask association memory. Every association is written from the pre-update mask while the corresponding image is current.',
 '', '**Historical recovery** instead uses the correct digit pair to select its phase-end mask. It is a diagnostic with task assistance, not automatic continual-learning accuracy. Final weights and classifier remain unchanged; all ten digit outputs remain eligible.',
 '', '## Main automatic accuracy comparison',
 '', 'Percentages average both orders within each seed, then the ten seed means. Dots in the figure show those ten paired seed means. The plotted y-axis begins at 60% to expose differences.',
 '', '![Automatic accuracy](factorial_accuracy.png)',
 '', '| Selection structure | Different views: self | Different views: exchange | Same view: self | Same view: exchange |',
 '|---|---:|---:|---:|---:|']
for s in STRUCTURES:
    vv=[cells[f'{s}__{v}__{c}']['automatic']['mean'] for v in ['complementary','redundant'] for c in ['self','neighbor']]
    lines.append('| '+LABELS[s]+' | '+' | '.join(f'{v:.2f}%' for v in vv)+' |')
lines += ['', '## The main causal comparison',
 '', 'The interaction below asks whether communication gains more from different sensory views than from repeated views. Specifically: (exchange minus self-only with complementary views) minus (exchange minus self-only with redundant views). A positive value favors the complementary-information hypothesis.',
 '', '| Structure | Exchange effect, different views | Exchange effect, same view | Extra benefit from different views |',
 '|---|---:|---:|---:|']
for s,d in interactions.items():
    lines.append('| '+LABELS[s]+' | '+fmt(d['complementary_gain'])+' | '+fmt(d['redundant_gain'])+' | '+fmt(d['interaction'])+' |')
lines += ['', 'Values are percentage-point differences with 95% paired-seed t intervals in brackets, using ten independent initialization seeds (nine degrees of freedom). Orders are paired, not independent seeds. The intervals are not adjusted for multiple comparisons.',
 '', '![Interaction](interaction.png)',
 '', '## Retention and acquisition',
 '', '**Old digits** means the eight digits from the earlier four pairs. **Newest digits** means the final pair. **Acquisition** measures current-pair accuracy immediately after learning each pair. These separate preserving old ability from merely learning the current task well.',
 '', '| Structure | Views | Signals | Automatic | Old digits | Newest digits | Historical recovery | Acquisition |',
 '|---|---|---|---:|---:|---:|---:|---:|']
for s in STRUCTURES:
    for v in ['complementary','redundant']:
        for c in ['self','neighbor']:
            d=cells[f'{s}__{v}__{c}']
            vv=[d[k]['mean'] for k in ['automatic','old','new','historical','acquisition']]
            lines.append('| '+LABELS[s]+' | '+v+' | '+c+' | '+' | '.join(f'{value:.2f}%' for value in vv)+' |')
direct=cells['grouped_shared__complementary__self']['direct']['mean']
lines += ['', f'The direct classifier, trained from the same image summaries and current digit labels, achieves **{direct:.2f}%** without a neural backbone. Its targets differ from mask prediction, so this is a complete-method baseline. Features are fixed; repeated seed values are not independent learned representations.',
 '', '## Is there an advantage specifically from grouping?',
 '', 'These differences subtract the ordinary-selection control from the grouped branch model under the same view and communication condition. The shared-output comparison is exactly parameter matched and differs only in the restriction on selecting responses. The independent-output comparison additionally changes classifier sharing and has 720 more control parameters.',
 '', '| Comparison | Automatic accuracy difference and 95% interval |', '|---|---:|']
for k,d in contrasts.items():lines.append('| '+k.replace('__',' / ').replace('_',' ')+' | '+fmt(d)+' |')
lines += ['', '## Frozen inference intervention',
 '', 'For each exchange-trained network, neighbor signals were replaced by self signals at the final fixed weights, with unchanged associative memory and therefore unchanged automatic masks. This measures dependence on exchanged evidence during prediction. It is distinct from comparing separately trained exchange and self-only networks, because an intervention can disrupt a learned computation.',
 '', '| Exchange-trained model | Original exchange minus self-only intervention |', '|---|---:|']
for a,d in inference.items():lines.append('| '+a.replace('__',' / ').replace('_',' ')+' | '+fmt(d)+' |')
lines += ['', '## Selection, state, and verification',
 '', f"- Development initialization 600 uses a stratified 50,000/10,000 split of training images. All 12 cells are trained at Adam rates 0.001 and 0.003. One common rate and one common ridge strength are selected by mean automatic validation accuracy across all cells, avoiding separately tuned treatment settings. Selected rate: {selection['chosen']['lr']}; selected mask-predictor ridge: {selection['chosen']['ridge']}; direct-classifier ridge: {selection['direct_chosen']['ridge']}.",
 '- The selection manifest and protocol hash were saved before any confirmation run. Confirmation uses seeds 601–610, forward and reverse pair order, all 60,000 training images and the official 10,000 test images: 240 neural confirmation runs, plus 24 development runs. No confirmation retuning, checkpoint selection, or optional stopping.',
 '- Two epochs per pair, batch 128, one continuous Adam optimizer, gradient clipping 5, usefulness EMA decay 0.95. No replay, pretraining, task-ID input, optimizer resets, trace resets, or explicit protection penalties.',
 '- Sensory exchange is one step: e = tanh(z), response = 0.75e + 0.25tanh(z + gain × incoming). Neighbor incoming averages two responses within the group; self incoming is e. The four complementary quarter views use geometrical adjacency around top-left, top-right, bottom-right, bottom-left. Gradients can pass through unselected senders, so the rule changes both integration and selective preservation.',
 '- Each deployed mask-regression memory needs 40,960 numeric bytes: a 64×64 covariance and a 96×64 cross-statistic matrix. The direct classifier needs 18,944 bytes. A field-only sum-based reader is a separate secondary method needing 24,576 bytes. All candidates coexist in experiment checkpoints for paired analysis; their union is not a deployment budget.',
 '- Shared-output model state occupies 81,256 numeric bytes, including parameter and routing/index buffers; adding mask-regression memory gives 122,216 bytes. Independent-output model state adds 2,880 bytes. Optimizer state, Python metadata, temporary tensors, and historical diagnostic banks are excluded. Actual input matrices contain only the allowed 196 contacts per response.',
 '- Verification checks globally balanced fields, the intended within-group overlap, exact compact projection equivalence, invariance to out-of-field pixels before exchange, equal active counts, exact shared-output parameter matching, zero-gain exchange equivalence, finite gradients, usefulness finite differences, pre-update association targets, ridge agreement with a batch reference, and memory-write isolation.',
 '- Every official evaluation verifies unchanged model and association-memory checksums. The frozen-inference audit covers all 120 exchange-trained models; it changes no learned weight, input association, or automatic selection. Historical masks enter only task-assisted diagnostics.',
 '', '## Interpretation limits',
 '', 'This is a confirmation over new initializations of the same MNIST dataset, not a new dataset or evidence about arbitrary tasks. Repeated same-quarter views still contain distinct learned filters; they are spatially redundant, not identical response values. Both conditions cover the full image globally, but their within-group access differs by design. Global image keys give the selector whole-image information even when sensory responses have partial fields; a field-only reader is reported separately in raw records.',
 '', 'This is a controlled confirmation of the hypothesis, not an exact rerun of the earlier three-seed pilot: the earlier model used 25 groups and masked 784-input matrices, whereas this one uses 24 balanced groups, actual 196-input matrices with their native initialization, and one common validation-selected optimizer/reader setting. These changes make the new factorial comparisons cleaner but limit direct attribution of the old-versus-new result to seed count alone.',
 '', 'A positive interaction supports a benefit from complementary evidence under this update and selection rule. It does not by itself show a branch-specific advantage: that requires outperforming the exactly matched ordinary-selection control. Negative or uncertain effects apply to this one-step exchange rule and finite validation search, not all possible communication mechanisms. A direct-classifier advantage must remain visible when assessing practical value.',
 '', '## Reproducibility',
 '', '- [Training and evaluation driver](../../bdh_cl/scripts/complementary_exchange.py), [frozen inference audit](../../bdh_cl/scripts/audit_complementary_exchange.py), [protocol](../../results/complementary_exchange_20261021/protocol.json).',
 '- [All development candidates](../../results/complementary_exchange_20261021/development.json), [frozen selection](../../results/complementary_exchange_20261021/selection.json), [240 confirmation records](../../results/complementary_exchange_20261021/results.json).',
 '- [Summary and paired intervals](../../results/complementary_exchange_20261021/summary.json), [120 inference interventions](../../results/complementary_exchange_20261021/inference_intervention.json), [verification](../../results/complementary_exchange_20261021/verification.json), [completion](../../results/complementary_exchange_20261021/completion.json).',
 '- Final model/memory checkpoints, phase-end diagnostic masks, run logs, and archived source are in the results directory. Primary and secondary analysis are retained for every cell.',
 '']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print('Saved report, two figures, and complete summary.')
