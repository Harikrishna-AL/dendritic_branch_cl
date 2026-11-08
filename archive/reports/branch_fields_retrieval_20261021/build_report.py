"""Summarize fixed confirmation records; does not select any settings."""
import json
import math
import statistics as st
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/branch_fields_retrieval_20261021'
HERE=Path(__file__).resolve().parent
rows=json.loads((DATA/'results.json').read_text())+json.loads((DATA/'flat50_results.json').read_text())
exchange=json.loads((DATA/'feature_exchange_results.json').read_text())
audit=json.loads((DATA/'communication_scramble_audit.json').read_text())
labels={
 'full_flat':'Whole image; ordinary neurons, 25 active',
 'full_flat50':'Whole image; ordinary neurons, 50 active',
 'full_group_independent':'Whole image; grouped, separate output weights',
 'full_branch1':'Whole image; one branch per group',
 'full_branch2':'Whole image; two branches per group',
 'quadrant_flat':'Image quarters; ordinary neurons, 25 active',
 'quadrant_flat50':'Image quarters; ordinary neurons, 50 active',
 'quadrant_group_independent':'Image quarters; grouped, separate output weights',
 'quadrant_branch1':'Image quarters; one branch per group',
 'quadrant_branch2':'Image quarters; two branches per group',
 'random_branch1':'Random pixel partitions; one branch per group'}

def get(r,metric):
    if metric in ['historical','latest']:return r['diagnostics'][metric]['accuracy']
    if metric=='acquisition':return r['acquisition']
    return next(m['accuracy'] for m in r['methods'] if m['method']==metric)

def avg(arm,metric,source=rows):return 100*st.mean(get(r,metric) for r in source if r['arm']==arm)

def paired(a,b,metric,source=rows):
    values=[]
    for seed in [561,562,563]:
        aa=st.mean(get(r,metric) for r in source if r['arm']==a and r['seed']==seed)
        bb=st.mean(get(r,metric) for r in source if r['arm']==b and r['seed']==seed)
        values.append(100*(aa-bb))
    mu=st.mean(values);half=4.30265273*st.stdev(values)/math.sqrt(3)
    return {'difference_pp':mu,'low':mu-half,'high':mu+half,'paired_seed_differences':values}

summary={}
for a in labels:
    summary[a]={metric:avg(a,metric) for metric in ['historical','ridge','latest','acquisition']}
comparisons={
 'grouping_without_output_sharing':paired('full_group_independent','full_flat','ridge'),
 'output_sharing_within_groups':paired('full_branch1','full_group_independent','ridge'),
 'two_versus_one_full_branches':paired('full_branch2','full_branch1','ridge'),
 'two_branches_versus_fifty_ordinary':paired('full_branch2','full_flat50','ridge'),
 'quarter_versus_full_branch1':paired('quadrant_branch1','full_branch1','ridge'),
 'quarter_versus_random_partition':paired('quadrant_branch1','random_branch1','ridge')}
for a in ['full_branch1','quadrant_branch1','quadrant_branch2']:
    comparisons[a+'_sensory_neighbor_minus_self']=paired(a+'_neighbor',a+'_self','ridge',exchange)
(DATA/'summary.json').write_text(json.dumps({'models':summary,'paired_comparisons':comparisons},indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(17,5.4))
groups=[('full_flat','Ordinary\n25 active'),('full_group_independent','Grouped\nseparate outputs'),('full_branch1','One branch\nshared outputs'),('full_branch2','Two branches\nshared outputs')]
for i,(a,l) in enumerate(groups):
    axes[0].bar(i-.17,avg(a,'historical'),.34,color='#9aa6b2',label='Correct historical selection' if i==0 else None)
    axes[0].bar(i+.17,avg(a,'ridge'),.34,color='#287d8e',label='Image-driven selection' if i==0 else None)
axes[0].axhline(avg('full_flat','direct'),color='#a84d20',ls='--',label='Direct image classifier')
axes[0].set_xticks(range(4),[l for a,l in groups]);axes[0].set_ylim(0,105);axes[0].set_title('Does grouping improve retrieval?');axes[0].set_ylabel('Ten-digit test accuracy (%)');axes[0].legend(fontsize=8,loc='lower left')
groups=[('full_branch1','Whole\nimage'),('quadrant_branch1','Image\nquarters'),('random_branch1','Random\npartitions')]
for i,(a,l) in enumerate(groups):
    axes[1].bar(i-.17,avg(a,'historical'),.34,color='#9aa6b2')
    axes[1].bar(i+.17,avg(a,'ridge'),.34,color='#287d8e')
axes[1].set_xticks(range(3),[l for a,l in groups]);axes[1].set_ylim(0,105);axes[1].set_title('Restricted receptive fields')
groups=[('full_branch1','Whole image'),('quadrant_branch1','Image quarters'),('quadrant_group_independent','Quarters;\nseparate outputs')]
for i,(a,l) in enumerate(groups):
    rr=[r for r in audit if r['arm']==a]
    for j,(key,color) in enumerate([('off','#9aa6b2'),('learned','#287d8e'),('scrambled_mean','#c59861')]):
        axes[2].bar(i+(j-1)*.24,100*st.mean(r[key] for r in rr),.24,color=color,label={'off':'No messages','learned':'Learned messages','scrambled_mean':'Scrambled messages'}[key] if i==0 else None)
axes[2].set_xticks(range(3),[l for a,l in groups]);axes[2].set_ylim(0,80);axes[2].set_title('Messages using restricted image cues');axes[2].legend(fontsize=8,loc='lower left')
fig.suptitle('Split MNIST: three seeds, both class orders; fixed final weights at evaluation',fontsize=13)
fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=180);plt.close(fig)

lines=['# Branch grouping, image regions, and communication',
 '', 'Completed 21 October 2026. **No automatic retrieval advantage for branches over ordinary selected neurons emerged. Restricted receptive fields preserve substantial skills with fewer effective input weights. Learned compatibility messages help when retrieval sees only partial images, but whole-image cues remain much stronger.**',
 '', 'The primary study has54 confirmation runs. A 50-active-neuron control adds12 and a direct sensory-exchange follow-up adds36. All102 confirmation runs reuse the same three initialization seeds and two paired class orders; they are not102 independent replications. There are34 development runs. Every automatic method uses images only at evaluation.',
 '', '![Comparison](comparison.png)',
 '', '## Definitions',
 '', '- **Ordinary network:** a fully connected multilayer perceptron (MLP), here784 inputs ->100 tanh hidden responses ->10 digit scores. Recent usefulness selects25 or50 responses globally.',
 '- **Grouped, separate outputs:** the same100 responses are arranged in25 groups of four. One response wins per group, but each has its own output connections. This isolates grouping from shared output weights relative to the ordinary25-active model.',
 '- **Branch network:** one or two responses participate within each group, and branches in a group share output connections. Two responses are summed and divided by sqrt(2). There are25 groups and four branches per group.',
 '- **Usefulness trace:** an exponentially weighted recent estimate of the benefit of a response, computed as negative loss-gradient times response, EMA decay.95. It controls training selection and is supervised by digit labels through the prediction loss.',
 '- **Historical selection:** the correct digit pair selects a phase-end neuron/branch mask. This diagnostic changes no final learned weight or classifier connection; all ten outputs remain eligible. It requires task information.',
 '- **Image-driven selection:** an online ridge-regression association learns to predict the current mask from a64-dimensional fixed pooled image key. At evaluation the predicted scores select neurons/branches; no saved mask bank, task ID, or label is available.',
 '- **Direct classifier:** the same ridge-regression learner predicts digits from the same image keys directly, requiring no neural backbone. Its supervised targets differ from the mask targets, so it is a complete-method baseline rather than a pure architecture intervention.',
 '- **Receptive field:** the pixels a response can directly read. Whole-image responses see784 pixels. Quarter-image responses see one196-pixel quadrant. Random-partition responses see196 scattered pixels; the four fields partition the image.',
 '- **Local cue memory:** each response stores the sum of input keys observed while it was active. Similarity to that stored sum supplies a retrieval score. Whole-image cues use all pixels; field cues use only the response\'s allowed pixels.',
 '- **Compatibility messages:** counts of which alternatives were active together in neighboring groups adjust their retrieval scores for three rounds. These are selection messages, not direct sensory activation exchange.',
 '- **Sensory exchange:** a separate follow-up lets branches directly mix neighboring partial-image responses before selection. Its equal-parameter self-only control performs the same extra computation using the branch\'s own response.',
 '', '## Primary accuracy results',
 '', 'All values are mean percentages across three seeds and both orders. Acquisition means accuracy on each pair immediately after learning it; it is not final ten-digit retention. Latest-mask accuracy remains near20% for these models.',
 '', '| Network and input access | Acquisition | Correct historical selection | Automatic image-driven selection |', '|---|---:|---:|---:|']
for a in labels:
    lines.append(f"| {labels[a]} | {avg(a,'acquisition'):.2f} | {avg(a,'historical'):.2f} | {avg(a,'ridge'):.2f} |")
lines += ['', f"The direct image classifier reaches approximately **{avg('full_flat','direct'):.2f}%**, above every neural retrieval pipeline in this experiment. It stores18,944 numeric bytes and uses labels while each example is current. Its fixed features are not independently learned across seeds; tiny seed variation comes from accumulation order/precision.",
 '', 'Grouping alone does not improve automatic retrieval here. Sharing output weights also supplies no gain in the full-image comparison. Allowing two branches harms the best one-branch configuration, although grouped two-branch models outperform the50-active ordinary-neuron control. Thus more simultaneous activity is not automatically better for retention.',
 '', 'Quarter-image branches have19,960 effective learned parameters, compared with78,760 for whole-image branches. The dense implementation still allocates masked-out weights and computes dense matrix products; no actual speedup or compressed deployment is established. Ordinary quarter-image networks have20,710 effective parameters and perform at least as well, so parameter efficiency is not unique to dendritic grouping. The random-partition control also performs well: spatially contiguous quarters are not uniquely beneficial.',
 '', '## Do partial views make compatibility messages more useful?',
 '', 'Same trained weights and associative sums in each off/on/scrambled comparison. Retrieval here uses only each response\'s own receptive-field cue. Scrambling permutes branch labels at each group in the connection matrices, preserving their values but breaking alignment with the learned cues. Five predetermined permutations are averaged; none is selected for performance.',
 '', '| Network | Messages off | Learned messages | Scrambled messages |', '|---|---:|---:|---:|']
for a in ['full_branch1','quadrant_branch1','quadrant_group_independent','quadrant_branch2','random_branch1']:
    rr=[r for r in audit if r['arm']==a]
    values=[100*st.mean(r[k] for r in rr) for k in ['off','learned','scrambled_mean']]
    lines.append(f'| {labels[a]} | '+ ' | '.join(f'{v:.2f}' for v in values)+' |')
lines += ['', 'This supports a bounded positive result: learned compatibility can help select computations from incomplete local cues. The gain also appears with separate output weights, so it does not require the branch model\'s shared classifier connections. It remains far below whole-image-key retrieval. In the quadrant one-branch network, whole-image local cue retrieval reaches63.04%, versus31.76% for field cues plus messages, and global ridge retrieval reaches78.16%.',
 '', '## Direct sensory-response exchange',
 '', 'This follow-up tests actual information transfer between branches, separately from routing messages. One step mixes each branch\'s local response with the mean responses of its two geometrically neighboring image quarters. There are100 trainable gains. The self-only control uses its own response in place of neighboring evidence and has identical parameter count and computation depth.',
 '', '| Field and active branches | Self-only historical | Neighbor-exchange historical | Self-only automatic | Neighbor-exchange automatic |', '|---|---:|---:|---:|---:|']
for a,label in [('full_branch1','Whole image; one branch'),('quadrant_branch1','Quarters; one branch'),('quadrant_branch2','Quarters; two branches')]:
    lines.append(f"| {label} | {avg(a+'_self','historical',exchange):.2f} | {avg(a+'_neighbor','historical',exchange):.2f} | {avg(a+'_self','ridge',exchange):.2f} | {avg(a+'_neighbor','ridge',exchange):.2f} |")
lines += ['', 'For one branch per quarter-image group, sensory exchange improves automatic accuracy by1.46 percentage points relative to the self-only control. All three seed means improve, but the95% paired-seed interval is -1.54 to +4.46 points, so the magnitude and reliability need fresh confirmation. There is no corresponding gain for full-image branches or two participating branches.',
 '', 'This limited sensory-exchange rule is not a general test of all branch communication. It changes feature integration and also lets gradients reach unselected sending branches, potentially weakening selective preservation. Its settings are selected on development data; the reused confirmation cohort is not independent replication.',
 '', '## Paired comparisons and uncertainty',
 '', 'Differences below are automatic image-driven accuracy in percentage points. Both orders are averaged within each seed. Intervals use a t distribution with two degrees of freedom over three independent seed differences. They are exploratory and not corrected for the many comparisons.',
 '', '| Comparison | Difference | 95% interval |','|---|---:|---:|']
for name,v in comparisons.items():
    lines.append(f"| {name.replace('_',' ')} | {v['difference_pp']:+.2f} | {v['low']:+.2f} to {v['high']:+.2f} |")
lines += ['', '## Protocol, information access, and limits',
 '', '- Real MNIST, pixels/255, five sequential pairs, two epochs per pair, batch128, one continuous Adam optimizer, clipping5. No old-example replay, pretraining, task-ID input, phase-boundary resets, or usage protection. Boundaries schedule the benchmark and diagnostic mask snapshots only.',
 '- Development seed560 uses a stratified50k fit/10k held-out training split. Each architecture chooses rate.001/.003 by historical validation recovery, with acquisition as tiebreak. Associative readers choose their ridge strength or message gain by automatic validation accuracy. This optimizes recoverable computation first, not the end-to-end automatic training objective. Official test data selects no settings.',
 '- Fresh seeds561–563, forward/reverse orders, full60k training and10k official test. The original25-active ordinary/grouped models and shared-head models use identical hidden initialization and minibatch order for a given seed. Output head shapes differ when sharing changes. Separate learning-rate choices and different active counts prevent interpreting every comparison as a single-factor causal effect.',
 '- Every association write occurs before the current neural update and sees only current images and current selected masks. Labels supervise neural training and the direct-classifier control, but never routing associations. No historical mask bank enters automatic prediction. Evaluation asserts unchanged model and associative memory checksums.',
 '- Ridge stores64x64 covariance and100x64 mask cross-statistics:41,984 bytes, common to all architectures here. A single local cue scope stores100x64 sums:25,600 bytes; optional25x4x4 links add1,600 bytes. All-candidate experiment checkpoints contain several competing memories; their combined storage is not the deployed budget. The direct classifier stores18,944 bytes. Numeric budgets omit scalar metadata and transient workspaces.',
 '- Whole-image flat models have79,510 allocated trainable parameters; shared-head branch models78,760. Region masks reduce effective parameter counts but are implemented using full dense tensors. Field masks, trace buffers, and audit artifacts must be distinguished from learned weights. No GPU, latency, or peak-memory advantage is claimed.',
 '- Local associative writes are cumulative; adaptive forgetting and discriminative local learning were not tuned. The ring of neighboring groups is arbitrary. All branches of the same quarter index use the same quarter, rather than independently learned receptive fields. These are bounded mechanism pilots.',
 '- Checks verify field partitions, independence from pixels outside a response\'s field before sensory exchange, correct two-branch pooling, finite gradients, current-mask association targets, zero-gain sensory-exchange equivalence, causal cross-field influence, and unchanged evaluation state. Scrambling evaluates frozen primary models and is a post-primary mechanism audit.',
 '', '## Next discriminating experiments',
 '', '1. Learn receptive fields or use overlapping regions, comparing against ordinary region-restricted neurons at matched effective contacts and active computation.',
 '2. Improve local cue learning before adding more message rounds. Whole-image retrieval and the direct classifier remain much stronger controls.',
 '3. If pursuing sensory exchange, separate inference-only exchange from gradients flowing through inactive branches, with an equal-parameter self-only control.',
 '4. Confirm promising message effects on a new cohort and a harder dataset after demonstrating strong acquisition; three seeds on MNIST do not establish a general branch advantage.',
 '', '## Reproducibility',
 '', '- [Primary driver](../../bdh_cl/scripts/branch_fields_retrieval.py), [50-active control](../../bdh_cl/scripts/branch_fields_flat50_control.py), [scrambling audit](../../bdh_cl/scripts/audit_branch_field_communication.py), [sensory exchange](../../bdh_cl/scripts/branch_field_feature_exchange.py).',
 '- [Primary protocol](../../results/branch_fields_retrieval_20261021/protocol.json), [validation selection](../../results/branch_fields_retrieval_20261021/selection.json), [primary results](../../results/branch_fields_retrieval_20261021/results.json), [50-active results](../../results/branch_fields_retrieval_20261021/flat50_results.json).',
 '- [Scrambled-message audit](../../results/branch_fields_retrieval_20261021/communication_scramble_audit.json), [sensory-exchange results](../../results/branch_fields_retrieval_20261021/feature_exchange_results.json), [summary and paired comparisons](../../results/branch_fields_retrieval_20261021/summary.json).',
 '- Complete checkpoints, all development candidates, verification records and completion manifests are in the results directory. Supplemental selection protocols are saved separately. No test-based model selection or discarded failed arm.',
 '']
clean=[]
for line in lines:
    if '](' not in line:
        line=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',line)
        line=re.sub(r'(?<=[A-Za-z])(?=\.\d)',' ',line)
    clean.append(line)
(HERE/'REPORT.md').write_text('\n'.join(clean))
print('Wrote report, comparison.png, and summary.json')
