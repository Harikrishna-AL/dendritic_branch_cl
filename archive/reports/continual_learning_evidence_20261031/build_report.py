"""Complete evidence plots for the comparison series; no new experiments.

Common-protocol results and supplementary pilots are never pooled together.
"""
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RESULTS = ROOT/'results'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})


def read(name):
    return json.loads((RESULTS/name).read_text())


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=190,facecolor='white')
    fig.savefig(OUT/f'{name}.svg',facecolor='white')
    plt.close(fig)


def seed_values(rows,mode='inferred'):
    return [st.mean(100*r['scores'][mode]['accuracy'] for r in rows if r['seed']==s)
            for s in sorted({r['seed'] for r in rows})]


def pareto(rows):
    result,best = [],-np.inf
    for row in sorted(rows,key=lambda r:(r['mb'],-r['accuracy'])):
        if row['accuracy']>best:
            result.append(row)
            best = row['accuracy']
    return result


matched = read('matched_numenta_20261031/results.json')
activity = read('matched_numenta_activity_20261031/results.json')
rows = read('strong_replay_20261031/all_summary_values.json')
for setting,label in [('kw20','Numenta (20% activity)*'),('relu','Numenta (ReLU)*')]:
    source = [r for r in activity if r['setting']==setting]
    values = seed_values(source)
    rows.append({'label':label,'accuracy':st.mean(values),'sd':st.stdev(values),
                 'mb':(source[0]['model_state_bytes']+5*784*4)/1e6,'kind':'other',
                 'experimental_status':'exploratory activity sensitivity'})
assert len(rows)==17
continual = [r for r in rows if r['kind']!='offline']
joint = next(r for r in rows if r['kind']=='offline')
front = pareto(continual)
lookup = {r['label']:r for r in rows}
assert 'Branch + auxiliary loss' not in {r['label'] for r in front}
colors = {'branch':'#236b9d','centroid':'#39846a','derpp':'#8954a7','er_ace':'#258e88',
          'replay':'#bc873a','other':'#89929a','offline':'#bfc3c7'}

with (OUT/'matched_baselines.csv').open('w',newline='') as handle:
    writer = csv.DictWriter(handle,fieldnames=['method','accuracy_percent','seed_sd_pp','retained_state_MB','family','experimental_status'])
    writer.writeheader()
    for row in rows:
        writer.writerow({'method':row['label'],'accuracy_percent':row['accuracy'],'seed_sd_pp':row['sd'],
                         'retained_state_MB':row['mb'],'family':row['kind'],
                         'experimental_status':row.get('experimental_status','common protocol; offline reference if labeled')})
(OUT/'matched_baselines.json').write_text(json.dumps(rows,indent=2))
(OUT/'complete_frontier.json').write_text(json.dumps(front,indent=2))

# 1. Full measured frontier. No class-mean controls are removed.
fig,ax = plt.subplots(figsize=(11.8,6.9))
fig.subplots_adjust(left=.085,right=.97,bottom=.23,top=.80)
for row in continual:
    marker = '*' if row['label']=='Branch + auxiliary loss' else 'D' if row['kind']=='centroid' else 'o'
    size = 240 if marker=='*' else 65
    ax.errorbar(row['mb'],row['accuracy'],yerr=row['sd'],fmt='none',ecolor=colors[row['kind']],capsize=3,lw=1,zorder=2)
    ax.scatter(row['mb'],row['accuracy'],s=size,marker=marker,c=colors[row['kind']],edgecolors='white',lw=.7,zorder=4)
ax.plot([r['mb'] for r in front],[r['accuracy'] for r in front],color='#424f48',ls='--',lw=1.4,zorder=1)
ax.axhline(joint['accuracy'],color='#a8adb2',ls=':',lw=1)
ax.set(xscale='log',xlim=(.022,12),ylim=(0,103),xlabel='Retained continual-learning state (MB, logarithmic scale) ↓',
       ylabel='Final ten-class accuracy (%) ↑')
ax.grid(alpha=.22)

def callout(axis,key,text,offset,align='left'):
    row = lookup[key]
    axis.annotate(text,(row['mb'],row['accuracy']),xytext=offset,textcoords='offset points',ha=align,va='center',
                  color=colors[row['kind']],fontsize=9.5,
                  bbox={'facecolor':'white','edgecolor':'none','pad':1,'alpha':.9},
                  arrowprops={'arrowstyle':'-','color':colors[row['kind']],'lw':.7})

callout(ax,'Raw-pixel class means','Raw class means\n82.32% · 0.0314 MB',(18,20))
callout(ax,'Frozen-feature class means','Frozen-feature class means\n81.71% · 2.118 MB',(-175,38))
callout(ax,'Branch + auxiliary loss','Current branch model\n74.76% · 2.136 MB',(-170,-12))
callout(ax,'Branch CE-only','Branch CE-only\n72.11%',(-170,-47))
callout(ax,'DER++: 1,000 images','DER++ / 1,000\n94.52% · 2.951 MB',(25,6))
callout(ax,'Online EWC','Online EWC\n23.38%',(-15,34),align='right')
callout(ax,'Synaptic Intelligence','SI\n20.34%',(-10,-26),align='right')
callout(ax,'Numenta (5% activity)','Numenta variants\n18.57–20.28%',(-30,-32),align='right')
callout(ax,'Fine-tuning','Fine-tuning\n20.31%',(-125,-3))
ax.text(.027,joint['accuracy']+1,'Joint offline reference: 96.97%',fontsize=9,color='#7a8289')
handles = [Line2D([0],[0],color='#424f48',ls='--',label='Frontier over ALL measured continual baselines')]
for kind,label in [('branch','Branch models'),('centroid','Class-mean controls'),('derpp','DER++ (2020)'),('er_ace','ER-ACE (2022)'),('replay','Plain replay'),('other','Other neural baselines')]:
    handles.append(Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[kind],markersize=7,label=label))
ax.legend(handles=handles,loc='center left',bbox_to_anchor=(.01,.44),frameon=False,fontsize=8.7)
fig.suptitle('Complete measured accuracy–memory comparison',x=.085,y=.96,ha='left',fontsize=18,fontweight='bold')
fig.text(.085,.90,'Common Split MNIST protocol · all class-mean controls included · no latest-SOTA claim',fontsize=10.5,color='#515963')
fig.text(.085,.125,'The current branch model is dominated when raw class means are included. This is a baseline to improve on.',fontsize=9.5,color='#354d5a')
fig.text(.085,.08,'Numeric retained state excludes Adam and peak working memory. Offline joint training does not define the frontier.',fontsize=9,color='#58616a')
fig.text(.085,.045,'Bars: SD across 3 seed means, averaging 2 task orders. Activity variants are exploratory; raw class means are deterministic.',fontsize=8.7,color='#58616a')
save(fig,'01_complete_pareto')

# 2. All matched-protocol baseline values, with memory beside accuracy.
ordered = sorted(rows,key=lambda r:r['accuracy'],reverse=True)
fig,axes = plt.subplots(1,2,figsize=(12.8,9),layout='constrained',sharey=True,gridspec_kw={'width_ratios':[1.6,1]})
yy = np.arange(len(ordered))
for axis,field in zip(axes,['accuracy','mb']):
    bars = axis.barh(yy,[r[field] for r in ordered],color=[colors[r['kind']] for r in ordered],height=.65)
    bars[0].set_hatch('//')
    for i,row in enumerate(ordered):
        offset = row['sd']+1 if field=='accuracy' else .08
        text = f'{row[field]:.2f}%' if field=='accuracy' else (f'{1000*row[field]:.1f} kB' if row[field]<.1 else f'{row[field]:.3f}')
        if field=='mb' and row['kind']=='offline':
            text += ' + offline data'
        axis.text(row[field]+offset,i,text,va='center',fontsize=8.5)
axes[0].errorbar([r['accuracy'] for r in ordered],yy,xerr=[r['sd'] for r in ordered],fmt='none',ecolor='#333333',capsize=2,lw=.8)
axes[0].set(yticks=yy,yticklabels=[r['label'] for r in ordered],xlim=(0,108),xlabel='Final ten-class accuracy (%)')
axes[0].invert_yaxis()
axes[1].set(xlim=(0,10),xlabel='Retained numeric state (MB)')
fig.suptitle('All measured baselines under the common protocol',fontsize=16)
fig.supxlabel('* Exploratory activity controls. Joint is offline. Memory excludes Adam, activations and input dataset caches.\nLearned networks use ~529k coefficients; raw class means are an intentionally smaller statistical control.',fontsize=9)
save(fig,'02_all_baseline_values')

# 3. Oracle-access gap is a diagnostic, never a deployable score.
diagnostics = []
for arm,label in [('branch_aux','Branch + auxiliary'),('branch_ce','Branch CE-only'),('numenta','Numenta 5%')]:
    group = [r for r in matched if r['arm']==arm]
    actual,oracle = seed_values(group),seed_values(group,'oracle')
    diagnostics.append({'method':label,'actual':st.mean(actual),'oracle':st.mean(oracle),'actual_sd':st.stdev(actual),'oracle_sd':st.stdev(oracle)})
for setting,label in [('kw20','Numenta 20%*'),('relu','Numenta ReLU*')]:
    group = [r for r in activity if r['setting']==setting]
    actual,oracle = seed_values(group),seed_values(group,'oracle')
    diagnostics.append({'method':label,'actual':st.mean(actual),'oracle':st.mean(oracle),'actual_sd':st.stdev(actual),'oracle_sd':st.stdev(oracle)})
fig,ax = plt.subplots(figsize=(10,5.3),layout='constrained')
xx = np.arange(len(diagnostics))
ax.bar(xx-.18,[r['actual'] for r in diagnostics],.34,yerr=[r['actual_sd'] for r in diagnostics],capsize=3,color='#286d9c',label='Autonomous inference')
ax.bar(xx+.18,[r['oracle'] for r in diagnostics],.34,yerr=[r['oracle_sd'] for r in diagnostics],capsize=3,color='#c39848',label='Oracle configuration/context (diagnostic)')
for i,row in enumerate(diagnostics):
    ax.text(i-.18,row['actual']+row['actual_sd']+1.2,f'{row["actual"]:.2f}',ha='center',fontsize=8.5)
    ax.text(i+.18,row['oracle']+row['oracle_sd']+1.2,f'{row["oracle"]:.2f}',ha='center',fontsize=8.5)
ax.set(xticks=xx,xticklabels=[r['method'] for r in diagnostics],ylim=(0,119),ylabel='Final ten-class accuracy (%)',title='Research opportunity: retained capability versus autonomous access')
ax.legend(frameon=False,fontsize=9,loc='upper right')
fig.supxlabel('Oracle uses the true task/pair to choose a stored route or context; all ten output classes remain eligible.\nThe 22.63-point branch gap is diagnostic headroom, not guaranteed recoverable accuracy. * Exploratory variants.',fontsize=9)
save(fig,'03_retention_access_gap')
(OUT/'retention_access_gap.json').write_text(json.dumps(diagnostics,indent=2))

# 4. Paired auxiliary-loss effect, with its uncertainty explicit.
seeds = [101,102,103]
aux = seed_values([r for r in matched if r['arm']=='branch_aux'])
ce = seed_values([r for r in matched if r['arm']=='branch_ce'])
delta = np.array(aux)-np.array(ce)
half = 4.30265273*st.stdev(delta)/np.sqrt(3)
effect = {'paired_seed_differences_pp':delta.tolist(),'mean_pp':float(np.mean(delta)),
          'descriptive_95pct_t_interval_pp':[float(np.mean(delta)-half),float(np.mean(delta)+half)]}
fig,(ax,bx) = plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
for i,seed in enumerate(seeds):
    ax.plot([0,1],[ce[i],aux[i]],'o-',lw=1.5,label=f'Seed {seed}')
ax.set(xticks=[0,1],xticklabels=['CE-only branch','Auxiliary-loss branch'],ylim=(65,81),ylabel='Mean accuracy across task orders (%)',title='Paired seeds: isolate the auxiliary loss')
ax.legend(frameon=False,fontsize=9)
bx.axvline(0,color='#939aa1',ls=':',lw=1)
bx.scatter(delta,[2,1,0],s=55,c='#527f9b')
bx.errorbar(np.mean(delta),-1.1,xerr=half,fmt='D',capsize=5,color='#1e567b',ms=7)
bx.set(yticks=[2,1,0,-1.1],yticklabels=['Seed 101','Seed 102','Seed 103','Mean + 95% interval'],xlabel='Auxiliary minus CE-only accuracy (pp)',title=f'Mean effect: {np.mean(delta):+.2f} percentage points')
bx.grid(axis='x',alpha=.2)
fig.supxlabel('Three independent seed means after averaging both orders. The descriptive paired t interval includes zero.\nThis small cohort does not establish a reliable causal advantage of the auxiliary loss.',fontsize=9)
save(fig,'04_auxiliary_loss_ablation')
(OUT/'auxiliary_effect.json').write_text(json.dumps(effect,indent=2))

# 5. Original small-network online-context pilot, kept separate.
pilot = read('online_prototype_routing_20261031/results.json')
arms = [('current','Current gate'),('random_saved_route','Random saved route'),('input_prototype','Online input prototypes'),
        ('input_prototype_noise_0.1','Input prototypes + noise'),('latent_prototype','Online latent prototypes'),
        ('latent_prototype_noise_0.1','Latent prototypes + noise'),('oracle','Oracle saved route')]
fig,ax = plt.subplots(figsize=(10,6.5),layout='constrained')
yy = np.arange(len(arms))
for offset,objective,color in [(-.17,'counterfactual','#286d9c'),(.17,'ce_only','#b88a46')]:
    group = [r for r in pilot if r['objective']==objective]
    means = [st.mean(100*r['scores'][arm]['accuracy'] for r in group) for arm,_ in arms]
    sds = [st.stdev(100*r['scores'][arm]['accuracy'] for r in group) for arm,_ in arms]
    ax.barh(yy+offset,means,.31,xerr=sds,capsize=3,color=color,label='Auxiliary objective' if objective=='counterfactual' else 'CE-only + error traces')
    for y,v,sd in zip(yy+offset,means,sds):
        ax.text(v+sd+.8,y,f'{v:.2f}',va='center',fontsize=8)
ax.set(yticks=yy,yticklabels=[label for _,label in arms],xlim=(0,111),xlabel='Final ten-class accuracy (%)',title='Supplementary: original online context-tracker pilot')
ax.invert_yaxis()
ax.legend(frameon=False,loc='upper right',fontsize=9)
fig.supxlabel('Different protocol: 75,610 parameters, 60,000 training inputs, seeds 34–36, forward order only.\nDo not pool with the matched-size runs. Banks saturated in task 1; latent prototypes also face encoder-coordinate drift.',fontsize=9)
save(fig,'05_online_context_pilot')

# 6. Benchmark sanity and unmatched full-size pilot, never pooled with primary.
native = read('numenta_native_split_mnist_20261031/results.json')
small = read('numenta_native_split_mnist_20261031/existing_branch_control.json')
sanity = read('numenta_permuted_mnist_sanity_20261031/results.json')
fig,(ax,bx) = plt.subplots(1,2,figsize=(11,5),layout='constrained')
vals = [[100*r['accuracy'] for r in small]]+[[100*r['scores']['autonomous']['accuracy'] for r in native if r['context']==mode] for mode in ['prototype','input']]
means = [st.mean(v) for v in vals]
ax.bar(range(3),means,yerr=[st.stdev(v) for v in vals],capsize=4,color=['#286d9c','#a28e67','#aab49b'])
for i,v in enumerate(means):
    ax.text(i,v+3,f'{v:.2f}%',ha='center',fontsize=9)
ax.set(xticks=range(3),xticklabels=['Small branch\n75.6k params','Numenta prototype\n21.9M params','Numenta input\n21.9M params'],ylim=(0,106),ylabel='Final accuracy (%)',title='Split MNIST: unmatched exploratory pilot')
values = [100*v for v in sanity['stages'][-1]['task_accuracy']]
bx.bar(range(1,6),values,color='#b88a46')
bx.axhline(st.mean(values),ls='--',color='#50575c',label=f'Mean {st.mean(values):.2f}%')
bx.set(xticks=range(1,6),ylim=(0,106),xlabel='Permutation task',title='Native Permuted MNIST recipe: one-seed check')
bx.legend(frameon=False,loc='lower left')
fig.supxlabel('Different benchmarks, data budgets, model sizes and schedules; no cross-panel ranking or superiority inference.\nThe native-recipe check uses 1 epoch/task, Adam .0005, batch 256 and optimizer resets; Split pilots use their recorded adaptations.',fontsize=9)
save(fig,'06_protocol_and_native_checks')

paper_dates = {'DER++':{'year':2020,'venue':'NeurIPS','url':'https://papers.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html'},
               'ER-ACE':{'year':2022,'venue':'ICLR','url':'https://openreview.net/forum?id=N8MaByOzUfb'},
               'scope':'Established strong comparators; not the latest papers or a complete current-SOTA evaluation.'}
(OUT/'paper_dates.json').write_text(json.dumps(paper_dates,indent=2))
readme = ['# Evidence for the next continual-learning research step','',
          '**DER++ is NeurIPS 2020; ER-ACE is ICLR 2022. They are established strong comparators, not the latest papers.** '
          'This plot package presents our measured results; it does not establish novelty or a current-SOTA claim.', '',
          '## Measured strengths and open questions','',
          '- Under the matched protocol, the branch model reaches 74.76% autonomous accuracy and 97.39% with oracle route selection, with the same final weights and all ten logits eligible. '
          'The 22.63-point gap motivates work on accessing retained capabilities. Oracle access is privileged and is not a deployable score or a guarantee of achievable improvement.',
          '- The current branch model exceeds the tested fine-tuning, SI, online EWC and Numenta variants under this protocol.',
          '- Raw class means reach 82.32% with 0.03144 MB and dominate the current branch model when all measured methods are included. '
          'Frozen-feature class means also exceed current branch accuracy. These controls are retained in the complete plots.',
          '- DER++ reaches 88.44% with 200 stored images and 94.52% with 1,000, at 2.285 and 2.951 MB respectively. '
          'These replay methods have extra gradient exposures and retain historical examples; their results provide concrete competitive targets.',
          f'- The auxiliary objective improves the matched CE-only branch by {effect["mean_pp"]:.2f} pp on average, '
          f'but the three-seed descriptive 95% paired interval is [{effect["descriptive_95pct_t_interval_pp"][0]:.2f}, {effect["descriptive_95pct_t_interval_pp"][1]:.2f}] pp. '
          'The specific contribution of the auxiliary loss remains uncertain.', '',
          'A defensible research statement is: **We investigate whether improved context learning can make retained task-associated capabilities autonomously accessible under fixed memory and without historical-example replay.** '
          'Whether the mechanism is novel and whether it improves competitive accuracy require further literature review and experiments.', '',
          '## Figures','',
          '1. [Complete accuracy–memory Pareto plot](01_complete_pareto.png): all matched-protocol baselines, including both class-mean controls and the exploratory Numenta activity variants. '
          'The branch model is not on this complete measured frontier. Joint is an offline reference.',
          '2. [All baseline values](02_all_baseline_values.png): complete measured accuracy and retained state, with no class-mean exclusions.',
          '3. [Retained capability versus access](03_retention_access_gap.png): oracle diagnostics are shown separately from autonomous results.',
          '4. [Paired auxiliary-loss ablation](04_auxiliary_loss_ablation.png): per-seed effects and uncertainty.',
          '5. [Online prototype-tracker pilot](05_online_context_pilot.png): original small-network protocol, kept separate.',
          '6. [Native model and benchmark checks](06_protocol_and_native_checks.png): unmatched pilots and one-seed native-recipe sanity check, kept separate.', '',
          'Each figure has an editable SVG with the same basename. [CSV](matched_baselines.csv), [JSON](matched_baselines.json), '
          '[complete frontier](complete_frontier.json), [oracle diagnostics](retention_access_gap.json), and [paired effect](auxiliary_effect.json) contain exact values.', '',
          '## Interpretation and provenance','',
          'All primary scores use the same 55,000 training / 5,000 validation / official 10,000 test split, three seeds, and forward/reverse task orders. '
          'SD averages orders within each seed before measuring between-seed variability. Raw class means are deterministic. '
          'Joint training is offline and requires dataset access; its displayed state excludes that dataset. '
          'Continual models use two current-task epochs, matched preprocessing, Adam, clip5, and separately validation-selected hyperparameters. '
          'Hyperparameter trial counts and method-specific deviations are in the original reports; regularization and DER++ receive more search trials than the branch model.', '',
          'Retained state includes model tensors, masks, prototypes, error traces, route archives, regularizer state, replay images/labels and stored logits where applicable. '
          'It excludes Adam, transient gradients/activations, dataset caches and Python/RNG overhead. It is not peak RAM. '
          'The original baseline report records additional optimizer state. The log memory axis is explicitly labeled. '
          'Frontiers use exact bytes and measured means; segments are visual guides and no statistical significance is inferred from membership.', '',
          'Sources: [baseline suite](../../results/continual_baseline_memory_20261031/REPORT.md), '
          '[matched comparison](../../results/matched_numenta_20261031/REPORT.md), '
          '[activity controls](../../results/matched_numenta_activity_20261031/REPORT.md), '
          '[strong replay comparison](../../results/strong_replay_20261031/REPORT.md), '
          '[online context pilot](../../results/online_prototype_routing_20261031/REPORT.md), '
          '[native model pilots](../../results/numenta_native_split_mnist_20261031/REPORT.md), '
          '[Permuted MNIST check](../../results/numenta_permuted_mnist_sanity_20261031/REPORT.md).', '',
          'Paper sources: [DER++ / NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), '
          '[ER-ACE / ICLR 2022](https://openreview.net/forum?id=N8MaByOzUfb).', '',
          'Rebuild: `python reports/continual_learning_evidence_20261031/build_report.py`.', '']
(OUT/'README.md').write_text('\n'.join(readme))
print('Created 6 PNG/SVG figure pairs, complete baseline CSV/JSON, diagnostic values and README.')
print('Complete frontier:',[(r['label'],round(r['accuracy'],2)) for r in front])
print('Auxiliary effect:',effect)
