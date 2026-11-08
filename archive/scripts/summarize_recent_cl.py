"""Recent comparator results, explicit adaptation limits and all-method plots."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/recent_cl_20261031'
OUT = ROOT/'reports/recent_cl_comparison_20261031'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results = json.loads((SOURCE/'results.json').read_text())
    prior = json.loads((ROOT/'reports/continual_learning_evidence_20261031/matched_baselines.json').read_text())
    rows = list(prior)
    recent = []
    for method,label in [('ocar','OCAR (ICML 2025), adapted'),('rfe','RFE (TMLR 2026), adapted')]:
        runs = [r for r in results if r['method']==method]
        assert len(runs)==6 and len({(r['seed'],r['order']) for r in runs})==6
        values = [st.mean(100*r['scores']['accuracy'] for r in runs if r['seed']==s) for s in [101,102,103]]
        first = runs[0]
        entry = dict(label=label,accuracy=st.mean(values),sd=st.stdev(values),
                     mb=first['persistent_numeric_bytes']/1e6,kind=method,
                     experimental_status='Explicit local adaptation; different intrinsic optimizer/modules; not original-paper reproduction',
                     parameters=first['parameters'],model_mb=first['model_bytes']/1e6,
                     teacher_mb=first.get('teacher_bytes',0)/1e6,curvature_mb=first.get('curvature_bytes',0)/1e6,
                     buffer_mb=first['buffer_bytes']/1e6,optimizer_mb=first['optimizer_state_bytes']/1e6,
                     max_optimizer_mb=first.get('max_optimizer_state_bytes',first['optimizer_state_bytes'])/1e6,
                     extra_module_presentations=first['extra_module_presentations'],
                     mean_training_seconds=st.mean(r['training_seconds'] for r in runs))
        rows.append(entry)
        recent.append(entry)
    (OUT/'all_baselines.json').write_text(json.dumps(rows,indent=2))
    (OUT/'recent_summary.json').write_text(json.dumps(recent,indent=2))
    with (OUT/'all_baselines.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['method','accuracy_percent','seed_sd_pp','retained_state_MB','status'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(method=r['label'],accuracy_percent=r['accuracy'],seed_sd_pp=r['sd'],
                                 retained_state_MB=r['mb'],status=r.get('experimental_status','Earlier common-protocol run')))
    colors = {'branch':'#236b9d','centroid':'#39846a','derpp':'#8954a7','er_ace':'#258e88',
              'replay':'#bc873a','other':'#89929a','offline':'#bfc3c7','ocar':'#d75445','rfe':'#cf478d'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white')
        fig.savefig(OUT/f'{name}.svg',facecolor='white')
        plt.close(fig)
    # Complete values, with adaptations marked rather than silently pooled.
    ordered = sorted(rows,key=lambda r:r['accuracy'],reverse=True)
    fig,(ax,bx) = plt.subplots(1,2,figsize=(13,10),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    y = np.arange(len(ordered))
    ax.barh(y,[r['accuracy'] for r in ordered],color=[colors[r['kind']] for r in ordered])
    ax.errorbar([r['accuracy'] for r in ordered],y,xerr=[r['sd'] for r in ordered],fmt='none',capsize=2,color='#333',lw=1)
    bx.barh(y,[r['mb'] for r in ordered],color=[colors[r['kind']] for r in ordered])
    for i,r in enumerate(ordered):
        ax.text(r['accuracy']+r['sd']+.8,i,f"{r['accuracy']:.2f}%",va='center',fontsize=9)
        bx.text(r['mb']+.1,i,f"{r['mb']:.3f}"+(' + offline data' if r['kind']=='offline' else ''),va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=[r['label'] for r in ordered],xlim=(0,108),xlabel='Final ten-class accuracy (%)')
    ax.invert_yaxis()
    bx.set(xlim=(0,max(r['mb'] for r in rows)*1.18),xlabel='Retained numeric state (MB)')
    fig.suptitle('Split MNIST: complete measured comparison, extended to 2025–2026 papers',fontsize=15)
    fig.supxlabel('Same split, current-data exposure and evaluation cohort. OCAR/RFE are adapted runs; intrinsic optimizers and extra compute differ.\nMemory includes retained teachers/curvature/buffers; excludes optimizer state and peak working RAM. * Exploratory controls.',fontsize=9)
    save(fig,'01_all_baselines_recent')
    # Pareto over all methods, not just favorable selected methods.
    continual = [r for r in rows if r['kind']!='offline']
    frontier,best = [],-1
    for r in sorted(continual,key=lambda r:(r['mb'],-r['accuracy'])):
        if r['accuracy']>best:
            frontier.append(r)
            best = r['accuracy']
    fig,ax = plt.subplots(figsize=(11,6),layout='constrained')
    for r in continual:
        special = r['kind'] in ['ocar','rfe','branch']
        marker = '*' if r['label']=='Branch + auxiliary loss' else 's' if r['kind'] in ['ocar','rfe'] else 'o'
        ax.errorbar(r['mb'],r['accuracy'],yerr=r['sd'],fmt='none',color=colors[r['kind']],capsize=3)
        ax.scatter(r['mb'],r['accuracy'],s=140 if special else 50,marker=marker,color=colors[r['kind']],edgecolor='white',zorder=3)
    ax.plot([r['mb'] for r in frontier],[r['accuracy'] for r in frontier],ls='--',color='#555',lw=1,label='Complete measured frontier')
    offsets = {'Branch + auxiliary loss':(-160,-15),'Raw-pixel class means':(15,15),
               'DER++: 1,000 images':(-140,15),'OCAR (ICML 2025), adapted':(-175,-15),
               'RFE (TMLR 2026), adapted':(15,-15)}
    for label,offset in offsets.items():
        r = next(r for r in rows if r['label']==label)
        ax.annotate(f"{label}\n{r['accuracy']:.2f}% · {r['mb']:.3f} MB",(r['mb'],r['accuracy']),
                    xytext=offset,textcoords='offset points',fontsize=9,color=colors[r['kind']],
                    arrowprops=dict(arrowstyle='-',color=colors[r['kind']]),
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
    ax.set(xscale='log',xlim=(.02,25),ylim=(0,103),xlabel='Retained numeric state (MB, logarithmic scale) ↓',ylabel='Final ten-class accuracy (%) ↑',
           title='Accuracy–memory trade-off, including adapted recent methods')
    ax.legend(loc='lower left',frameon=False)
    ax.grid(alpha=.2)
    fig.supxlabel('All class-mean controls included. Adaptations are squares; no claim of reproducing published SOTA scores.\nError bars: SD across three seed means, each averaging forward/reverse task order. Optimizer/peak RAM excluded.',fontsize=9)
    save(fig,'02_complete_pareto_recent')
    # Numeric state components explain why similar parameter budgets differ.
    fig,ax = plt.subplots(figsize=(10,4.5),layout='constrained')
    labels = ['Branch + auxiliary','OCAR 2025, adapted','RFE 2026, adapted']
    branch = next(r for r in rows if r['label']=='Branch + auxiliary loss')
    data = [dict(model_mb=branch['mb'],teacher_mb=0,curvature_mb=0,buffer_mb=0)]+recent
    left = np.zeros(3)
    for key,label,color in [('model_mb','Model / branch state','#236b9d'),('teacher_mb','Retained teacher','#cf478d'),
                            ('curvature_mb','Curvature factors + inverse','#d75445'),('buffer_mb','Historical images','#bc873a')]:
        values = np.array([r[key] for r in data])
        ax.barh(np.arange(3),values,left=left,label=label,color=color)
        left += values
    for i,r in enumerate([branch]+recent):
        ax.text(left[i]+.15,i,f"{r['mb']:.3f} MB",va='center')
    ax.set(yticks=np.arange(3),yticklabels=labels,xlim=(0,max(left)*1.2),xlabel='Retained numeric state (MB)',title='Similar deployed parameter counts; different continual-learning state')
    ax.invert_yaxis()
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fig.supxlabel('Branch: 529,210 coefficients; OCAR: 529,480; RFE: 528,014 including all auxiliary modules.\nTiny mapping/scalar state is included in totals. Branch model/context/routes are combined. Optimizers and peak RAM are separate.',fontsize=9)
    save(fig,'03_retained_state_components')
    lines = ['# Recent continual-learning comparison: 2025–2026 publications','',
             'These are new measured **adaptations**, not original-paper reproductions or a verified latest-SOTA leaderboard. '
             'Recent publication year alone does not establish SOTA. Published scores on other datasets are not mixed into Split-MNIST plots.','',
             '| Method | Accuracy ± seed SD | Retained state MB | Additional optimizer MB | Deployed coefficients |',
             '|---|---:|---:|---:|---:|']
    for r in recent:
        lines.append(f"| {r['label']} | {r['accuracy']:.2f} ± {r['sd']:.2f}% | {r['mb']:.6f} | {r['optimizer_mb']:.6f} | {r['parameters']:,} |")
    lines += ['', '## What the evidence supports','',
              'The matched branch model has 74.76% autonomous accuracy versus 97.39% with oracle route selection: a 22.63-point diagnostic access gap. '
              'Oracle selection uses the true task/pair to select a route, with all ten output classes eligible. This motivates context/routing research; '
              'it does not guarantee recovery or establish novelty. The auxiliary-loss estimate is +2.65 points over the CE-only branch, '
              'with a descriptive paired 95% t interval of −3.30 to +8.61 points across only three seed means.','',
              'Raw class means (82.32%, 0.03144 MB) and DER++ with 1,000 images (94.52%, 2.951096 MB) remain important measured targets. '
              'A short adapted run that performs poorly does not refute a method’s native published results.','',
              '## Recent literature selection and status','',
              '- **OCAR, ICML 2025:** [official publication](https://proceedings.mlr.press/v267/urettini25a.html), '
              '[official code](https://github.com/edo-urettini/CL_stability). Curvature-aware replay using K-FAC. Reports improved continual metrics on its native benchmarks; adapted and measured here.',
              '- **RFE, TMLR January 2026:** [paper](https://openreview.net/pdf?id=9NnhVME4Q6), '
              '[official code](https://github.com/mail-research/retrospective-feature-estimation). Chains retrospector modules to estimate old feature spaces. '
              'Relevant recent replay-free comparator, not claimed here as an overall SOTA winner; adapted and measured here.',
              '- **LOR, CVPR 2026:** [official publication](https://openaccess.thecvf.com/content/CVPR2026/html/Lai_Beyond_Myopic_Alignment_Lookahead_Optimization_for_Online_Class-Incremental_Learning_CVPR_2026_paper.html). '
              'Claims SOTA on Seq-CIFAR10/100 and Seq-TinyImageNet. No official implementation was verified during this audit; **not run**, no invented Split-MNIST value.',
              '- **SES, ICLR 2025:** [official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/92ce40962b4098f7bf6eed33128fc606-Paper-Conference.pdf). '
              'Includes Split-MNIST Class-IL and structural-entropy sample selection. Its task-level coreset selection has a different data-access regime from an online reservoir; **not run** here.',
              '- **MoAL, CVPR 2025:** [official publication](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Knowledge_Memorization_and_Rumination_for_Pre-trained_Model-based_Class-Incremental_Learning_CVPR_2025_paper.html). '
              'Uses pre-trained models and analytical learning; not directly comparable to this from-scratch MLP budget. **Not run** here.','',
              'This is a targeted recent-literature extension, not an exhaustive 2025–2026 SOTA audit.','',
              '## Shared controls','',
              'Same cached MNIST, fixed 55k training / 5k validation split and official 10k test, standardized pixels, '
              'two current-task epochs, batch 128, seeds 101/102/103 and forward/reverse task orders. All ten classes compete at test time without true task IDs. '
              'Six validation configurations per method on seed 113 forward; frozen before any official-test evaluation. '
              'SD is across three seed means after averaging orders, not six independent seeds.','',
              '## Material adaptations','',
              '**OCAR:** same 784→666 tanh MLP as previous dense baselines. Uses the official nngeometry FIM, PMatKFAC inverse, and gradient-vector kernels. '
              'The released NGPlugin is configured with CLEAR/domain-specific weighting and warns that Class-IL requires manual changes. '
              'Our explicit Class-IL branch weights replay Fisher contributions by known/current class count, using all ten outputs, '
              'rather than importing that CLEAR branch unchanged. SGD is intrinsic to the natural-gradient algorithm; no Adam and no gradient clipping, as upstream. '
              'One update per batch. Same 200-image uint8 reservoir as previous replay arms. '
              'EMA is the released CIFAR100 value 0.1385127; validation tests three LRs (.0005/.001/.003) × two damping increments (3.67164e−8, 1e−5). '
              'This preserves exact library curvature kernels but is **not** a claim of reproducing the full original training pipeline.','',
              '**RFE:** uses the official auxiliary CNN and Retrospector classes verbatim via AST loading. '
              'Main extractor 784→444 ReLU; d_aux=32 and d_projection=16; all five two-way classifiers, five auxiliary CNNs and four retrospectors are counted. '
              'This yields 528,014 deployed coefficients (0.23% fewer than the branch model). '
              'ReLU matches the original nonnegative feature space; it differs from the earlier tanh backbone. '
              'Auxiliary inputs use bilinear 28→32 resizing and repeat grayscale across RGB channels to satisfy the official module shape. '
              'Uses two additional module-training epochs per task, Adam .005, and a continuous backbone Adam with validation LR .0005/.001/.003 × regularization 0/1. '
              'No augmentation or learning-rate scheduler in this short shared-budget adaptation. '
              'The two-way task heads are evaluated for every image and combined exactly as the official Class-IL inference; no test task is supplied. '
              'Teacher extractor/classifier copies required to continue learning are counted separately from deployed coefficients. '
              'These smaller modules and short schedules differ materially from the original ResNet experiment.','',
              '## Resources and checks','',
              'Numeric retained state includes models, teachers, context/routes, curvature factors and retained inverse factors, raw replay buffers, and mapping/scalar state. '
              'Optimizer state is additional. This is not peak system RAM: dataset caches, gradients, temporary modules/activations and Python/RNG overhead are excluded. '
              'RFE’s temporary module Adam peak is recorded separately in JSON, as are extra module presentations and training time. '
              'A parameter match is not a compute or peak-memory match.','',
              'Three focused tests pass: Class-IL predictions match the actual upstream RFE inference; curvature/EMA agree with the upstream Fisher implementation and remove hooks; '
              'all auxiliary RFE modules fit the declared parameter budget. Evaluation state hashes remain unchanged. '
              'Pinned upstream commits and exact source are in the experiment directory.','',
              '## Artifacts','',
              '- [All measured baseline values](01_all_baselines_recent.png)',
              '- [Complete Pareto plot](02_complete_pareto_recent.png)',
              '- [Retained-state components](03_retained_state_components.png)',
              '- [All data CSV](all_baselines.csv); editable SVG copies accompany every PNG.',
              '- Earlier diagnostic and ablation plots remain in `../continual_learning_evidence_20261031/`.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(recent,indent=2))


if __name__=='__main__':
    main()
