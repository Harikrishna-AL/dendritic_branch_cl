"""Summarize missing-method runs without replacing historical comparisons."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/leaderboard_missing_20261031'
OUT=ROOT/'reports/leaderboard_constraints_20261031'
LABELS={'pec':'PEC (ICLR 2024), size-matched','slda':'Streaming LDA, PCA50','sqda':'Streaming QDA, PCA50',
        'rls':'RLS / ridge, PCA50','vfdt':'VFDT (River), PCA50','mist_g':'MIST-G reconstruction, PCA50',
        'mist_k':'MIST-K, unverified diagnostic, PCA50'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=json.loads((SOURCE/'results.json').read_text())
    selected=json.loads((SOURCE/'selected_hyperparameters.json').read_text())
    summary=[]
    for method,label in LABELS.items():
        group=[r for r in runs if r['method']==method]
        assert len(group)==6 and len({(r['seed'],r['order']) for r in group})==6
        values=[st.mean(100*r['scores']['accuracy'] for r in group if r['seed']==seed) for seed in [101,102,103]]
        memory=[r['persistent_numeric_bytes']/1e6 for r in group]
        summary.append(dict(method=method,label=label,accuracy=st.mean(values),sd=st.stdev(values),
            mb=st.mean(memory),memory_min_mb=min(memory),memory_max_mb=max(memory),
            kind='pec' if method=='pec' else 'statistical' if method in ['slda','sqda','rls'] else 'tree',
            representation='784 standardized pixels' if method=='pec' else 'PCA50 from first1000 initial-task samples',
            parameters=group[0]['parameters'],optimizer_mb=group[0]['optimizer_state_bytes']/1e6,
            projection_mb=group[0]['projection_bytes']/1e6,
            training_seconds=st.mean(r['training_seconds'] for r in group),
            nodes_mean=st.mean(r.get('nodes',0) for r in group),
            python_model_heap_mb=st.mean(r.get('python_model_heap_bytes',0)/1e6 for r in group),
            scope=group[0]['scope'],experimental_status='Explicit local adaptation; MIST is a paper reconstruction, not official implementation'))
    prior=json.loads((ROOT/'reports/recent_cl_comparison_20261031/all_baselines.json').read_text())
    # Preserve the requested presentation exclusion. Covariance/analytic and
    # tree comparators are included now because this request explicitly asks
    # to test those missing families. Nearest-class-mean controls stay separate.
    prior=[r for r in prior if r['kind'] not in ['centroid','offline']]
    # A known unresolved reconstruction is retained as a diagnostic, rather
    # than used as evidence in the headline competitive rankings.
    rows=prior+[r for r in summary if r['method']!='mist_k']
    (OUT/'unverified_mist_k_diagnostic.json').write_text(json.dumps(next(r for r in summary if r['method']=='mist_k'),indent=2))
    (OUT/'new_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'all_presented_methods.json').write_text(json.dumps(rows,indent=2))
    with (OUT/'all_presented_methods.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','accuracy_percent','seed_sd_pp','retained_payload_MB','representation','status'])
        writer.writeheader()
        for r in rows:writer.writerow(dict(method=r['label'],accuracy_percent=r['accuracy'],seed_sd_pp=r['sd'],
            retained_payload_MB=r['mb'],representation=r.get('representation','Earlier standardized-pixel protocol'),
            status=r.get('experimental_status','Earlier matched-protocol result')))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white');fig.savefig(OUT/f'{name}.svg',facecolor='white');plt.close(fig)
    def color(r):return '#146eb1' if r['label']=='Branch + auxiliary loss' else '#b84492' if r['kind']=='pec' else '#31846d' if r['kind']=='statistical' else '#cc8a33' if r['kind']=='tree' else '#a3abb4'
    replay_kinds=['replay','derpp','er_ace','ocar']
    no_examples=[r for r in rows if r['kind'] not in replay_kinds]
    ordered=sorted(no_examples,key=lambda r:r['accuracy'])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,8),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    yy=np.arange(len(ordered))
    ax.barh(yy,[r['accuracy'] for r in ordered],color=[color(r) for r in ordered],height=.65)
    ax.errorbar([r['accuracy'] for r in ordered],yy,xerr=[r['sd'] for r in ordered],fmt='none',color='#48525a',capsize=2,lw=.8)
    for i,r in enumerate(ordered):ax.text(r['accuracy']+r['sd']+.8,i,f"{r['accuracy']:.2f}%",va='center',fontsize=9,color=color(r))
    ax.set(yticks=yy,yticklabels=[r['label'].replace('Branch + auxiliary loss','YOUR BRANCH METHOD') for r in ordered],
           xlim=(0,106),xlabel='Final ten-class accuracy (%)',title='No stored historical examples: expanded comparison')
    ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    own=next(r for r in rows if r['label']=='Branch + auxiliary loss')
    pec=next(r for r in rows if r['kind']=='pec')
    bx.bar([0,1],[own['accuracy'],pec['accuracy']],color=['#146eb1','#b84492'],width=.6)
    bx.errorbar([0,1],[own['accuracy'],pec['accuracy']],yerr=[own['sd'],pec['sd']],fmt='none',color='#333',capsize=4)
    for i,r in enumerate([own,pec]):bx.text(i,r['accuracy']+r['sd']+1,f"{r['accuracy']:.2f}%\n{r['mb']:.3f} MB",ha='center',fontsize=12,color=color(r))
    bx.set(xticks=[0,1],xticklabels=['Your branch model','PEC, adapted'],ylim=(0,110),ylabel='Final ten-class accuracy (%)',
           title='Direct neural comparison: same input and total size')
    bx.text(.5,103,'Branch: 529,210 parameters\nPEC: 529,145 including frozen teacher',ha='center',fontsize=10,color='#4d5963')
    fig.suptitle('New exemplar-free baselines under the existing constraints',fontsize=17,fontweight='bold')
    fig.supxlabel('Same 55k/5k/10k split, two epochs/task, seeds 101–103 × two orders; no test task hints. PCA50 methods retain and pay for the projection.\nMIST-G is reconstructed; unverified MIST-K is kept in the report. Raw/frozen nearest-class-mean controls excluded.',fontsize=9)
    save(fig,'01_expanded_replay_free_comparison')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,10),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.35,1]})
    ordered=sorted(rows,key=lambda r:r['accuracy'],reverse=True);yy=np.arange(len(ordered))
    ax.barh(yy,[r['accuracy'] for r in ordered],color=[color(r) for r in ordered],height=.7)
    bx.barh(yy,[r['mb'] for r in ordered],color=[color(r) for r in ordered],height=.7)
    ax.errorbar([r['accuracy'] for r in ordered],yy,xerr=[r['sd'] for r in ordered],fmt='none',color='#555',capsize=2,lw=.7)
    for i,r in enumerate(ordered):
        ax.text(r['accuracy']+r['sd']+.7,i,f"{r['accuracy']:.2f}%",va='center',fontsize=8.5)
        bx.text(r['mb']+.1,i,f"{r['mb']:.3f}",va='center',fontsize=8.5)
    ax.set(yticks=yy,yticklabels=[r['label'] for r in ordered],xlim=(0,108),xlabel='Final ten-class accuracy (%)')
    ax.invert_yaxis();bx.set(xlim=(0,max(r['mb'] for r in rows)*1.2),xlabel='Retained numerical / encoded sketch payload (MB)')
    fig.suptitle('All presented continual methods, including the missing families',fontsize=17)
    fig.supxlabel('Two-epoch shared protocol; feature representations and intrinsic training differ as labeled. MIST reconstructed; VFDT uses River.\nMemory includes PCA, shared frozen teachers, covariance and caches. Optimizers, Python/RNG overhead and peak working RAM excluded.',fontsize=9)
    save(fig,'02_all_methods_accuracy_and_payload')
    fig,ax=plt.subplots(figsize=(11,6.8),layout='constrained')
    for r in rows:
        marker='*' if r is own else 'D' if r['kind']=='pec' else '^' if r['kind'] in ['statistical','tree'] else 'o'
        ax.scatter(r['mb'],r['accuracy'],color=color(r),marker=marker,s=240 if r is own else 85,edgecolor='white',zorder=3)
        ax.errorbar(r['mb'],r['accuracy'],yerr=r['sd'],fmt='none',color=color(r),capsize=2)
    highlights=[(own,(-130,-14)),(pec,(25,8))]
    for method,offset in [('sqda',(10,12)),('slda',(-15,20)),('rls',(10,-48)),('mist_g',(15,-15))]:
        highlights.append((next(r for r in summary if r['method']==method),offset))
    for r,offset in highlights:
        ax.annotate(f"{r['label']}\n{r['accuracy']:.2f}% · {r['mb']:.3f} MB",(r['mb'],r['accuracy']),
                    xytext=offset,textcoords='offset points',fontsize=9,color=color(r),
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1),arrowprops=dict(arrowstyle='-',color=color(r)))
    ax.set(xscale='log',xlim=(min(r['mb'] for r in rows)*.65,max(r['mb'] for r in rows)*1.4),ylim=(0,104),
           xlabel='Retained numeric / encoded sketch payload (MB, log scale)',ylabel='Final ten-class accuracy (%)',
           title='Expanded accuracy–storage comparison; class-mean controls excluded')
    ax.grid(alpha=.18)
    fig.supxlabel('Blue star: branch model. Magenta diamond: size-matched PEC. Triangles: PCA50 statistical/tree methods.\nKLL is charged by serialized state; Python heap and peak working RAM are not comparable from this plot. MIST is a reconstruction.',fontsize=9)
    save(fig,'03_expanded_accuracy_memory')
    lines=['# Missing leaderboard methods tested under our constraints','',
        'This study extends the previous comparison. All values below are **new local measurements**, not copied leaderboard scores. '
        'MIST is a documented paper reconstruction, not an official reproduction.','',
        '| Method | Final accuracy ± seed SD | Retained payload MB | Payload range MB | Additional optimizer MB |',
        '|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['label']} | {r['accuracy']:.2f} ± {r['sd']:.2f}% | {r['mb']:.6f} | {r['memory_min_mb']:.6f}–{r['memory_max_mb']:.6f} | {r['optimizer_mb']:.6f} |")
    lines+=['','Branch + auxiliary reference: **74.76 ± 0.53%, 2.136048 MB**, 529,210 parameters. '
        'Its oracle-route diagnostic remains 97.39%, giving a 22.63-point reference gap. '
        'The new controls do not erase that within-model gap, but they change comparative claims about exemplar-free superiority. '
        '**PEC is the decisive new neural control: it outperforms the branch model with essentially the same total parameter budget and slightly less numerical state.** '
        'The earlier selected-view claim that the branch model leads the measured neural no-example-storage methods is superseded by this run.','',
        '## Shared constraints','',
        'Same cached MNIST, 55,000 training / 5,000 validation indices and official 10,000 test examples as the earlier study; '
        'two epochs per digit pair, batch128 for neural updates, seeds101/102/103, each forward and reverse order. '
        'All ten classes compete at final test; no true test task IDs, pretrained external data, raw-example replay, or old validation examples used for fitting. '
        'Known training boundaries are allowed by the shared protocol. Classical/tree updates consume the same sample sequence one example at a time or through mathematically additive sufficient statistics. '
        'This is a two-epoch adaptation, **not strict single-pass Online CIL**.','',
        '## Representation and capacity','',
        '**PEC:** exact official PecMLP modules via AST import; GELU and LayerNorm, two linear layers, output99. '
        'Ten students of hidden width20 and one shared frozen teacher of width396. Students begin with identical weights, as upstream. '
        'Total529,145 parameters, including350,955 frozen teacher coefficients and178,190 trainable student coefficients; teacher counted once. '
        'The parameter gap from the branch model is only65 coefficients (0.0123%). This matches total stored size, not trainable capacity or FLOPs. '
        'Uses standardized784-pixel inputs, independent continuous Adam optimizers, gradient clipping5, no scheduler, and the shared minibatch/epoch schedule. '
        'Validation chooses LR .0005/.001/.003/.01. The native paper script uses one epoch, batch1, a linear schedule, width10 students and width5000 teacher; '
        'our smaller teacher and different update budget are material adaptations.','',
        '**SLDA/SQDA/RLS/VFDT/MIST:** PCA50 fitted once to the first1,000 presented examples of the initial training task only, then frozen. '
        'The mean and50×784 projection are retained and charged (0.319872 MB, FP64). This follows the leaderboard source paper’s compressed input protocol, '
        'but differs from the standardized-pixel neural backbones. Every transform is fitted without future-task or test data; fitting examples are not retained. '
        'These models do not have an equivalent neural parameter count; all their persistent statistics and transform are instead reported explicitly.','',
        '## Statistical baselines','',
        'SLDA and SQDA use class-wise online counts/means/full centered scatter matrices. Parallel-Welford minibatch merging equals sequential sufficient-statistic accumulation. '
        'SLDA pools within-class covariance; SQDA keeps per-class covariance. Both use (1−s)Σ+sI, empirical class priors, Cholesky precision/log-determinant evaluation; '
        'validation selects s from1e−4/.001/.01. Stored scatter, means/counts, retained precision and log-determinant caches are all charged. '
        'These are covariance-aware classifiers, not the excluded nearest-class-mean controls.','',
        'RLS uses exact ridge sufficient statistics ΦᵀΦ and ΦᵀY, with an appended bias feature and ridge applied to all coordinates. '
        'A solution can be computed from these statistics at any point; final inference solves once and retains the head. '
        'Validation selects ridge .1/1/10. No random expansion or pretrained features beyond the charged PCA transform.','',
        '## Trees and reconstruction limits','',
        '**VFDT:** unmodified River0.26.1 HoeffdingTreeClassifier, Gaussian splitters, Gini criterion, Naive Bayes leaves, tau.05, no drift resets. '
        'Validation candidates (delta,grace)=(.95,200),(1e−7,200),(.95,1000). The library max_size100 MiB is a safety ceiling, not a claimed2 MB memory match. '
        'Reported nodes and actual Python-model heap estimates are saved per run. This is a River adaptation, not the MIST authors’ implementation.','',
        '**MIST-G/K:** no official public implementation was verified. The local reconstruction follows arXiv2605.11617v2 Eq.3–5 and Algorithms1–3: '
        'McDiarmid gap radius sqrt(32ln(2dm/delta)/n), class-conditional KLL medians and CDFs, midpoint threshold candidates, '
        'alpha-discounted truncated-Gaussian child moments, and empty child sketches. Uses k64, alpha.6, delta.1, grace200, tie.05. '
        'MIST-G uses Gaussian leaves; MIST-K uses sketch-CDF likelihoods. The training tree is shared between the two predictors. '
        'KLL library5.2.0 is used directly; its C++ compaction RNG is not explicitly seedable, so the stored checkpoints record each realized state.','',
        'Implementation choices where the paper does not fully specify code behavior: test split every200 new leaf arrivals; confidence n includes inherited weighted mass; '
        'ties may split when the radius falls below.05 as listed in the paper’s configuration table; '
        'absent sketch CDFs fall back to inherited Gaussian estimates for split evaluation; absent MIST-K likelihoods contribute only class priors until observations arrive; '
        'variance floor1e−6 and small truncated-tail probability floors are used for stability. '
        'MIST-K validation selects (epsilon,beta) from(1,.5),(1e−5,.5),(1e−5,1); this is additional tuning beyond the published defaults. '
        'These choices and the two-epoch stream can materially change results. **Do not treat these runs as a verdict on official MIST performance.** '
        'In particular, MIST-K has a severe validation-to-final performance change. Its empty-sketch prediction behavior and split sensitivity are not verified against official code. '
        'Treat MIST-K as an exploratory implementation diagnostic; its low value must not be used to claim superiority over the published method. '
        'Its raw runs and summary are preserved, but it is excluded from headline competitive plots and their presentation CSV.','',
        '## Selection, resources and verification','',
        f'All selections were frozen on seed113 forward validation before final-test evaluation. Selected settings: `{json.dumps(selected,sort_keys=True)}`. '
        'Accuracy SD is across three seed means after averaging orders, not across six independent seeds. '
        'Final models see110,000 labeled sample presentations; PCA has an additional one-time unlabeled fit to the first1,000 current-task examples.','',
        'Memory is decimal MB of retained numerical payload: neural weights, sufficient statistics/caches, PCA, tree numerical fields and encoded KLL sketch state. '
        'River tree numeric fields are traversed explicitly; its separate Python heap estimate includes object overhead. KLL is charged by serialized payload because native allocation size is not exposed. '
        'This metric is not process RSS or peak RAM, and it must not be described as a universal hardware-memory measurement. '
        'Optimizer payload is additional. Input caches, activations/gradients, Python/RNG overhead and temporary PCA/linear-algebra workspaces are excluded, consistently with the earlier numeric-state comparison. '
        'Tree payload varies with the seed/order; both mean and range are shown. Training timings exclude the PCA fit and inference.','',
        'Five focused checks pass: PEC update/inference match actual upstream methods; online moments/QDA match batch Gaussian likelihoods; '
        'RLS matches sequential Sherman–Morrison updates; inherited Gaussian moments conserve discounted mass/mean/variance; MIST checkpoints preserve sketch-based predictions. '
        'Evaluation-state checks verify that test inference does not update models. KLL’s lazy internal sorting is canonicalized before bytewise state comparison.','',
        '## References','',
        '- [PEC paper, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html); '
        '[official code](https://github.com/michalzajac-ml/pec), pinned commit3c15633bb93775b9220ce79080c98f4b1ddc86e1.',
        '- [MIST source paper, v2](https://arxiv.org/html/2605.11617v2).',
        '- [SLDA reference](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf).','',
        '## Artifacts','',
        '- [Expanded no-example-storage comparison](01_expanded_replay_free_comparison.png)',
        '- [All presented methods and payload](02_all_methods_accuracy_and_payload.png)',
        '- [Expanded accuracy–memory plot](03_expanded_accuracy_memory.png)',
        '- [All presented values CSV](all_presented_methods.csv); editable SVG accompanies every PNG.',
        '- Raw runs, validation trials, selections, source, provenance and checkpoints are in `../../results/leaderboard_missing_20261031/`.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
