"""Extend the existing neural-only Pareto view with measured DER++/ER-ACE."""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def frontier(rows):
    result = []
    best = -np.inf
    for row in sorted(rows,key=lambda r:(r['mb'],-r['accuracy'])):
        if row['accuracy']>best:
            result.append(row)
            best = row['accuracy']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    args = parser.parse_args()
    out = args.out
    measured = json.loads((out/'results.json').read_text())
    trials = json.loads((out/'validation_trials.json').read_text())
    selections = json.loads((out/'selected_hyperparameters.json').read_text())
    assert len(measured)==24 and len(trials)==24
    assert len({(r['method'],r['capacity'],r['seed'],r['order']) for r in measured})==24
    for key,config in selections.items():
        method,capacity = key.rsplit('_',1)
        best = max((r for r in trials if r['method']==method and r['capacity']==int(capacity)),key=lambda r:r['validation_accuracy'])
        assert config==best['config']
    all_rows = json.loads((args.prior/'summary_values.json').read_text())
    for method,label in [('derpp','DER++'),('er_ace','ER-ACE')]:
        for capacity in [200,1000]:
            rows = [r for r in measured if r['method']==method and r['capacity']==capacity]
            vals = [st.mean(100*r['scores']['accuracy'] for r in rows if r['seed']==s) for s in [101,102,103]]
            all_rows.append({'label':f'{label}: {capacity:,} images','accuracy':st.mean(vals),'sd':st.stdev(vals),
                             'mb':rows[0]['persistent_numeric_bytes']/1e6,'kind':method,'capacity':capacity,
                             'buffer_mb':rows[0]['buffer_bytes']/1e6,'optimizer_mb':rows[0]['optimizer_state_bytes']/1e6})
    (out/'all_summary_values.json').write_text(json.dumps(all_rows,indent=2))
    included = [r for r in all_rows if r['kind'] not in ('centroid','offline')]
    selected_frontier = frontier(included)
    complete_frontier = frontier([r for r in all_rows if r['kind']!='offline'])
    joint = next(r for r in all_rows if r['kind']=='offline')
    payload = {'scope':'Learned neural continual-learning methods only; class-mean controls excluded by requested scope. Joint offline reference not on frontier.',
               'included':included,'selected_frontier':selected_frontier,
               'excluded_centroid_controls':[r for r in all_rows if r['kind']=='centroid'],
               'complete_continual_frontier_including_centroids':complete_frontier,'joint_offline_reference':joint,
               'definition':'Exact retained numeric bytes and measured mean accuracy; not a confidence-aware dominance test. Tiny bookkeeping differences can affect strict dominance.'}
    (out/'pareto_data.json').write_text(json.dumps(payload,indent=2))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,(ax,zoom) = plt.subplots(1,2,figsize=(13.5,6.9),gridspec_kw={'width_ratios':[1.42,1]})
    fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.79,wspace=.17)

    def style(row):
        if row['label']=='Branch + auxiliary loss':
            return '#1767a0','*',230
        if row['label']=='Branch CE-only':
            return '#7ba5c1','D',55
        if row['kind']=='derpp':
            return '#8653a6','P',85
        if row['kind']=='er_ace':
            return '#278a7c','^',85
        if row['kind']=='replay':
            return '#bc8638','o',65
        return '#737c85','o',45

    def panel(axis):
        axis.plot([r['mb'] for r in selected_frontier],[r['accuracy'] for r in selected_frontier],
                  color='#485a51',ls=(0,(4,3)),lw=1.5,zorder=1)
        for row in included:
            color,marker,size = style(row)
            axis.errorbar(row['mb'],row['accuracy'],yerr=row['sd'],fmt='none',ecolor=color,elinewidth=1,capsize=2.5,zorder=3)
            axis.scatter(row['mb'],row['accuracy'],s=size,c=color,marker=marker,edgecolors='white',linewidths=.7,zorder=5)
        axis.axhline(joint['accuracy'],color='#a5a9af',ls=':',lw=1,zorder=0)
        axis.grid(color='#e8ebed',lw=.7)
        axis.set_axisbelow(True)
        axis.spines[['top','right']].set_visible(False)
        axis.set_xlabel('Retained continual-learning state (MB) ↓')

    panel(ax)
    panel(zoom)
    ax.set(xlim=(1.8,9.3),ylim=(0,102),ylabel='Final ten-class accuracy (%) ↑',title='Selected neural baselines')
    zoom.set(xlim=(2.08,3.04),ylim=(65,101),title='Low-memory trade-off: enlarged view')
    lookup = {r['label']:r for r in included}

    def label(axis,key,text,offset,align='left',bold=False):
        row = lookup[key]
        color,_,_ = style(row)
        axis.annotate(text,(row['mb'],row['accuracy']),xytext=offset,textcoords='offset points',
                      ha=align,va='center',fontsize=9,color=color,fontweight='bold' if bold else 'normal',
                      bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1.2},
                      arrowprops={'arrowstyle':'-','color':color,'lw':.7,'shrinkA':3,'shrinkB':5})

    label(ax,'Branch + auxiliary loss','Our branch model\n74.76% · 2.136 MB',(95,-10),bold=True)
    label(ax,'DER++: 1,000 images','DER++ · 1,000\n94.52% · 2.951 MB',(100,-2))
    label(ax,'Fine-tuning','Fine-tuning\n20.31%',(8,-26))
    label(ax,'Numenta (5% activity)','Numenta (5% activity)\n18.57%',(12,22))
    label(ax,'Online EWC','Online EWC\n23.38%',(-12,21),align='right')
    label(ax,'Synaptic Intelligence','SI\n20.34%',(-8,-23),align='right')
    ax.text(4.6,joint['accuracy']+1,'Joint offline reference: 96.97%',color='#7d838a',fontsize=8.4)

    label(zoom,'Branch + auxiliary loss','Our model\n74.76%',(20,0),bold=True)
    label(zoom,'Branch CE-only','Branch CE-only\n72.11%',(20,-25))
    label(zoom,'Replay: 200 images','ER 200\n81.59%',(15,-18))
    label(zoom,'ER-ACE: 200 images','ER-ACE 200\n85.72%',(15,0))
    label(zoom,'DER++: 200 images','DER++ 200\n88.44%',(15,18))
    label(zoom,'Replay: 1,000 images','ER 1,000\n90.55%',(-12,-30),align='right')
    label(zoom,'ER-ACE: 1,000 images','ER-ACE 1,000\n91.13%',(-12,13),align='right')
    label(zoom,'DER++: 1,000 images','DER++ 1,000\n94.52%',(-12,25),align='right')

    handles = [Line2D([0],[0],color='#485a51',ls='--',label='Mean frontier in selected set'),
               Line2D([0],[0],marker='*',color='none',markerfacecolor='#1767a0',markersize=11,label='Branch + auxiliary loss'),
               Line2D([0],[0],marker='o',color='none',markerfacecolor='#bc8638',markersize=7,label='Plain experience replay (ER)'),
               Line2D([0],[0],marker='P',color='none',markerfacecolor='#8653a6',markersize=8,label='DER++'),
               Line2D([0],[0],marker='^',color='none',markerfacecolor='#278a7c',markersize=8,label='ER-ACE')]
    ax.legend(handles=handles,loc='center right',bbox_to_anchor=(1.,.60),frameon=False,fontsize=8.4)
    fig.suptitle('Accuracy–memory comparison with strong replay methods',x=.065,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.065,.895,'DER++ and ER-ACE added · Split MNIST · same current-data exposure and ~529k-parameter backbone',fontsize=10.5,color='#515963')
    fig.text(.065,.115,'Selected neural methods only: class-mean controls excluded; joint training is an offline reference. This is not a complete latest-SOTA leaderboard.',fontsize=8.8,color='#4f5760')
    fig.text(.065,.08,'Memory includes replay images, labels and stored logits; excludes Adam, gradients, activations, dataset caches and Python overhead.',fontsize=8.8,color='#4f5760')
    fig.text(.065,.045,'Bars: SD across 3 seed means (2 task orders each). Replay methods have extra gradient exposures. Frontier segments are visual guides between measured means.',fontsize=8.4,color='#4f5760')
    fig.savefig(out/'pareto_strong_replay.png',dpi=200,facecolor='white')
    fig.savefig(out/'pareto_strong_replay.svg',facecolor='white')
    plt.close(fig)

    lines = ['# Extended comparison with strong replay methods','',
             'The new methods use the same cached split, normalized inputs, approximate parameter budget, current-data schedule, Adam optimizer, '
             'clip norm and evaluation seeds/orders as the earlier experiments. DER++ stores pre-update logits; ER-ACE uses the official cosine head and task-free training option. '
             'These are strong established comparators adapted to the common protocol, not a full latest-SOTA survey or original-paper-score reproduction.', '',
             '| Method | Mean accuracy ± seed SD | Retained numeric state (MB) |', '|---|---:|---:|']
    names = ['Branch + auxiliary loss','Branch CE-only','Replay: 200 images','ER-ACE: 200 images','DER++: 200 images',
             'Replay: 1,000 images','ER-ACE: 1,000 images','DER++: 1,000 images']
    for name in names:
        row = lookup[name]
        lines.append(f'| {name} | {row["accuracy"]:.2f} ± {row["sd"]:.2f}% | {row["mb"]:.6f} |')
    branch = lookup['Branch + auxiliary loss']
    dark = lookup['DER++: 200 images']
    lines += ['', f'DER++ with 200 stored images improves accuracy over the branch model by {dark["accuracy"]-branch["accuracy"]:.2f} percentage points '
              f'while retaining {dark["mb"]-branch["mb"]:.6f} MB more numeric state ({100*(dark["mb"]/branch["mb"]-1):.2f}% more). '
              'It also uses stored training images, stored logits and two replay passes per current update.', '',
              'The branch model remains nondominated within this selected neural-only two-objective set because it uses less retained state than the replay methods. '
              'This is a budget trade-off, not evidence of a general accuracy win. '
              'Strict frontier membership can depend on small bookkeeping differences; uncertainty bars do not establish significant dominance.', '',
              'Centroid controls remain excluded from this plot as previously requested. Including raw class means (82.32%, 0.03144 MB) still dominates the branch model. '
              'The machine-readable data records both the selected frontier and the complete measured continual-learning frontier including centroids. '
              'Joint training remains an offline reference.', '',
              'Verification: nine focused tests passed, including actual official observe-function update parity, class-order remapping, '
              'logit-memory accounting, replay membership, empirical Fisher and SI checks. '
              'All 24 new final runs and 24 validation trials are unique and selected configurations match the saved validation maxima. '
              'Official test evaluation verifies unchanged model state.', '',
              'See REPORT.md for settings and deviations, protocol.json for the full validation grid, upstream.json and official source snapshots for provenance, '
              'and results.json for exact numeric state, optimizer state, exposure counts and times. '
              'Values describe retained numeric state for continued learning, not peak RAM; optimizer state is additional.', '',
              '![Extended Pareto plot](pareto_strong_replay.png)', '',
              '[Vector SVG](pareto_strong_replay.svg) · [Full exact comparison and frontiers](pareto_data.json)', '']
    (out/'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:14]))


if __name__=='__main__':
    main()
