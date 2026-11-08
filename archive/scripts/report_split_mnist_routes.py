"""Build publication-style figures from recorded Split MNIST route diagnostics."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAIRS = ['0–1', '2–3', '4–5', '6–7', '8–9']
INK, TEAL, GOLD, GRAY = '#233348', '#087e8b', '#b67b18', '#7b8593'


def read(folder):
    return json.loads((ROOT / 'results' / folder / 'results.json').read_text())


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='reports/split_mnist_route_retention_20261012/figures')
    args = p.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    runs = read('split_mnist_usefulness_hard_20261011')
    audit = read('split_mnist_route_audit_20261011')
    probes = read('split_mnist_retention_diagnostic_20261011')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'figure.titlesize': 14, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'svg.fonttype': 'none'})

    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, text, color='#edf2f4'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009',
                                   facecolor=color, edgecolor='none'))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
    def arrow(start, end, color=GRAY, dashed=False):
        ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle':'->',
                    'color':color, 'lw':1.5, 'linestyle':'--' if dashed else '-'})
    box(.015, .61, .15, .19, 'Image\n784 pixels')
    box(.22, .61, .23, .19, '24 neurons × 4 branches\n96 tanh responses')
    box(.51, .61, .19, .19, 'One branch per neuron\n24 selected responses', '#d9eef0')
    box(.76, .61, .225, .19, 'Shared linear classifier\n10 digit scores')
    for a, b in [((.165,.705),(.22,.705)),((.45,.705),(.51,.705)),((.70,.705),(.76,.705))]: arrow(a,b)
    box(.025,.22,.19,.18,'Training digit labels\n(no task IDs)')
    box(.285,.22,.225,.18,'Counterfactual errors\nreplace one neuron value')
    box(.585,.22,.19,.18,'Recent usefulness EMA\n24 × 4 scores', '#d9eef0')
    arrow((.215,.31),(.285,.31)); arrow((.51,.31),(.585,.31))
    arrow((.875,.61),(.875,.46)); arrow((.875,.46),(.398,.46)); arrow((.398,.46),(.398,.40))
    ax.text(.77,.485,'Predictions, then feedback',ha='center',fontsize=9)
    arrow((.68,.40),(.61,.61),TEAL)
    box(.805,.055,.18,.19,'External archive\n5 saved configurations', '#f5ead7')
    arrow((.895,.25),(.72,.55),GOLD,True); arrow((.72,.55),(.61,.61),GOLD,True)
    ax.text(.70,.055,'Oracle audit only:\ntrue pair chooses saved route',ha='center',va='bottom',color=GOLD,fontsize=9)
    ax.text(.5,.94,'Architecture and information flow',ha='center',fontsize=14)
    ax.text(.5,.865,'During training, all branch and classifier weights remain plastic.',ha='center',fontsize=10)
    ax.text(.02,.065,'Solid: ordinary computation / training feedback\nDashed: privileged diagnostic intervention',fontsize=9,va='bottom')
    save(fig,out,'01_architecture')

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), layout='constrained')
    for arm,label,color,style in [('mlp','MLP',GRAY,'--'),('local','Local branch memory',TEAL,'-')]:
        rs=[r for r in runs if r['mode']=='sequential' and r['arm']==arm]
        values=np.array([[s['seen_accuracy']*100 for s in r['stages']] for r in rs])
        axes[0].plot(np.arange(1,6),values.mean(0),style,marker='o',color=color,label=label)
        axes[0].fill_between(np.arange(1,6),values.min(0),values.max(0),color=color,alpha=.14)
    axes[0].set(xticks=range(1,6),xticklabels=PAIRS,ylim=(0,105),
                xlabel='Latest digit pair learned',ylabel='Accuracy on all learned digits (%)',
                title='A. Ordinary evaluation: severe forgetting')
    axes[0].grid(axis='y',alpha=.15); axes[0].legend(frameon=False)
    labels=['Uniform\naveraging','Shared\nusefulness','Local\nusefulness']
    for offset,key,label,color in [(-.18,'original_accuracy','Ordinary: no task ID',GRAY),
                                  (.18,'oracle_saved_route_accuracy','Oracle: correct saved route',TEAL)]:
        means=[]; sds=[]
        for arm in ('uniform','shared','local'):
            v=[r[key]*100 for r in audit if r['arm']==arm]
            means.append(statistics.mean(v)); sds.append(statistics.stdev(v))
        x=np.arange(3)+offset
        axes[1].bar(x,means,width=.32,color=color,label=label,yerr=sds,capsize=3,alpha=.82)
        for i,(a,b) in enumerate(zip(x,means)):
            axes[1].text(a,b+sds[i]+2.2,f'{b:.1f}%',ha='center',fontsize=9)
    axes[1].set(xticks=range(3),xticklabels=labels,ylim=(0,113),
                ylabel='Final accuracy across all ten digits (%)',
                title='B. Same final weights; route restored')
    axes[1].legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(0,1.02))
    axes[1].grid(axis='y',alpha=.15); axes[1].set_axisbelow(True)
    fig.suptitle('Three seeds · oracle retrieval is not a task-ID-free result')
    save(fig,out,'02_accuracy_and_retrieval')

    fig,axes=plt.subplots(1,2,figsize=(10.8,4.8),layout='constrained')
    for ax,arm,title in zip(axes,('shared','local'),('Shared gate: limited global routes','Local gates: task-associated configurations')):
        rs=[r for r in audit if r['arm']==arm]
        matrix=np.mean([r['route_by_test_pair_accuracy'] for r in rs],axis=0)*100
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues')
        ax.set(xticks=range(5),xticklabels=PAIRS,yticks=range(5),yticklabels=PAIRS,
               xlabel='Test digit pair',ylabel='Route saved after learning pair',title=title)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=9,
                        color='white' if matrix[i,j]>55 else INK)
    fig.colorbar(im,ax=axes,label='Accuracy with fixed saved route (%)',shrink=.8)
    fig.suptitle('All ten outputs compete · final classifier unchanged · no output mask')
    save(fig,out,'03_route_cross_accuracy')

    keys=[('original','Original classifier'),('final_features_fresh_readout','Fresh classifier: current features'),
          ('random_features_fresh_readout','Fresh classifier: random features'),
          ('joint_features_fresh_readout','Fresh classifier: joint-trained features')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,arm,title in zip(axes,('mlp','local'),('MLP: 95-dimensional features','Branch model: 24-dimensional features')):
        choices=list(keys)
        if arm=='local':choices += [('saved_routes_original_readout_oracle','Original classifier: oracle saved routes')]
        rs=[r for r in probes if r['arm']==arm]
        vals=[statistics.mean(r[k]['all_accuracy'] for r in rs)*100 for k,_ in choices]
        names=[name for _,name in choices]
        colors=[TEAL if k=='original' else GOLD if 'oracle' in k else GRAY for k,_ in choices]
        ax.barh(np.arange(len(vals)),vals,color=colors,alpha=.85)
        for i,value in enumerate(vals):ax.text(value+1,i,f'{value:.1f}%',va='center',fontsize=9)
        ax.set(yticks=range(len(vals)),yticklabels=names,xlim=(0,111),
               xlabel='All-digit test accuracy (%)',title=title)
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Fresh-classifier probes use all historical training labels; oracle route uses true pair')
    save(fig,out,'04_readout_diagnostics')
    manifest={'source_results': ['results/split_mnist_usefulness_hard_20261011/results.json',
               'results/split_mnist_route_audit_20261011/results.json',
               'results/split_mnist_retention_diagnostic_20261011/results.json'],
              'seeds':[34,35,36],'figures':sorted(p.name for p in out.iterdir() if p.suffix in ('.png','.svg')),
              'aggregation':'Arithmetic seed means; Figure 2 whiskers are sample SD across seeds, not confidence intervals; heatmaps are seed means of per-pair accuracy; ordinary learning curve shades seed min/max.'}
    (out.parent/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(out)


if __name__=='__main__':main()
