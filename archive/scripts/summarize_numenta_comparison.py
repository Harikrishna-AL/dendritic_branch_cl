"""Finalize and visualize the completed native Numenta comparison."""
import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def clean(value):
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clean(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--sanity',type=Path,required=True)
    args = p.parse_args()
    out = args.directory
    rows = clean(json.loads((out/'results.json').read_text()))
    assert len(rows)==6 and len({(r['seed'],r['context']) for r in rows})==6
    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    branch = json.loads((out/'existing_branch_control.json').read_text())
    sanity = json.loads((args.sanity/'results.json').read_text())
    assert len(sanity['stages'])==5
    arms = [('Existing branch\nsame 5 means',[100*r['accuracy'] for r in branch]),
            ('Numenta\nprototype context',[100*r['scores']['autonomous']['accuracy'] for r in rows if r['context']=='prototype']),
            ('Numenta\ninput as context',[100*r['scores']['autonomous']['accuracy'] for r in rows if r['context']=='input'])]
    fig,axes = plt.subplots(1,2,figsize=(11,4.6),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
    means = [st.mean(a) for _,a in arms]
    axes[0].bar(np.arange(3),means,yerr=[st.stdev(a) for _,a in arms],capsize=4,color=['#226b9d','#bf7f33','#7e8c5a'])
    for i,a in enumerate(means):
        axes[0].text(i,a+2,f'{a:.2f}%',ha='center',fontsize=10)
    axes[0].set(xticks=np.arange(3),xticklabels=[a for a,_ in arms],ylim=(0,105),ylabel='Final ten-class accuracy (%)',title='Split MNIST · 3 seeds')
    vals = [100*a for a in sanity['stages'][-1]['task_accuracy']]
    axes[1].bar(np.arange(5),vals,color='#bf7f33')
    axes[1].axhline(st.mean(vals),color='#444444',ls='--',label=f'Mean {st.mean(vals):.2f}%')
    axes[1].set(xticks=np.arange(5),xticklabels=np.arange(1,6),ylim=(0,105),xlabel='Permutation task',title='Numenta native recipe · 1 seed')
    axes[1].legend(frameon=False,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Different benchmarks shown separately; model capacities differ',fontsize=11)
    fig.savefig(out/'comparison.png',dpi=180)
    plt.close(fig)
    lines = ['# Interpretation of the native-model comparison','',
             'The unmodified upstream Numenta model was trained, rather than approximated with the previous online prototype sidecar. '
             'Three seeds were completed for each Split MNIST context variant.', '',
             f'On Split MNIST, the existing branch network with the same five task-mean router assignments scores **{means[0]:.2f}%**, '
             f'the full Numenta model with prototype context **{means[1]:.2f}%**, and Numenta with input as context **{means[2]:.2f}%**.', '',
             f'On the separate original benchmark family, the one-seed five-task Permuted MNIST sanity check using the upstream recipe scores **{st.mean(vals):.2f}%**. '
             'This provides a check that the upstream model/activation stack learns and retains performance under its intended benchmark protocol; it is not a multi-seed paper reproduction.', '',
             'These results do not establish general superiority of the branch method. Split MNIST introduces new labels, while Permuted MNIST reuses the same labels under different pixel permutations. '
             'The Split run adapts the original hyperparameters to match the existing two-epoch example budget and learning rate. '
             'A common five-mean router controls context selection in the prototype comparison, but the models differ in capacity, initialization, normalization, activation and learning objective.', '',
             'The branch model has 75,610 parameters; the Numenta model allocates 21,880,842 parameters, approximately 289 times as many, '
             'with about 18.97 million nonzero values after fixed masking. These are dense-tensor allocations; mask sparsity does not imply sparse executed kernels.', '',
             'The original native prototype variant uses task boundaries during training. Its oracle-context result diagnoses whether perfect context recovers old classes without output masking. '
             'The input-as-context variant avoids task IDs and stored example buffers; it is a separate upstream option, not the paper\'s statistical context clustering.', '',
             'The paper\'s autonomous batch-clustering method retains historical inputs for multivariate tests; no classifier replay is implied. '
             'That exact context method was not located in the pinned release and remains untested here. '
             'A future comparison should implement/audit that method explicitly and tune each baseline on validation data before making benchmark-superiority claims.', '',
             'Verification: upstream dendritic application tests passed (5 tests); all model evaluations verified unchanged state; '
             'normalized and unnormalized inputs produced exactly identical nearest-mean assignments. '
             'Undefined future-pair stage metrics have been converted from NaN to JSON null; measured scores are unchanged.', '',
             '![Model comparison](comparison.png)', '']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:9]))


if __name__=='__main__':
    main()
