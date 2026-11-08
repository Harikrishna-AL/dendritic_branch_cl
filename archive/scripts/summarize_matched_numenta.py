"""Summarize fixed-size primary results and exploratory activity controls."""
import argparse
import hashlib
import json
import statistics as st
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def mean_seed_values(rows,mode='inferred'):
    return [st.mean(100*r['scores'][mode]['accuracy'] for r in rows if r['seed']==seed)
            for seed in sorted({r['seed'] for r in rows})]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--sensitivity',type=Path,required=True)
    args = p.parse_args()
    rows = json.loads((args.base/'results.json').read_text())
    sensitivity = json.loads((args.sensitivity/'results.json').read_text())
    assert len(rows)==18 and len(sensitivity)==12
    assert len({(r['arm'],r['seed'],r['order']) for r in rows})==18
    assert len({(r['setting'],r['seed'],r['order']) for r in sensitivity})==12
    groups = [(label,[r for r in rows if r['arm']==arm]) for label,arm in
              [('Branch + auxiliary loss','branch_aux'),('Branch CE-only','branch_ce'),('Numenta 5% activity','numenta')]]
    groups += [(label,[r for r in sensitivity if r['setting']==setting]) for label,setting in
               [('Numenta 20% activity (exploratory)','kw20'),('Numenta ReLU (exploratory)','relu')]]
    lines = ['# Matched-size results and fairness assessment','',
             'All models use approximately 529,000 unconstrained learnable coefficients, identical normalized inputs, '
             '55,000 training images, two epochs per pair, batch 128, Adam without task resets, gradient clip 5, '
             'common task-mean context assignments, three independent learning-rate trials per configuration, '
             'and three evaluation seeds in both forward and reverse order. No replay, SI, test IDs or label masks.', '',
             '| Model | Final inferred-context accuracy ± seed SD | Oracle-context accuracy |', '|---|---:|---:|']
    for label,selected in groups:
        vals = mean_seed_values(selected)
        oracle = st.mean(mean_seed_values(selected,'oracle'))
        lines.append(f'| {label} | {st.mean(vals):.2f} ± {st.stdev(vals):.2f}% | {oracle:.2f}% |')
    lines += ['', 'Each SD is computed over three seed means after averaging both orders within seed, not over six independent runs. '
              'Numenta activity alternatives were added after inspecting the initial run and validation acquisition instability; '
              'they are exploratory controls and reported separately, with their own three-trial validation selections.', '',
              '## Resource matching', '',
              '| Model | Free coefficients | Allocated parameters | Persistent numeric bytes | Training seconds | Batch-256 inference ms |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ['numenta','branch_aux','branch_ce']:
        selected = [r for r in rows if r['arm']==arm]
        r = selected[0]
        lines.append(f'| {arm} | {r["free_coefficients"]:,} | {r["allocated_parameters"]:,} | {r["persistent_numeric_bytes"]:,} | '
                     f'{st.mean(x["training_seconds_excluding_validation"] for x in selected):.2f} | '
                     f'{st.mean(x["inference_batch256_median_ms"] for x in selected):.3f} |')
    delta = []
    for seed in sorted({r['seed'] for r in rows}):
        aux = st.mean(100*r['scores']['inferred']['accuracy'] for r in rows if r['arm']=='branch_aux' and r['seed']==seed)
        ce = st.mean(100*r['scores']['inferred']['accuracy'] for r in rows if r['arm']=='branch_ce' and r['seed']==seed)
        delta.append(aux-ce)
    half = 4.30265273*st.stdev(delta)/np.sqrt(3)
    lines += ['', 'Free coefficients count all parameters except permanently masked zeros. Allocated tensors differ by 5.2%; '
              'Numenta also stores mask buffers. Persistent bytes include the shared 15,680-byte prototype bank and, for branch arms, '
              'error traces and a 5×168-byte route archive. Counts exclude optimizer, Python overhead, temporary activations and checkpoint file metadata. '
              'Optimizer state is separately logged. Timing is CPU wall time, not FLOPs or an M4 benchmark; training times exclude prototype preparation, validation and saving.', '',
              '## What the result supports', '',
              'Under this fixed coefficient and exposure budget, the branch models retain substantially higher Split MNIST accuracy than these Numenta variants. '
              'Higher native activity and ReLU controls help check that the result is not solely an artifact of retaining only three winners per layer.', '',
              f'The auxiliary-loss arm improves over the matched CE-only branch arm by **{st.mean(delta):+.2f} percentage points** on average. '
              f'Paired seed differences are {", ".join(f"{v:+.2f}" for v in delta)} pp. '
              f'A descriptive paired t interval with only three seeds is [{st.mean(delta)-half:+.2f}, {st.mean(delta)+half:+.2f}] pp (95%). '
              'This small cohort does not establish a reliable auxiliary-loss advantage.', '',
              '## Remaining fairness limits', '',
              'This is one resource budget and a small validation search. Depth, initialization, activation sparsity and architectural mechanisms differ. '
              'Equal free coefficients do not mean equal active capacity, memory, compute or tuning FLOPs. '
              'The prototype protocol permits known training boundaries and task-associated route archives; it is not autonomous context acquisition. '
              'Numenta\'s SI extension and statistical batch-clustering variant remain outside this comparison. '
              'More budgets, independent task orders, validation of method-specific settings and harder datasets are needed for a broad superiority claim. '
              'A longer convergence-matched schedule is also needed to separate learning speed from eventual performance.', '',
              '## Reproduce', '',
              'Primary: `python -m bdh_cl.scripts.matched_numenta_comparison --out NEW_PRIMARY`.', '',
              'Activity controls: `python -m bdh_cl.scripts.matched_numenta_activity_sensitivity --base NEW_PRIMARY --out NEW_ACTIVITY`.', '',
              'Summary: `python -m bdh_cl.scripts.summarize_matched_numenta --base NEW_PRIMARY --sensitivity NEW_ACTIVITY`.', '',
              'All final evaluations verified unchanged model tensors and route archives. Validation selections were saved before evaluating official test images in each experiment. '
              'The primary and exploratory protocols and results remain separate on disk.', '',
              '![Matched-size comparison](comparison.png)', '']
    (args.base/'SUMMARY.md').write_text('\n'.join(lines))
    fig,ax = plt.subplots(figsize=(9,5),layout='constrained')
    vals = [mean_seed_values(selected) for _,selected in groups]
    ax.bar(np.arange(5),[st.mean(v) for v in vals],yerr=[st.stdev(v) for v in vals],capsize=4,
           color=['#226b9d','#74a4bf','#bd8238','#d3a367','#dacaaf'])
    for i,v in enumerate(vals):
        ax.text(i,st.mean(v)+2,f'{st.mean(v):.2f}%',ha='center',fontsize=10)
    ax.set(xticks=np.arange(5),xticklabels=['Branch\nauxiliary loss','Branch\nCE-only','Numenta\n5% activity','Numenta\n20% activity*','Numenta\nReLU*'],ylim=(0,100),
           ylabel='Final ten-class accuracy (%)',title='Split MNIST · ~529k free coefficients · 3 seeds × 2 orders')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.12,.93,'* Exploratory activity controls; error bars are SD across seed means',fontsize=9)
    fig.savefig(args.base/'comparison.png',dpi=180)
    plt.close(fig)
    root = Path(__file__).resolve().parents[2]
    provenance = {'revisions':{name:subprocess.check_output(['git','-C',str(root/'external'/name),'rev-parse','HEAD'],text=True).strip()
                               for name in ['htmpapers','nupic.research','nupic.torch']},
                  'source_sha256':{name:hashlib.sha256((root/'bdh_cl/scripts'/name).read_bytes()).hexdigest() for name in
                                   ['matched_numenta_comparison.py','matched_numenta_activity_sensitivity.py','split_mnist_usefulness.py','numenta_split_mnist_comparison.py']}}
    (args.base/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('\n'.join(lines[:16]))


if __name__=='__main__':
    main()
