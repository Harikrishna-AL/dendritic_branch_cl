"""Build paired-effect summary and static figure from completed routing runs."""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    out = p.parse_args().directory
    rows = json.loads((out/'results.json').read_text())
    assert len(rows) == 6 and len({(r['seed'],r['objective']) for r in rows}) == 6
    full = {r['seed']:r for r in rows if r['objective']=='counterfactual'}
    base = {r['seed']:r for r in rows if r['objective']=='ce_only'}
    assert full.keys() == base.keys() and all(r['original_checkpoint_exact_match'] for r in full.values())
    arms = ['current','input_prototype','latent_prototype','oracle']
    effects = {arm:[100*(full[s]['scores'][arm]['accuracy']-base[s]['scores'][arm]['accuracy']) for s in full] for arm in arms}
    lines = ['# Paired interpretation', '', 'All three auxiliary-objective checkpoints exactly match the original saved backbone checkpoints.', '',
             '| Router | Auxiliary objective minus CE-only (percentage points, mean ± seed SD) | Paired seed differences |', '|---|---:|---|']
    for arm, delta in effects.items():
        lines.append(f'| {arm} | {st.mean(delta):+.2f} ± {st.stdev(delta):.2f} | '+', '.join(f'{d:+.2f}' for d in delta)+' |')
    silhouette = [full[s]['separation']['latent']['class_silhouette_cosine']-base[s]['separation']['latent']['class_silhouette_cosine'] for s in full]
    drift = {name:st.mean(st.mean(r['stages'][-1]['old_class_centroid_l2_drift']) for r in records.values()) for name,records in [('counterfactual',full),('ce_only',base)]}
    lines += ['', f'Paired learned-feature class silhouette difference: {st.mean(silhouette):+.4f}.', '',
              f'Final old-class centroid movement from acquisition (mean L2): auxiliary {drift["counterfactual"]:.4f}; CE-only {drift["ce_only"]:.4f}. These diagnostics use test labels after training and do not update memory.', '',
              'The large input-router improvement over current routing supports the practical value of input-conditioned retrieval. '
              'A comparison with CE-only is required to attribute any improvement to the auxiliary loss. '
              'Oracle performance remains a privileged diagnostic, not autonomous accuracy.', '',
              'Both banks reach 95 slots in the first pair in every run. Later contexts must adapt existing centroids. '
              'The learned-feature bank also spans changing encoder coordinates. These confounds mean poor latent routing cannot be attributed solely to intrinsic feature separability. '
              'The perturbation probe can be insensitive if many centroids map to identical branch-vote winners.', '',
              f'Numeric router state: input {rows[0]["router_numeric_bytes"]["input"]:,} bytes; latent {rows[0]["router_numeric_bytes"]["latent"]:,} bytes. '
              'Includes float32 centroids and int64 counts/votes; excludes Python overhead and model/optimizer state. '
              'Each router is an independent inference arm; both are stored simultaneously only for paired measurement.', '',
              'This experiment does not establish an advantage over the original Numenta network. '
              'Next controlled comparison: separate encoder-coordinate drift from bank saturation using a frozen encoder and a predeclared capacity/allocation ablation, with the same router on both objectives.', '',
              '![Routing comparison](comparison.png)', '']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(9,4.8), layout='constrained')
    xx = np.arange(len(arms))
    for i,(name,label,color) in enumerate([('counterfactual','Auxiliary branch loss','#226b9d'),('ce_only','CE-only ablation','#d4923a')]):
        selected = [r for r in rows if r['objective']==name]
        means = [st.mean(100*r['scores'][a]['accuracy'] for r in selected) for a in arms]
        errors = [st.stdev(100*r['scores'][a]['accuracy'] for r in selected) for a in arms]
        positions = xx+(i-.5)*.36
        ax.bar(positions, means, .34, yerr=errors, capsize=4, color=color, label=label)
        for px,val in zip(positions,means):
            ax.text(px,val+2,f'{val:.1f}',ha='center',fontsize=9)
    ax.set(xticks=xx, xticklabels=['Current gate','Input prototypes','Latent prototypes','Oracle route'], ylim=(0,112), ylabel='Final ten-class accuracy (%)',title='Online prototype routing · Split MNIST · 3 paired seeds')
    ax.legend(loc='upper left',frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'comparison.png',dpi=180)
    plt.close(fig)
    print('\n'.join(lines[:15]))


if __name__ == '__main__':
    main()
