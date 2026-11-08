"""Summarize matched branch-compartment recurrence runs and state ablations."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'bdh_cl/results/cluster'
out=base/'exp14_analysis';out.mkdir(exist_ok=True)
summary={}
for folder in ['exp14_compartment','exp14_compartment_domain']:
    files=sorted((base/folder).glob('seed*.json'))
    runs=[json.loads(p.read_text()) for p in files]
    assert len(runs)==3 and all(len(r['arms'])==4 for r in runs)
    stats={}
    for arm in ['none','shared','uniform','selective']:
        rr=[r['arms'][arm] for r in runs]
        values={'return_A':[r['retention_before_return'] for r in rr],
                'B_after_learning':[r['B_after_learning'] for r in rr],
                'mixed_final':[r['mixed_final_accuracy'] for r in rr]}
        if arm!='none':
            values['return_A_zero_memory']=[r['return_zero_memory'][0]['accuracy'] for r in rr]
            values['return_A_permuted_memory']=[r['return_permuted_memory'][0]['accuracy'] for r in rr]
        stats[arm]={k:{'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)),'per_seed':v} for k,v in values.items()}
    summary[folder]=stats
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Branch-compartment memory: completed three-seed tests', '',
'Both confirmation arrays completed successfully on H100s. Three seeds per setting; all arms use the same trainable dendritic BDH and memory arms have identical state budgets. Full protocol is ../exp14_compartment_pilot/PROTOCOL.md. All reported primary predictions compete across ten digit logits without task IDs.', '',
'## Main outcome', '',
'This bank-level selective-access implementation did not establish the proposed memory-retrieval benefit. On disjoint classes every arm forgot A completely. On same-label domains, selective access did not beat shared memory for pre-update A retention, and erasing or permuting the state did not reduce selective retention on average. Better retained accuracy than no memory therefore cannot be credited to correct retrieval from separate compartments.', '',
'## Accuracy immediately before A relearning', '',
'| Arm | Disjoint classes | Same-label domains | Same-label, state erased | Same-label, state permuted |', '|---|---:|---:|---:|---:|']
for arm in ['none','shared','uniform','selective']:
    c=summary['exp14_compartment'][arm]
    d=summary['exp14_compartment_domain'][arm]
    def fmt(key):
        if key not in d:return '—'
        v=d[key];return f'{v["mean"]:.1%} ± {v["sample_sd"]:.1%}'
    lines.append(f'| {arm} | {c["return_A"]["mean"]:.1%} | {fmt("return_A")} | {fmt("return_A_zero_memory")} | {fmt("return_A_permuted_memory")} |')
lines += ['', '± is sample standard deviation across three seeds, not a confidence interval.', '',
'## Acquisition and mixed evaluation', '',
'| Arm | B learned, disjoint | B learned, same-label | Final mixed same-label |', '|---|---:|---:|---:|']
for arm in ['none','shared','uniform','selective']:
    c=summary['exp14_compartment'][arm];d=summary['exp14_compartment_domain'][arm]
    lines.append(f'| {arm} | {c["B_after_learning"]["mean"]:.1%} | {d["B_after_learning"]["mean"]:.1%} | {d["mixed_final"]["mean"]:.1%} |')
lines += ['',
'The models learned B at roughly 97–98%; lack of retention is not simply failure to learn the new environment. Selective final mixed same-label accuracy was 78.5% versus 76.1% for no memory, but this small three-seed difference is a secondary result and does not demonstrate retrieval. Primary retention and state-ablation criteria were not met.', '',
'## Interpretation', '',
'In the class-incremental setting, the selective arm chose the same dominant bank for A and B in all 12 seed/head combinations at return. Differences in mean soft routing were small. These aggregate probes indicate weak context separation; they do not prove that all individual inputs or all BDH layers route identically.', '',
'In the same-label setting, shared memory retained 45.0% with its state intact and 49.3% after the state was erased. Selective memory retained 43.7% intact and 45.0% erased. The ablations hold learned weights fixed. Thus the stored state did not supply useful old-context retrieval at this operating point. A change in training trajectory may explain improvements over no memory, but that explanation remains untested.', '',
'## Scope and next decision', '',
'This prototype groups branch planes into four banks per attention head and uses learned image/head gates. It is not a fully neuron-local dendritic memory design. There is one compression scheme, read strength 0.1, decay 0.999, eta 0.01 and optimizer protocol. Changing encoder/key representations, inadequate gates, readout drift and bank interference remain possible causes; the experiment does not isolate each one.', '',
'The next experiment should test stable, input-dependent addressing before adding more memory or growth: can a boundary-free router consistently select different compartments for distinguishable recurring input contexts and recover the original compartment on return? Compare a stable sensory-address router with the present moving branch-feature router under the same state budget. This can keep the BDH computational backbone trainable; a fixed addressing path is a narrower intervention than freezing all BDH features. Do not claim a dendrite-specific benefit until it beats equally sized generic memory slots with the same routing policy.', '',
'## Verification and cluster accounting', '',
'Five local unit tests passed; full jobs also gated on these tests. No-memory forward matches native dendritic BDH exactly in a dropout-free test. Shared and uniform controls produced identical learning curves/results. Reads precede writes; evaluation and ablations do not write state. The same fixed permutation is applied to both train and test in the domain setting and does not change labels.', '',
'Pilot: job 4330542, about 2 minutes. Class array: 4330684, 4:43–4:49 per seed. Domain array: 4330763, 4:47–4:59 per seed. All exit codes zero. Combined pilot and confirmation GPU allocation elapsed time is approximately 31 minutes, excluding queue waits. Individual full jobs were capped at one GPU-hour.', '',
'Raw results: ../exp14_compartment/seed0.json–seed2.json and ../exp14_compartment_domain/seed0.json–seed2.json. Aggregates and per-seed values: summary.json. Full checkpoints remain on the cluster.', '']
(out/'REPORT.md').write_text('\n'.join(lines))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['none','shared','uniform','selective']
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(4)
    d=summary['exp14_compartment_domain']
    means=[d[n]['return_A']['mean']*100 for n in names]
    erased=[np.nan if n=='none' else d[n]['return_A_zero_memory']['mean']*100 for n in names]
    ax.bar(x-.18,means,.36,label='Intact memory',color='#3065ad')
    ax.bar(x+.18,erased,.36,label='Memory erased, same weights',color='#df9640')
    for i,n in enumerate(names):
        ax.scatter(np.full(3,x[i]-.18),np.array(d[n]['return_A']['per_seed'])*100,s=20,color='black',zorder=3)
        if n!='none':ax.scatter(np.full(3,x[i]+.18),np.array(d[n]['return_A_zero_memory']['per_seed'])*100,s=20,color='black',zorder=3)
    ax.set_xticks(x,['No memory','Shared','Uniform banks','Selective banks'])
    ax.set_ylabel('A accuracy before relearning (%)')
    ax.set_ylim(0,75)
    ax.set_title('Same-label domain recurrence: erasing state does not hurt retention')
    ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'return_memory_ablation.png',dpi=180);plt.close(fig)
except ImportError:
    pass
print(out/'REPORT.md')
