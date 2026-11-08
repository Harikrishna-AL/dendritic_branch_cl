"""Fixed reporting and scientific plots for the preregistered stress conditions."""
import json,statistics,random,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/bdh_stress_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/predictive_memory_stress_20261011')
VARIANTS=('baseline','linear','neuron_shared','history_only')
LABELS=('Branch original','Branch linear','Shared context','History only')
COLORS=('#3977ac','#27906c','#af7541','#a35483')

def main():
    rows=json.loads((OUT/'results.json').read_text());assert len(rows)==40
    conditions=list(rows[0]['conditions']);paired={};rng=random.Random(79)
    for c in conditions:
        paired[c]={}
        for other in ('baseline','neuron_shared','history_only'):
            differences=[]
            for seed in range(3,8):
                def avg(v):return statistics.mean(r['conditions'][c]['all_mse'] for r in rows if r['variant']==v and r['seed']==seed)
                differences.append(avg(other)-avg('linear'))
            boot=sorted(statistics.mean(rng.choices(differences,k=5)) for _ in range(10000))
            paired[c][other]={'other_minus_linear_per_seed':differences,'mean_gain':statistics.mean(differences),'linear_seed_wins':sum(x>0 for x in differences),'seed_bootstrap95':[boot[250],boot[9749]]}
    summary=OUT/'summary.md'
    summary.write_text(summary.read_text().replace('|---|---|---:|---:|---:|---:|---:|','|---|---|---:|---:|---:|---:|'))
    (OUT/'paired.json').write_text(json.dumps(paired,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for v,label,col in zip(VARIANTS,LABELS,COLORS):
        ys=[statistics.mean(r['conditions'][c]['all_mse']-r['conditions'][c]['oracle_mse'] for r in rows if r['variant']==v) for c in conditions]
        axes[0].plot(range(len(conditions)),ys,'o-',label=label,color=col,linewidth=1.6,markersize=4)
        for ax,c in zip(axes[1:],('reference','gradual128')):
            rs=[r['conditions'][c] for r in rows if r['variant']==v]
            values=[statistics.mean(p['binned_mse'][j] for r in rs for p in r['phases'][1:]) for j in range(16)]
            ax.plot([8+16*j for j in range(16)],values,label=label,color=col,linewidth=1.6)
    axes[0].set_xticks(range(len(conditions)),['64 steps','256','1024','Ramp32','Ramp128','Gap .4','Gap .1'],rotation=40,ha='right')
    axes[0].set_title('Whole stream, including transitions');axes[0].set_ylabel('MSE above latent-state oracle')
    for ax,title in zip(axes[1:],('Abrupt switches','128-step gradual switches')):
        ax.set_title(title);ax.set_xlabel('Observations after switch');ax.set_ylabel('Prediction MSE (16-step bins)')
    axes[2].axvline(128,color='gray',linestyle=':',alpha=.6)
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Locked models: five seeds, both training orders; weights frozen',fontsize=12)
    fig.savefig(OUT/'stress.png',dpi=170);fig.savefig(OUT/'stress.pdf');plt.close(fig)
    lines=['# Stress-test interpretation','','All seven conditions and settings were fixed before evaluation. Five trained seeds, both orders; averages count every observation. This tests inference adaptation and retained weights, not learning under altered training schedules.','','| Condition | Linear vs original: gain | Linear seed wins | Linear vs history: gain | Linear seed wins |','|---|---:|---:|---:|---:|']
    for c in conditions:
        a=paired[c]['baseline'];b=paired[c]['history_only'];lines.append(f'| {c} | {a["mean_gain"]:.3f} | {a["linear_seed_wins"]}/5 | {b["mean_gain"]:.3f} | {b["linear_seed_wins"]}/5 |')
    lines+=['','Positive gain means lower error for the linear branch model. Five seeds give limited uncertainty estimates; intervals and every paired seed are in paired.json.','', 'See summary.md for oracle errors, first32 switch errors, old-task revisits and route overlap/churn. Reduced recruitment overlap alone does not prove useful task separation. Gradual switches interpolate the prediction target too. The output is still a hard winner, which can restrict prediction of intermediate mixed targets.','', 'The oracle changes across ambiguity conditions because the intrinsic prediction noise changes. The above-oracle plot helps interpret this, but it does not normalize every possible difficulty difference. No tested condition supplies its identity to the network.','', f'![Stress tests]({(OUT / "stress.png").resolve()})']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))
if __name__=='__main__':main()
