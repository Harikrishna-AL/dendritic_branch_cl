"""Combined retention evidence with learning controls and frozen-probe limitations."""
import json,statistics,random,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/bdh_retention_matplotlib')
ROOT=Path('results');OUT=ROOT/'electricity_retention_pilot_20261011'
def read(folder):return json.loads((ROOT/folder/'results.json').read_text())
def main():
    natural=read('electricity_retention_pilot_20261011');toy=read('usefulness_retention_synthetic_20261011')+read('usefulness_retention_shared_20261011');joint=read('usefulness_retention_joint_20261011')
    assert len(natural)==30 and len(toy)==48 and len(joint)==48
    variants=['predict_winner','quality_soft','quality_all','uniform_all','shared','history','mlp','gru']
    lines=['# Continual-learning retention checks','','## Natural electricity','','Five new seeds, older/newer heldouts. All models improve on the older period after newer learning, so this protocol shows backward transfer, not an anti-forgetting advantage. MLP older MSE0.1637→0.1553;soft usefulness0.1888→0.1650;uniform0.1889→0.1649. No heldout slow-weight training occurs.','','## Controlled task-ID-free regimes','','Three new seeds, both orders. Positive forgetting is older heldout errorafterB minuserrorafterA. Main scores include transitions. Audit weights are frozen; short-term memory can update causally, with one-step target feedback for quality variants.','','| Model | Old beforeB | Old afterB | Forgetting | New afterB | Joint average |','|---|---:|---:|---:|---:|---:|']
    metrics={}
    for v in variants:
        rs=[r for r in toy if r['variant']==v];js=[r for r in joint if r['variant']==v]
        def error(stage,old):return statistics.mean(p['all_mse'] for r in rs for p in r[stage] if (p['regime']==r['first'])==old)
        m={'old_before':error('after_A',True),'old_after':error('after_B',True),'forgetting':statistics.mean(r['forgetting_all_mse'] for r in rs),'new_after':error('after_B',False),
           'joint_average':statistics.mean(p['all_mse'] for r in js for p in r['joint']),
           'old_after_moreA':error('after_moreA',True),'first32_return':statistics.mean(r['return_carry'][0]['first32_mse'] for r in rs)}
        metrics[v]=m;lines.append('| '+v+' | '+' | '.join(f'{m[k]:.3f}' for k in ('old_before','old_after','forgetting','new_after','joint_average'))+' |')
    paired={};rng=random.Random(143)
    for candidate,control in [('quality_soft','predict_winner'),('quality_soft','uniform_all'),('quality_all','uniform_all')]:
        values=[]
        for seed in (8,9,10):
            def f(v):return statistics.mean(r['forgetting_all_mse'] for r in toy if r['seed']==seed and r['variant']==v)
            values.append(f(control)-f(candidate))
        boot=sorted(statistics.mean(rng.choices(values,k=3)) for _ in range(10000))
        paired[candidate+'_vs_'+control]={'forgetting_reduction_per_seed':values,'mean':statistics.mean(values),'seed_wins':sum(a>0 for a in values),'bootstrap95':[boot[250],boot[9749]],'caution':'only3initializations;orders grouped withinseed;engineeredtaskfamily'}
    (OUT/'combined_metrics.json').write_text(json.dumps({'controlled':metrics,'paired_forgetting':paired},indent=2))
    lines+=['','Joint control uses equal8192trainingsteps in alternating128stepregimes; it is not a tuned oracle optimum. Its lower baseline error shows that simultaneous exposure substantially alleviates the sequential failure, though fixed-window context/architecture also affects capacity. MoreA controls retain the same optimizer and provide equaladditional training budget; their results and return transients are in rawrecords.','', '## What can be claimed','','There is evidence of reduced forgetting for these memory/competition architectures in the controlled, task-ID-free stream. The usefulness variant additionally improves retention relative to the same all-output learning rule with uniform averaging. This does not establish real-data anti-forgetting benefits, dendrite-specific biological necessity, or superiority over all generic memory/mixture routers. The quality cue requires causal forecast feedback; it is not unlabelled target-free inference.','', 'Cold audit resets happen only in separate probes; trainingstate and optimizer persist through controlled task changes. ScoresafterB include both returningoldtask visits andnewtask visits, with no slowweight updates. Fast feedback adaptation may recover an existing readout; retainedslow knowledge and short-term contextual reconstruction are distinct. Full freezing of all memory would answer a different question.','', 'The natural electricity test has24h target delay; the controlled next-step task has1step feedback and was engineered around predictability. These differences limit transfer claims. Runtime memory counts differ; branch384slowparameters,MLP397,GRU361,shared384. Shared routing is itself branch-structured computation with common temporal statistics, not a unique-locality baseline.','', 'Four targeted checks passed: heldout causality and unchanged audit/live state, preserved pending feedback through switches, and matched local/shared slow initialization. No variant was selected using these retention outcomes.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    labels=['Original','Soft quality','Hard quality','Uniform','Shared','History','MLP','Window GRU']
    x=list(range(len(variants)))
    axes[0].bar([a-.18 for a in x],[metrics[v]['old_before'] for v in variants],width=.36,label='Before new learning',color='#849eb7')
    axes[0].bar([a+.18 for a in x],[metrics[v]['old_after'] for v in variants],width=.36,label='After new learning',color='#b45f56')
    axes[0].set_xticks(x,labels,rotation=35,ha='right');axes[0].set_ylabel('Old-task MSE');axes[0].set_title('Controlled regimes: weights frozen in probes');axes[0].legend(fontsize=8)
    nvariants=['predict_winner','quality_soft','quality_all','uniform_all','mlp','gru'];nx=list(range(len(nvariants)))
    for label,stage,color,offset in [('Before','after_A','#849eb7',-.18),('After','after_B','#38906b',.18)]:
        values=[statistics.mean(r[stage]['old']['mse'] for r in natural if r['variant']==v) for v in nvariants]
        axes[1].bar([a+offset for a in nx],values,width=.36,label=label,color=color)
    axes[1].set_xticks(nx,['Original','Soft quality','Hard quality','Uniform','MLP','GRU'],rotation=35,ha='right');axes[1].set_ylabel('Older electricity MSE');axes[1].set_title('Natural years: backward transfer for all');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'retention.png',dpi=170);fig.savefig(OUT/'retention.pdf');plt.close(fig)
    lines+=['',f'![Retention comparisons]({(OUT/"retention.png").resolve()})']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:25]))
if __name__=='__main__':main()
