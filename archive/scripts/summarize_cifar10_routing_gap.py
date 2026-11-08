"""Report the CIFAR-10 diagnostic, including outcomes that limit the hypothesis."""
import json
from pathlib import Path
import statistics as st
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from bdh_cl.scripts.cifar10_routing_gap import ROOT,VisionBranch,ContextRidge,evaluate_branch
from bdh_cl.data.cifar10 import load

SOURCE=ROOT/'results/cifar10_routing_gap_20261031'
OUT=ROOT/'reports/cifar10_routing_gap_20261031'
MODES=['current','prototype','learned_context','entropy','margin','random_route','wrong_route','oracle','bank_any_correct']
LABELS={'current':'Current global gate','prototype':'Input-prototype router','learned_context':'Learned context router',
        'entropy':'Lowest-entropy route','margin':'Largest-margin route','random_route':'Random saved route',
        'wrong_route':'Wrong-pair route (diagnostic)','oracle':'True-pair route (diagnostic)',
        'bank_any_correct':'Any saved route correct (diagnostic)'}


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((SOURCE/'results.json').read_text())
    assert len(rows)==24
    selected=json.loads((SOURCE/'selected_hyperparameters.json').read_text())
    summary=[]
    for method in ['aux','ce','pec','joint_aux','joint_ce']:
        group=[r for r in rows if r['method']==method]
        expected=3 if method.startswith('joint_') else 6
        assert len(group)==expected and len({(r['seed'],r['order']) for r in group})==expected
        modes=MODES if method in ['aux','ce'] else ['autonomous']
        for mode in modes:
            values=[st.mean(100*r['scores'][mode]['accuracy'] for r in group if r['seed']==seed) for seed in [201,202,203]]
            if mode=='prototype':memory=group[0]['prototype_persistent_bytes']
            elif mode=='learned_context':memory=group[0]['learned_router_persistent_bytes']
            elif mode=='current' or mode=='autonomous':memory=group[0]['model_bytes']
            else:memory=group[0]['model_bytes']+group[0]['route_bytes']
            summary.append(dict(method=method,mode=mode,accuracy=st.mean(values),sd=st.stdev(values),seed_values=values,
                                retained_payload_mb=memory/1e6,optimizer_mb=group[0]['optimizer_state_bytes']/1e6,
                                parameters=group[0]['parameters'],mean_training_seconds=st.mean(r['training_seconds'] for r in group)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lookup={(r['method'],r['mode']):r for r in summary}
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','mode','accuracy','sd','retained_payload_mb','optimizer_mb','parameters'])
        writer.writeheader()
        for r in summary:writer.writerow({k:r[k] for k in writer.fieldnames})
    gaps={}
    for arm in ['aux','ce']:
        auto=lookup[arm,'prototype'];oracle=lookup[arm,'oracle'];learned=lookup[arm,'learned_context']
        differences=[o-a for o,a in zip(oracle['seed_values'],auto['seed_values'])]
        gaps[arm]=dict(prototype_to_oracle_pp=st.mean(differences),seed_sd_pp=st.stdev(differences),
                       learned_to_oracle_pp=oracle['accuracy']-learned['accuracy'],
                       bank_any_correct_percent=lookup[arm,'bank_any_correct']['accuracy'])
    effects={}
    for mode in MODES:
        a=lookup['aux',mode]['seed_values'];b=lookup['ce',mode]['seed_values']
        delta=[x-y for x,y in zip(a,b)];mean=st.mean(delta);half=4.302652729911275*st.stdev(delta)/np.sqrt(3)
        effects[mode]=dict(aux_minus_ce_pp=mean,seed_deltas_pp=delta,descriptive_95pct_t_interval_pp=[mean-half,mean+half])
    (OUT/'gap_and_loss_effect.json').write_text(json.dumps(dict(gaps=gaps,auxiliary_effects=effects),indent=2))
    order_summary=[]
    for method in ['aux','ce','pec']:
        for order in ['forward','reverse']:
            group=[r for r in rows if r['method']==method and r['order']==order]
            for mode in (['prototype','learned_context','oracle'] if method!='pec' else ['autonomous']):
                values=[100*r['scores'][mode]['accuracy'] for r in group]
                order_summary.append(dict(method=method,order=order,mode=mode,accuracy=st.mean(values),sd=st.stdev(values)))
    (OUT/'order_summary.json').write_text(json.dumps(order_summary,indent=2))
    # Same validation images at acquisition and at final evaluation distinguish
    # retention loss from the mere inability to select the archived route.
    audit_path=SOURCE/'retention_audit.json'
    retention=json.loads(audit_path.read_text()) if audit_path.exists() else []
    missing=[r for r in rows if r['method'] in ['aux','ce'] and not any(a['method']==r['method'] and a['seed']==r['seed'] and a['order']==r['order'] for a in retention)]
    if missing:
        data=load(ROOT/'data_cache/cifar10');vx=torch.tensor(data['images'][data['validation']].copy());vy=torch.tensor(data['labels'][data['validation']])
        for row in missing:
            state=torch.load(SOURCE/f"{row['method']}_{row['seed']}_{row['order']}.pt",weights_only=True,map_location='cpu')
            model=VisionBranch(row['seed']);model.load_state_dict(state['model'])
            router=ContextRidge(row['seed'])
            for name,value in state['router'].items():setattr(router,name,value)
            score=evaluate_branch(model,vx,vy,state['contexts'],state['routes'],router,row['config'])
            earlier=row['acquisition'][:-1];old_groups=[a['group'] for a in earlier]
            acquisition=st.mean(100*a['validation_accuracy'] for a in earlier)
            final_oracle=st.mean(100*score['oracle']['pair_accuracy'][k] for k in old_groups)
            retention.append(dict(method=row['method'],seed=row['seed'],order=row['order'],
                earlier_pairs_acquisition_percent=acquisition,earlier_pairs_final_oracle_percent=final_oracle,
                earlier_pairs_acquisition_to_final_drop_pp=acquisition-final_oracle,validation_final_scores=score))
            audit_path.write_text(json.dumps(retention,indent=2))
    retention_summary={}
    for arm in ['aux','ce']:
        group=[r for r in retention if r['method']==arm]
        retention_summary[arm]={field:st.mean(r[field] for r in group) for field in ['earlier_pairs_acquisition_percent','earlier_pairs_final_oracle_percent','earlier_pairs_acquisition_to_final_drop_pp']}
    (OUT/'retention_summary.json').write_text(json.dumps(retention_summary,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white');fig.savefig(OUT/f'{name}.svg',facecolor='white');plt.close(fig)
    # All routing conditions, with privileged diagnostics visibly separated.
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    modes=['current','prototype','learned_context','entropy','margin','random_route','oracle','bank_any_correct']
    x=np.arange(len(modes));width=.36
    for offset,arm,color,label in [(-width/2,'aux','#176ea8','Branch + auxiliary loss'),(width/2,'ce','#9ea7af','Branch CE-only')]:
        vals=[lookup[arm,m] for m in modes]
        bars=ax.bar(x+offset,[r['accuracy'] for r in vals],width,yerr=[r['sd'] for r in vals],capsize=3,color=color,label=label)
        for i,bar in enumerate(bars):
            if modes[i] in ['oracle','bank_any_correct']:bar.set_hatch('//')
            ax.text(bar.get_x()+width/2,bar.get_height()+vals[i]['sd']+1,f"{bar.get_height():.1f}",ha='center',fontsize=8)
    ax.axvline(5.5,color='#b39155',ls='--',lw=1)
    ax.set(xticks=x,xticklabels=['Current\ngate','Input\nprototype','Learned\ncontext','Route\nentropy','Route\nmargin','Random\nroute','True-pair\noracle','Any route\ncorrect'],
           ylim=(0,110),ylabel='Final ten-class accuracy (%)',title='Split CIFAR-10: does saved-route access recover retained skills?')
    ax.legend(frameon=False,loc='upper left')
    fig.supxlabel('Same final model weights within each arm; all ten output classes remain eligible. Hatched bars use test labels diagnostically.\nThree seed means, each averaging two task orders; bars show seed SD. Known training boundaries used for route archives and context learning.',fontsize=9)
    save(fig,'01_routing_and_oracle')
    # The independent joint references are controls for learnability, not an
    # alternative inference mode of the continually trained final weights.
    entries=[('aux','prototype','Auxiliary + prototype'),('aux','learned_context','Auxiliary + learned context'),
             ('aux','oracle','Auxiliary + oracle route'),('ce','prototype','CE-only + prototype'),('ce','oracle','CE-only + oracle route'),
             ('pec','autonomous','PEC, size-matched CNNs'),('joint_aux','autonomous','Joint auxiliary (offline)'),('joint_ce','autonomous','Joint CE-only (offline)')]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,6),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
    yy=np.arange(len(entries))
    for i,(method,mode,label) in enumerate(entries):
        r=lookup[method,mode];color='#b64a90' if method=='pec' else '#c8a051' if mode=='oracle' else '#176ea8' if method=='aux' else '#9ea7af'
        ax.barh(i,r['accuracy'],color=color);bx.barh(i,r['retained_payload_mb'],color=color)
        ax.errorbar(r['accuracy'],i,xerr=r['sd'],fmt='none',color='#333',capsize=3)
        ax.text(r['accuracy']+r['sd']+1,i,f"{r['accuracy']:.2f}%",va='center',fontsize=9)
        bx.text(r['retained_payload_mb']+.03,i,f"{r['retained_payload_mb']:.3f} MB",va='center',fontsize=9)
    ax.set(yticks=yy,yticklabels=[v[2] for v in entries],xlim=(0,110),xlabel='Final ten-class accuracy (%)')
    ax.invert_yaxis();bx.set(xlim=(0,max(lookup[m,k]['retained_payload_mb'] for m,k,_ in entries)*1.4),xlabel='Retained numerical state (MB)')
    fig.suptitle('From-scratch CNN capacity, continual learning and storage controls',fontsize=15)
    fig.supxlabel('Branch:525,014 parameters; PEC:522,927 including shared frozen teacher. PCA and external pretraining are absent.\nJoint training uses all classes together; oracle inference uses true pair IDs. Neither is a deployable continual-learning result. Adam and peak RAM excluded.',fontsize=9)
    save(fig,'02_capacity_pec_and_memory')
    # Acquisition and final oracle use exactly the same validation examples.
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for i,arm in enumerate(['aux','ce']):
        r=retention_summary[arm]
        ax.bar(i-.18,r['earlier_pairs_acquisition_percent'],.36,color='#8faabd',label='At acquisition' if i==0 else None)
        ax.bar(i+.18,r['earlier_pairs_final_oracle_percent'],.36,color='#c8a051',label='Final weights, true-pair route' if i==0 else None)
        for offset,key in [(-.18,'earlier_pairs_acquisition_percent'),(.18,'earlier_pairs_final_oracle_percent')]:
            ax.text(i+offset,r[key]+1,f"{r[key]:.1f}%",ha='center',fontsize=10)
    ax.set(xticks=[0,1],xticklabels=['Auxiliary loss','CE-only'],ylim=(0,110),ylabel='Earlier-pair validation accuracy (%)',
           title='Are earlier capabilities still retained under oracle routing?')
    ax.legend(frameon=False,loc='upper right')
    fig.supxlabel('First four acquired pairs, averaged across three seeds × two orders. Same held-out validation images at both times.\nAll ten classes compete; the final diagnostic selects the saved pair route but leaves weights unchanged.',fontsize=9)
    save(fig,'03_acquisition_and_retention')
    lines=['# Split CIFAR-10: from-scratch CNN routing-gap experiment','',
        '**New measured CNN experiment. No pretrained encoder, PCA, historical-example replay or test-time learning.** '
        'The aim is to test whether the MNIST route-access diagnosis survives natural images; failure to reproduce that diagnosis is an informative outcome.','',
        '| Method | Current gate | Prototype router | Learned context | Entropy router | True-pair oracle | Any route correct |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,name in [('aux','Auxiliary loss'),('ce','CE-only')]:
        cells=[f"{lookup[arm,m]['accuracy']:.2f} ± {lookup[arm,m]['sd']:.2f}%" for m in ['current','prototype','learned_context','entropy','oracle','bank_any_correct']]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','| Capacity / comparator control | Final test accuracy | Numerical model state |',
        '|---|---:|---:|']
    for method,name in [('pec','PEC'),('joint_aux','Offline joint auxiliary'),('joint_ce','Offline joint CE-only')]:
        r=lookup[method,'autonomous'];lines.append(f"| {name} | {r['accuracy']:.2f} ± {r['sd']:.2f}% | {r['retained_payload_mb']:.6f} MB |")
    lines+=['','## Diagnostic findings','']
    for arm,name in [('aux','Auxiliary loss'),('ce','CE-only')]:
        r=gaps[arm];ret=retention_summary[arm]
        lines.append(f"- **{name}:** true-pair oracle minus prototype accuracy = {r['prototype_to_oracle_pp']:.2f} points; "
                     f"oracle minus learned-context accuracy = {r['learned_to_oracle_pp']:.2f} points. "
                     f"Earlier-pair validation acquisition {ret['earlier_pairs_acquisition_percent']:.2f}% versus final oracle "
                     f"{ret['earlier_pairs_final_oracle_percent']:.2f}%, a {ret['earlier_pairs_acquisition_to_final_drop_pp']:.2f}-point acquisition-to-final decline.")
    effect=effects['prototype']
    lines+=['','### Task-order check','',
            '| Objective / task order | Prototype | Learned context | Oracle |',
            '|---|---:|---:|---:|']
    order_lookup={(r['method'],r['order'],r['mode']):r for r in order_summary}
    for arm in ['aux','ce']:
        for order in ['forward','reverse']:
            cells=[f"{order_lookup[arm,order,mode]['accuracy']:.2f}%" for mode in ['prototype','learned_context','oracle']]
            lines.append('| '+arm+' / '+order+' | '+' | '.join(cells)+' |')
    lines+=['',f"Auxiliary minus CE-only prototype accuracy: {effect['aux_minus_ce_pp']:+.2f} points; descriptive paired95%t interval "
        f"[{effect['descriptive_95pct_t_interval_pp'][0]:.2f}, {effect['descriptive_95pct_t_interval_pp'][1]:.2f}] across only three seed means.",'',
        'The true-pair oracle is not a guaranteed recoverable ceiling or proof that all weights/skills survived. '
        'The any-route-correct diagnostic is a label-informed best-of-the-five-archives reference. If acquisition-to-final oracle accuracy falls substantially, '
        'retention/representation or classifier calibration also needs work; improved autonomous selection alone cannot be assumed to restore acquisition performance. '
        'Neither saved-route diagnostic searches the full combinatorial configuration space. Low archived-route accuracy does not prove that no other configuration retains useful capability. '
        'The joint models are independently trained capacity controls, not inference configurations of the sequential model.','',
        '## Protocol','',
        'Official CIFAR-10 archive matches MD5c58f30108f718f92721af3b95e74349a. Downloaded from MIT Open Learning’s mirror after the original server proved slow; '
        'the archive is byte-equivalent under the official checksum. 45,000 fit images,5,000 validation images (500/class, split RNG20261102), and official10,000 test images. '
        'Natural class pairs0/1,2/3,4/5,6/7,8/9, forward/reverse order, final seeds201/202/203; tuning seed213 forward. '
        'Five epochs per task, batch128, continuous Adam, clipping5, standard channel normalization, no augmentation, CPUtwo threads. '
        'This is a short resource-controlled CNN experiment, not a full CIFAR-SOTA reproduction or strict single-pass online benchmark. '
        'The5epoch schedule differs from the preceding MNIST2epoch schedule; scores must not be pooled across datasets.','',
        '**Branch network:** three3×3 stride2 padded Conv/ReLU layers3→16→32→64, flatten1024, '
        '122neurons each with4tanh branches and one shared10-class head.525,014 trainable coefficients. '
        'The original per-neuron branch-error EMA.95 and hard minimum-error gate are preserved. '
        'Auxiliary loss is ordinary CE+.5×mean CE of single-neuron branch substitutions; it does not enumerate all joint branch combinations. '
        'CE-only retains error feedback, gates and route archives, removing only the auxiliary gradient. '
        'The encoder learns continually by backpropagation; no claim of isolated or biologically local CNN gradients.','',
        '**Autonomous routers:** prototype uses nearest full standardized RGB input task mean. '
        'Learned context uses128fixed random tanh features of8×8pooled RGB, online Gram/cross statistics and a ridge classifier over the five training phases. '
        'It uses known training boundaries to obtain context targets; it is not unsupervised task discovery. '
        'No old samples/features are replayed. Phase-end route archives use the same training boundary privilege. '
        'Entropy and margin compare all five route configurations; they cost additional branch/head evaluations. '
        'All ten classes remain eligible in every route and in every autonomous prediction. '
        'Only the true-pair/cyclic-wrong/best-of-bank diagnostics consult test labels to select or assess routes.','',
        '**PEC:** official PecCNN classes with GELU/InstanceNorm and2×2adaptive pooling, output99. '
        'Ten students with16/32 channels plus one shared frozen random teacher with117/234 channels,522,927 total coefficients '
        '(0.40% fewer than the branch network). All teacher weights are counted once; none is pretrained. '
        'Its intrinsic architecture and trainable fraction differ; this is a total-size comparison, not identical capacity or FLOPs. '
        'The frozen per-image teacher targets are cached for current-task epochs and discarded at the task boundary. '
        'No historical target cache survives into later task training. This is equivalent to recomputing the deterministic teacher without augmentation; '
        'the temporary current-task cache is working memory, not reported retained state.','',
        '**Selection:** each branch objective independently chooses LR .0005/.001/.003 by final prototype validation accuracy. '
        'PEC tests those LRs plus.01. Joint controls independently test the same three LRs using ordinary validation accuracy. '
        'Learned-context ridge .1/1/10 and entropy temperature .5/1/2 are selected on the validation run of each learning-rate candidate. '
        'All selections are frozen before final-test inference. The same selected configuration is then used on all six sequential confirmation runs. '
        'SD is across three seed means after averaging orders. No test-based best-router selection is reported as a primary result.','',
        f'Selected settings are saved in `../../results/cifar10_routing_gap_20261031/selected_hyperparameters.json`.','',
        '## Memory and reproducibility','',
        'Numerical retained state includes model weights and error traces, phase-end route indices, and the selected router’s means or '
        'random projection/Gram/cross/head tensors. Model/route/prototype and model/route/learned-router costs are separate deployment variants. '
        'Adam state is additional. Peak RAM, dataset caches, gradients, activations, temporary teacher targets and allocator/Python overhead are excluded. '
        'Offline joint reference memory entries show the model only and do not charge its privileged access to the full training dataset.','',
        'Checks cover exact counterfactual neuron replacement, equivalence to explicit restored gates, fixed-state inference, '
        'online router statistics versus a batch solution, all10outputs and full teacher/student parameter accounting. '
        'The saved split, protocol, source, validation trials, all24confirmation runs and checkpoints are in the results directory. '
        'The retention audit uses held-out validation images only after models/settings are frozen; it fits no parameters.','',
        '## Artifacts','',
        '- [Routing conditions and oracle diagnostics](01_routing_and_oracle.png)',
        '- [Joint capacity, PEC and memory](02_capacity_pec_and_memory.png)',
        '- [Earlier-pair acquisition versus final oracle retention](03_acquisition_and_retention.png)',
        '- [Summary CSV](summary.csv), JSON, editable SVGs, and per-seed gap/loss-effect values accompany the figures.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gaps=gaps,retention=retention_summary,controls=[r for r in summary if r['mode']=='autonomous']),indent=2))


if __name__=='__main__':main()
