"""Ablation conclusions and branch/expert parameter-retention diagnostics."""
import os,json,statistics,copy
from pathlib import Path
import torch
import numpy as np
from bdh_cl.scripts.branch_mechanism_ablations import make
from bdh_cl.scripts.branch_short_memory_pilot import stream
os.environ.setdefault('MPLCONFIGDIR','/tmp/bdh_mechanism_matplotlib')
BASE=Path('results');TOY=BASE/'branch_mechanism_toy_20261011';OUT=BASE/'branch_mechanism_real_20261011'
def read(folder):return json.loads((BASE/folder/'results.json').read_text())

@torch.no_grad()
def usage(model,seed,first,expert=False):
    m=copy.deepcopy(model);m.reset();pending=None;phases=[[] for _ in range(4)]
    for t,(x,y,_) in enumerate(stream(seed,1024,256,order=(first,1-first)*2)):
        if pending is not None and getattr(m,'feedback',False):m.observe(*pending)
        _,ticket=m.forecast(x);pending=(ticket,y.clone())
        if t%256>=32:
            u=ticket[1] if expert else torch.nn.functional.one_hot(ticket[1],4).float()
            phases[t//256].append(u.mean(0))
    p=[torch.stack(a).mean(0) for a in phases];return (p[0]+p[2])/2,(p[1]+p[3])/2

def main():
    torch.set_num_threads(2);toy=read('branch_mechanism_toy_20261011');real=read('branch_mechanism_real_20261011');shock=read('branch_mechanism_shock_20261011');extra=read('branch_mechanism_moe_gradient_20261011')
    assert len(toy)==66 and len(real)==33 and len(shock)==33 and len(extra)==6
    toy+=extra;modes=list(dict.fromkeys(r['mode'] for r in toy));summary={}
    for kind,rows in [('toy',toy),('real',real),('stress',shock)]:
        summary[kind]={v:{k:statistics.mean(r[k] for r in rows if r['mode']==v) for k in ('old_before','old_after','forgetting','new_after')} for v in dict.fromkeys(r['mode'] for r in rows)}
    exact=[]
    for kind,rows in [('toy',toy),('real',real),('stress',shock)]:
        for a in rows:
            if a['mode']!='full':continue
            b=next(r for r in rows if r['mode']=='central_exact' and r['seed']==a['seed'] and r.get('first')==a.get('first'))
            exact.append(all(a[k]==b[k] for k in ('old_before','old_after','forgetting','new_after')))
    diagnostics=[]
    for r in toy:
        mode=r['mode']
        if mode not in ('full','common_cue','dense_gradient','moe_sparse','uniform_output'):continue
        m=make(mode,2);m.reset();saved=torch.load(TOY/f'{r["seed"]}_{r["first"]}_{mode}.pt',weights_only=False)
        m.load_state_dict(saved['afterB']);a,b=usage(m,r['seed']+6000,r['first'],mode=='moe_sparse')
        overlap=torch.minimum(a,b).sum(-1).mean().item()
        m.load_state_dict(saved['afterA']);old,_=usage(m,r['seed']+7000,r['first'],mode=='moe_sparse')
        change=saved['afterB']['first.weight']-saved['afterA']['first.weight'];bias=saved['afterB']['first.bias']-saved['afterA']['first.bias']
        if mode=='moe_sparse':
            # Movement includes all parameters in each complete expert.
            firstchange=torch.cat((change,bias[:,None]),1).reshape(m.n,-1)
            output=(saved['afterB']['out']-saved['afterA']['out']);outbias=(saved['afterB']['bias']-saved['afterA']['bias'])[:,None]
            delta=torch.cat((firstchange,output,outbias),1).norm(dim=1);weight=old.flatten()
        else:delta=torch.cat((change,bias[:,None]),1).norm(dim=1);weight=old.flatten()
        oldmove=(delta*weight).sum()/weight.sum();othermove=(delta*(1-weight)).sum()/(1-weight).sum()
        diagnostics.append({'seed':r['seed'],'first':r['first'],'mode':mode,'route_overlap':overlap,'old_usage_weighted_parameter_movement':oldmove.item(),'complement_weighted_parameter_movement':othermove.item(),'note':'branch rows vs wholeexpertparameter groups have differentdimensions; comparemovementwithinmodel,notabsoluteacrossarchitectures'})
    (OUT/'mechanism_diagnostics.json').write_text(json.dumps(diagnostics,indent=2));(OUT/'combined.json').write_text(json.dumps({'summary':summary,'central_exact_all_pairs_identical':all(exact)},indent=2))
    lines=['# Mechanism ablations: evidence and falsification','','Three new initializations13–15; both controlled task orders. All selected settings fixed. Old/new heldout weights frozen; fast context and causal feedback may adapt.','','## Controlled regime test','','| Ablation | Old before | Old after | Forgetting | New after |','|---|---:|---:|---:|---:|']
    for v in modes:
        m=summary['toy'][v];lines.append('| '+v+' | '+' | '.join(f'{m[k]:.3f}' for k in ('old_before','old_after','forgetting','new_after'))+' |')
    lines+=['','## Conclusions supported by these tests','','1. Correct context-to-computation correspondence matters: erasing or cyclically misassigning first-branch scores sharply damages retention while leaving parameter/state counts unchanged (except erasure reduces effective use).','2. Selective hidden updates matter: the custom dense-gradient ablation keeps the hard forward rule but spreads backward credit to every hidden branch; old-task performance deteriorates strongly. Gradient allocation also changes optimization, so this is not solely an intervention on anatomical memory protection.','3. Private branch-response traces are unnecessary in this family: a shared full input-moment cue achieves similar or better performance with learned branch readers. Storage/representation differ, and the real shared matrix uses more memory.','4. Physical memory placement is unnecessary: central_exact preserves branch-indexed information and gives exactly identical results across every paired synthetic, customer-domain and stress run. This does not prove branch-indexed information itself is unnecessary.','5. A generic sparse mixture of small MLP experts reproduces and exceeds the synthetic retention benefit, with almost equal slow parameters and less persistent state. The hard-readout all-expert-update companion isolates training coverage within that generic architecture.','6. The output usefulness cue adds only a modest gain over uniform averaging in this architecture; the large baseline separation mainly concerns context routing and selective plasticity.','', '## Real measured electricity','','Customer-domain shift (original16→disjoint16) improves old-domain performance in every model. Fullold0.2296→0.1968;erased-hidden0.2310→0.1966;MLP0.2193→0.1929. Thus real-data anti-forgetting benefit is not established, and hidden predictive memory is not necessary for that observed transfer.','', 'Exploratory seasonal stress:July/August2014was selected from already-observed errors. Raw readings and24h targets were unchanged. Most models do not improve newer heldout forecasts after July training (full1.7605→1.7936). Weak/absent forgetting with unsuccessful acquisition is not evidence of successful continual learning. The high-variance pattern also occurs in2012; this is a seasonal stress period, not a verified novel exogenous shock.','', '## Diagnostic checks','','The route and parameter movement measurements below are associations supporting the causal ablations. Tasklabels are used only for analysis. Old-usage-weighted movement measures whether parameters serving the older regime move less during B; it does not establish biological protection.','','| Model | Old/new route overlap | Old-use-weighted movement | Complement movement |','|---|---:|---:|---:|']
    for v in ('full','common_cue','dense_gradient','uniform_output','moe_sparse'):
        rs=[r for r in diagnostics if r['mode']==v];vals=[statistics.mean(r[k] for r in rs) for k in ('route_overlap','old_usage_weighted_parameter_movement','complement_weighted_parameter_movement')]
        lines.append('| '+v+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','Movement dimensions differ between branches and complete experts; compare old/complement ratios within an architecture. Three seeds and an engineered family are limited evidence. Runtime state, pending-feedback storage and computation are reported, not exactly matched.','', 'The defensible mechanism claim is **context-dependent sparse credit assignment can reduce interference** in the controlled regime task. These tests do not establish **a uniquely dendritic mechanism** or **a real-data anti-forgetting advantage**. A generic expert system provides a counterexample to necessity. No failed controls, customers or periods were excluded.','',f'Exact central packing identity across all paired results: {all(exact)}. Three targeted tests passed (exact reexpression during training, forward-preserving dense credit, generic parameter budget/read-only tickets).']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    selected=['full','common_cue','erase_hidden','shuffle_hidden','dense_gradient','uniform_output','moe_sparse','moe_dense_hard']
    labels=['Full','Common cue','Erase cue','Misalign cue','Dense gradient','Uniform output','Generic sparse','Generic dense updates']
    axes[0].bar(range(len(selected)),[summary['toy'][v]['old_after'] for v in selected],color=['#3977ac','#27906c','#ac625d','#ac625d','#ac625d','#8c9baf','#27906c','#ac625d'])
    axes[0].set_xticks(range(len(selected)),labels,rotation=40,ha='right');axes[0].set_ylabel('Old-task MSE after new learning');axes[0].set_title('Controlled retention: mechanism ablations')
    rv=['full','erase_hidden','uniform_output','moe_sparse','mlp','gru'];x=list(range(len(rv)))
    axes[1].bar([t-.18 for t in x],[summary['real'][v]['old_before'] for v in rv],width=.36,label='Before',color='#8c9baf')
    axes[1].bar([t+.18 for t in x],[summary['real'][v]['old_after'] for v in rv],width=.36,label='After',color='#27906c')
    axes[1].set_xticks(x,['Full','Erase cue','Uniform','Generic sparse','MLP','GRU'],rotation=40,ha='right');axes[1].set_ylabel('Older customer-domain MSE');axes[1].set_title('Electricity: all models transfer positively');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'ablations.png',dpi=170);fig.savefig(OUT/'ablations.pdf');plt.close(fig)
    lines+=['',f'![Ablations]({(OUT/"ablations.png").resolve()})'];(OUT/'interpretation.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:28]))
if __name__=='__main__':main()
