"""Report saved runs without retraining or selecting on test outcomes."""
import json,statistics,random
from pathlib import Path
import torch
from bdh_cl.scripts.shared_context_memory_pilot import Model,evaluate,mean_error

def finish_controls():
    out=Path('results/shared_context_memory_corrected_20261011')
    records=json.loads((out/'validation.json').read_text())
    assert len(records)==12, 'wait for all corrected control training runs'
    torch.set_num_threads(2)
    for r in records:
        model=Model(r['variant']);key=f'{r["seed"]}_{r["order"][0]}_{r["variant"]}'
        states=torch.load(out/f'{key}.pt',weights_only=True);model.load_state_dict(states[-1])
        r['test']=evaluate(model,r['seed']+3000);model.load_state_dict(states[-1]);r['erased']=evaluate(model,r['seed']+3000,True)
    (out/'results.json').write_text(json.dumps(records,indent=2))
    return records

def rows(path):return json.loads(Path(path).read_text())
def metrics(rs):
    return {'old':statistics.mean(p['mse'] for r in rs for p in r['test'] if p['regime']==r['order'][0]),
            'new':statistics.mean(p['mse'] for r in rs for p in r['test'] if p['regime']==r['order'][1]),
            'avg':statistics.mean(mean_error(r['test']) for r in rs),
            'transition':statistics.mean(p['early_mse'] for r in rs for p in r['test']),
            'erased':statistics.mean(mean_error(r['erased']) for r in rs)}

def main():
    corrected=Path('results/shared_context_memory_corrected_20261011/results.json')
    controls=rows(corrected) if corrected.exists() else finish_controls()
    dev=rows('results/shared_context_memory_dev_20261011/results.json');confirm=rows('results/branch_memory_confirmation_20261011/results.json')
    lines=['# Comparison and fresh-seed confirmation','','Lower MSE is better; same 384 trainable parameters, no task IDs or boundary flags. Main MSE excludes first 32 observations per visit. Shared controls have a larger effective state budget.','','## Matched development seeds 0–2, both orders','','| Context | Old MSE | New MSE | Avg MSE | Transition MSE | Erased MSE |','|---|---:|---:|---:|---:|---:|']
    for variant,source in [('baseline',dev),('centered',dev),('linear',dev),('neuron_shared',controls),('layer_shared',controls)]:
        m=metrics([r for r in source if r['variant']==variant]);lines.append('| '+variant+' | '+' | '.join(f'{m[k]:.3f}' for k in ('old','new','avg','transition','erased'))+' |')
    fresh_shared=rows('results/shared_context_memory_confirmation_20261011/results.json')
    lines+=['','## Fresh seeds 3–7, both orders','','Linear was selected on development validation; centered was the prespecified nonlinear companion. Neither is reselected using confirmation outcomes.','','| Variant | Old MSE | New MSE | Avg MSE | Transition MSE | Erased MSE |','|---|---:|---:|---:|---:|---:|']
    for v in ('baseline','linear','centered','neuron_shared','layer_shared'):
        source=fresh_shared if v in ('neuron_shared','layer_shared') else confirm
        m=metrics([r for r in source if r['variant']==v]);lines.append('| '+v+' | '+' | '.join(f'{m[k]:.3f}' for k in ('old','new','avg','transition','erased'))+' |')
    paired={};rng=random.Random(73)
    for v in ('linear','centered'):
        differences=[]
        for seed in range(3,8):
            a=metrics([r for r in confirm if r['seed']==seed and r['variant']=='baseline'])['avg']
            b=metrics([r for r in confirm if r['seed']==seed and r['variant']==v])['avg'];differences.append(a-b)
        boot=sorted(statistics.mean(rng.choices(differences,k=5)) for _ in range(10000))
        paired[v]={'baseline_minus_candidate_by_seed':differences,'mean_gain':statistics.mean(differences),'seed_wins':sum(d>0 for d in differences),'seed_cluster_bootstrap_95_interval':[boot[250],boot[9749]],'caution':'only five independent seeds; exploratory interval, two orders grouped within seed'}
        lines+=['',f'{v}: improves {paired[v]["seed_wins"]}/5 seed means; paired MSE gain {statistics.mean(differences):.4f}, exploratory 95% seed-bootstrap interval [{boot[250]:.4f}, {boot[9749]:.4f}].']
    for v in ('neuron_shared','layer_shared'):
        diffs=[]
        for seed in range(3,8):
            local=metrics([r for r in confirm if r['seed']==seed and r['variant']=='baseline'])['avg']
            shared=metrics([r for r in fresh_shared if r['seed']==seed and r['variant']==v])['avg'];diffs.append(shared-local)
        paired[v]={'shared_minus_baseline_by_seed':diffs,'mean_local_gain':statistics.mean(diffs),'local_seed_wins':sum(d>0 for d in diffs)}
        lines+=['',f'Local baseline vs {v}: lower MSE on {sum(d>0 for d in diffs)}/5 seed means; mean local gain {statistics.mean(diffs):.4f}.']
    lines+=['','Linear removes the hidden nonlinear activation; this gain is not evidence that the original nonlinear architecture improved by the same amount. Centered adds one state scalar per branch. Shared moments are analytically reconstructed in current weight coordinates, while local traces retain historical responses. No claim that memory location alone caused a difference or that dendrites are uniquely necessary.','', 'Initial development shared-control rows are superseded by the corrected control run. Failed softsign and individual seed regressions remain in raw records.']
    Path('results/branch_memory_confirmation_20261011/paired_analysis.json').write_text(json.dumps(paired,indent=2))
    Path('results/branch_memory_confirmation_20261011/comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
