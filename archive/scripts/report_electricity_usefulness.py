"""Paired attribution of dense training, aggregation, and usefulness memory."""
import json,statistics,random
from pathlib import Path
import torch

DEV=Path('results/electricity_usefulness_dev_20261011');OUT=Path('results/electricity_usefulness_confirmation_20261011')
def main():
    rows=json.loads((OUT/'results.json').read_text());assert len(rows)==35
    variants=list(dict.fromkeys(r['variant'] for r in rows));rng=random.Random(119);paired={}
    for candidate,control in [('quality_soft','predict_winner'),('quality_soft','predict_all'),('quality_soft','uniform_all'),('quality_all','uniform_all'),('quality_soft','mlp'),('quality_soft','gru')]:
        diff=[]
        for seed in range(3,8):
            a=next(r['online']['2014']['mse'] for r in rows if r['seed']==seed and r['variant']==control)
            b=next(r['online']['2014']['mse'] for r in rows if r['seed']==seed and r['variant']==candidate);diff.append(a-b)
        bootstrap=sorted(statistics.mean(rng.choices(diff,k=5)) for _ in range(10000))
        paired[candidate+'_vs_'+control]={'control_minus_candidate_by_seed':diff,'mean_gain':statistics.mean(diff),'candidate_seed_wins':sum(d>0 for d in diff),'seed_bootstrap95':[bootstrap[250],bootstrap[9749]],'limits':'five initializations of the same16customer cohort; not independent dataset replications'}
    identical=True
    for seed in range(3,8):
        states=[torch.load(OUT/f'{seed}_{v}_2012.pt',weights_only=False)['model'] for v in ('predict_all','quality_all','quality_soft','uniform_all')]
        for name in ('first.weight','first.bias','second.weight','second.bias'):
            for state in states[1:]:identical=identical and torch.equal(states[0][name],state[name])
    (OUT/'paired.json').write_text(json.dumps({'pairs':paired,'2012_dense_control_weights_identical':identical},indent=2))
    lines=['# Forecast usefulness follow-up','','Development: original16customers,2013only,seeds0–2. Confirmation: disjoint16customers,2014,seeds3–7. Best quality candidate was soft error weighting; uniform averaging was the overall development winner. Both were locked before confirmation.','','| Variant | 2014 frozen MSE | 2014 online MSE | Median client online MSE |','|---|---:|---:|---:|']
    import numpy as np
    for v in variants:
        rs=[r for r in rows if r['variant']==v]
        a=statistics.mean(r['frozen']['2014']['mse'] for r in rs);b=statistics.mean(r['online']['2014']['mse'] for r in rs)
        med=np.median(np.mean([r['online']['2014']['per_client_mse'] for r in rs],axis=0))
        lines.append(f'| {v} | {a:.4f} | {b:.4f} | {med:.4f} |')
    lines+=['','## Paired online comparisons','','Positive gain means lower MSE for the candidate. Seed bootstrap is exploratory and does not measure generalization across datasets.','','| Candidate vs control | Mean gain | Seed wins | Bootstrap95 |','|---|---:|---:|---|']
    for name,p in paired.items():lines.append(f'| {name} | {p["mean_gain"]:.5f} | {p["candidate_seed_wins"]}/5 | [{p["seed_bootstrap95"][0]:.5f}, {p["seed_bootstrap95"][1]:.5f}] |')
    lines+=['',f'All-output training variants have identical2012 slow weights for each paired seed: {identical}. Their forecast differences isolate output routing/aggregation, not extra parameter learning during initial training. The quality rule has extra delayed-feedback storage, so complete memory budgets are not matched.','', 'Quality feedback uses factual load only when it arrives24hours after the forecast. Stored forecast errors continue updating even when slow weights are frozen. This is a causal delayed-feedback rule, not the earlier target-free response-statistics rule. The hidden-layer branch cue remains predictability-based.','', 'Uniform averaging is an essential control: an improvement over selected-head training is not by itself evidence that fading branch-specific error memory caused it. Soft weighting also changes sparse output selection into an ensemble. Hard error routing is retained as a prespecified companion, not selected on confirmation outcomes.','', 'The same dataset/time period is used for development and the new customer cohort. All customers are selected using only the calibration prefix. No original-cohort2014 metrics were evaluated by this follow-up. Reduced forecasting error alone does not establish task separation, preservation of old circuits, or lower forgetting.']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
