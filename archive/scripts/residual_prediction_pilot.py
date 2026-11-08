"""Residual learning ablation; no task decomposition or coefficient oracle.

Explicit residual joint updates are algebraically identical to total-error
updates. A separate staged arm fits the reused readout for 100 batches, freezes
it, then trains the residual for 1500 batches, using the same 1600-batch budget.
"""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.frozen_reuse_pilot import GRID, STEPS, predict, update, target, run


def residual_update(x, y, branches, coefficients):
    shared = predict(x, branches[:1], coefficients[:1])
    residual_target = y - shared
    residual_prediction = predict(x, branches[1:], coefficients[1:])
    error = residual_prediction - residual_target
    h = np.tanh(x * branches[0,0] + branches[0,1])
    shared_gradient = 2*np.mean(error*h)
    update(x, residual_target, branches[1:], coefficients[1:], np.array([True]))
    coefficients[0] -= .08*shared_gradient


def experiment(seed, task, staged):
    rng = np.random.default_rng(seed)
    branch_a = np.array([[.2+rng.normal(0,.03),rng.normal(0,.03)]])
    readout_a = np.ones(1)
    for _ in range(800):
        x=rng.uniform(-1,1,32)
        update(x,np.tanh(2*x)+rng.normal(0,.03,32),branch_a,readout_a,np.array([True]))
    branches=np.array([branch_a[0],[-.2,-.1]])
    coefficients=np.array([0.,0. if staged else .1])
    before_a=float(np.mean((predict(GRID,branch_a,readout_a)-np.tanh(2*GRID))**2))
    rng=np.random.default_rng(seed+10000)
    curve=[]
    for step in range(1601):
        if step in STEPS:
            curve.append({'step':step,'B_mse':float(np.mean((predict(GRID,branches,coefficients)-target(GRID,task))**2)),
                          'A_mse':before_a})
        if step==1600: break
        x=rng.uniform(-1,1,32);y=target(x,task)+rng.normal(0,.03,32)
        if not staged:
            residual_update(x,y,branches,coefficients)
        elif step<100:
            update(x,y,branches[:1],coefficients[:1],np.array([False]))
        else:
            residual_target=y-predict(x,branches[:1],coefficients[:1])
            update(x,residual_target,branches[1:],coefficients[1:],np.array([True]))
    assert np.array_equal(branches[0],branch_a[0])
    return {'seed':seed,'task':task,'arm':'staged_residual' if staged else 'explicit_joint_residual',
            'A_before':before_a,'curve':curve,'final_coefficients':coefficients.tolist()}


def main():
    # Verify that merely rewriting total error as residual error changes nothing.
    rng=np.random.default_rng(99)
    for _ in range(50):
        x=rng.uniform(-1,1,32);y=rng.normal(size=32)
        b=rng.normal(size=(2,2));c=rng.normal(size=2)
        b2=b.copy();c2=c.copy()
        update(x,y,b,c,np.array([False,True]))
        residual_update(x,y,b2,c2)
        assert np.allclose(b,b2,atol=1e-12) and np.allclose(c,c2,atol=1e-12)
    runs=[]
    for task in ['scaled','extra_feature']:
        for seed in range(20):
            baseline=run(seed,task,'frozen_residual')
            explicit=experiment(seed,task,False)
            assert np.allclose([c['B_mse'] for c in baseline['curve']],
                               [c['B_mse'] for c in explicit['curve']],atol=1e-12)
            runs.extend([baseline,explicit,experiment(seed,task,True),run(seed,task,'scratch')])
    summary=[]
    for task in ['scaled','extra_feature']:
        for arm in ['frozen_residual','explicit_joint_residual','staged_residual','scratch']:
            group=[r for r in runs if r['task']==task and r['arm']==arm]
            curve=[]
            for i,step in enumerate(STEPS):
                values=[r['curve'][i]['B_mse'] for r in group]
                curve.append({'step':step,'B_mse':float(np.mean(values)),
                              'B_sd':float(np.std(values,ddof=1)),
                              'A_mse':float(np.mean([r['curve'][i]['A_mse'] for r in group]))})
            summary.append({'task':task,'arm':arm,'curve':curve})
    out=Path('results/residual_prediction_pilot');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summary},indent=2))
    lines=['# Explicit residual experiment','','Twenty paired seeds per task and arm. Same learned A feature, '
           'initialization and noisy B batches as frozen_reuse_pilot. A pretraining: 800 batches. '
           'B budget: 1600 batches of 32 observations. Explicit task selection; no A replay.','','Joint residual: '
           'target residual = observed consequence minus current shared prediction. Both B coefficients '
           'and the new branch train. This is algebraically identical to the previous joint loss, '
           'verified with 50 random gradient comparisons and all 40 paired learning curves.','','Staged residual: '
           'fit only B reuse coefficient for 100 batches, freeze it, then train the new branch '
           'and its coefficient for 1500 batches. No true decomposition or true coefficient supplied.','','| Task | Arm | B error at 100 | B error at 1600 | A error |','|---|---|---|---|---|']
    for s in summary:
        early=next(c for c in s['curve'] if c['step']==100);last=s['curve'][-1]
        line=f"| {s['task']} | {s['arm']} | {early['B_mse']:.6f} | {last['B_mse']:.6f} | {last['A_mse']:.6f} |"
        lines.append(line);print(line)
    lines+=['','A target tanh(2a). B is 0.75*tanh(2a), or that plus 0.4*tanh(3a+1). '
            'The new feature overlaps the old feature, so fitting only the reuse coefficient '
            'does not identify the generating coefficient 0.75. Its residual may not fit a single tanh. '
            'Scratch retains A separately, using three stored branches versus two for frozen reuse. '
            'B architecture capacity is matched; total storage is not. Frozen A assertions passed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
