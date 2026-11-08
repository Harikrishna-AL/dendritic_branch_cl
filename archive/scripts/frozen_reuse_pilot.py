"""Frozen-feature forward transfer with explicit task-selected readouts.

Task identities are supplied: routing is NOT learned in this experiment.
Branches compute tanh(k*a+b); coefficients are task-specific. No replay of A
during B. The extra-feature task is outside the scaled A-feature family.
"""
import json
from pathlib import Path
import numpy as np

ARMS = ['modify', 'frozen', 'frozen_residual', 'scratch']
STEPS = [0, 10, 25, 50, 100, 200, 400, 800, 1600]
GRID = np.linspace(-1, 1, 401)


def predict(x, branches, coefficients):
    return np.tanh(x[:, None] * branches[:, 0] + branches[:, 1]) @ coefficients


def update(x, y, branches, coefficients, trainable, rate=.08):
    h = np.tanh(x[:, None] * branches[:, 0] + branches[:, 1])
    error = h @ coefficients - y
    local = 2 * error[:, None] * coefficients * (1 - h*h)
    gradients = np.stack([(local*x[:, None]).mean(0), local.mean(0)], axis=1)
    coefficient_gradient = 2 * (error[:, None] * h).mean(0)
    branches[trainable] -= rate * gradients[trainable]
    coefficients -= rate * coefficient_gradient


def target(x, task):
    return .75*np.tanh(2*x) + (0 if task == 'scaled' else .4*np.tanh(3*x+1))


def run(seed, task, arm):
    rng = np.random.default_rng(seed)
    branch_a = np.array([[.2+rng.normal(0,.03), rng.normal(0,.03)]])
    readout_a = np.ones(1)
    for _ in range(800):
        x = rng.uniform(-1,1,32)
        update(x, np.tanh(2*x)+rng.normal(0,.03,32), branch_a, readout_a, np.array([True]))
    # Copy the learned A branch but give B its own new coefficient(s).
    before_a = float(np.mean((predict(GRID,branch_a,readout_a)-np.tanh(2*GRID))**2))
    if arm in ['modify','frozen']:
        branches = branch_a.copy()
        coefficients = np.zeros(1)
        trainable = np.array([arm == 'modify'])
    else:
        branches = np.array([[.2, .1],[-.2,-.1]])
        if arm == 'frozen_residual':
            branches[0] = branch_a[0]
        coefficients = np.array([0., .1])
        trainable = np.array([arm == 'scratch', True])
    # Paired B batches across arms; evaluation uses no noisy targets or updates.
    rng = np.random.default_rng(seed+10000)
    curve=[]
    for step in range(max(STEPS)+1):
        if step in STEPS:
            b_error=float(np.mean((predict(GRID,branches,coefficients)-target(GRID,task))**2))
            a_branch=branches[:1] if arm == 'modify' else branch_a
            a_error=float(np.mean((predict(GRID,a_branch,readout_a)-np.tanh(2*GRID))**2))
            curve.append({'step':step,'B_mse':b_error,'A_mse':a_error})
        if step < max(STEPS):
            x=rng.uniform(-1,1,32)
            update(x,target(x,task)+rng.normal(0,.03,32),branches,coefficients,trainable)
    if arm in ['frozen','frozen_residual']:
        assert np.array_equal(branches[0],branch_a[0]), 'Frozen feature changed'
    return {'seed':seed,'task':task,'arm':arm,'A_before':before_a,'curve':curve}


def main():
    runs=[run(seed,task,arm) for task in ['scaled','extra_feature'] for arm in ARMS for seed in range(20)]
    summaries=[]
    for task in ['scaled','extra_feature']:
        for arm in ARMS:
            group=[r for r in runs if r['task']==task and r['arm']==arm]
            curve=[]
            for i,step in enumerate(STEPS):
                values=np.array([r['curve'][i]['B_mse'] for r in group])
                curve.append({'step':step,'B_mse':float(values.mean()),'B_sd':float(values.std(ddof=1)),
                              'A_mse':float(np.mean([r['curve'][i]['A_mse'] for r in group]))})
            summaries.append({'task':task,'arm':arm,'curve':curve})
    out=Path('results/frozen_reuse_pilot');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summaries},indent=2))
    lines=['# Frozen reuse experiment','','Explicit task-selected readouts isolate transfer from routing. '
           'Twenty paired seeds; A pretraining: 800 minibatches; B: 1600 minibatches of 32 observations. '
           'Noise SD 0.03. Evaluation: 401 noiseless action values. Branch features tanh(k*a+b) are learned. '
           'Frozen reuse trains a new coefficient. Frozen residual adds a trainable branch. Scratch has '
           'the same two-branch architecture as frozen residual, without pretrained features. '
           'Modify shares and changes the A branch but keeps distinct readouts. No A replay during B.', '',
           '| Task | Arm | B MSE at 100 batches | B MSE at 1600 | A MSE after B |',
           '|---|---|---|---|---|']
    for s in summaries:
        early=next(c for c in s['curve'] if c['step']==100); final=s['curve'][-1]
        line=f"| {s['task']} | {s['arm']} | {early['B_mse']:.6f} | {final['B_mse']:.6f} | {final['A_mse']:.6f} |"
        lines.append(line);print(line)
    lines+=['','A target: tanh(2a). Scaled B: 0.75*tanh(2a). Extra-feature B: '
            '0.75*tanh(2a)+0.4*tanh(3a+1). Forward transfer must be judged from B learning '
            'curves against scratch, not retention alone. Pretraining cost is excluded from B curves. '
            'Scratch retains A in a separate untouched model: its A retention is trivial, and '
            'its total stored capacity exceeds frozen residual by one branch. B architecture '
            'capacity is matched for scratch versus frozen residual; total storage is not. '
            'This toy uses explicit context, fixed capacity and favorable feature overlap; it does '
            'not establish task-free routing or general continual learning. Frozen-weight assertions '
            'run for every frozen arm. Full curves and seed variability are in results.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
