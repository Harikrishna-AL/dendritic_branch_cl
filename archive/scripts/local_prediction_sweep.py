"""Similarity/noise/threshold ablation of the engineered branch predictor.

Scores compare predictions to noiseless expected consequences; noisy observed
consequences route only the NEXT prediction. Evaluation never updates weights.
"""
import csv
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.local_prediction_pilot import Predictor


def evaluate(model, coefficient, noise, seed):
    rng = np.random.default_rng(seed)
    active = model.active
    losses = []
    switches = 0
    before = model.weights.copy()
    for _ in range(200):
        feature = np.tanh(2 * rng.choice([-1, 1]) * rng.uniform(.7, 1))
        target = coefficient * feature
        losses.append(float((target - model.weights[active] * feature) ** 2))
        observed = target + rng.normal(0, noise)
        if model.mode == 'competitive':
            candidates = np.flatnonzero(model.used)
            chosen = int(candidates[np.argmin((observed - model.weights[candidates] * feature) ** 2)])
            switches += chosen != active
            active = chosen
    assert np.array_equal(before, model.weights), 'Evaluation changed weights'
    return {'first': losses[0], 'settled': float(np.mean(losses[1:])),
            'switch_rate': switches / 200}


def run(seed, mode, alpha, noise, threshold):
    rng = np.random.default_rng(seed)
    model = Predictor(mode, threshold)
    stages = {}
    for stage, coefficient in [('A', 1), ('B', alpha), ('return', 1)]:
        for _ in range(400):
            action = rng.choice([-1, 1]) * rng.uniform(.7, 1)
            model.observe(action, coefficient * np.tanh(2 * action) + rng.normal(0, noise))
        stages[stage] = {'A': evaluate(model, 1, noise, seed + 1000),
                         'B': evaluate(model, alpha, noise, seed + 2000),
                         'branches': int(model.used.sum()), 'weights': model.weights.tolist()}
    return {'seed': seed, 'mode': mode, 'alpha': alpha, 'noise': noise, 'threshold': threshold,
            'A_before': stages['A']['A']['settled'], 'A_after': stages['B']['A']['settled'],
            'forgetting': stages['B']['A']['settled'] - stages['A']['A']['settled'],
            'B_after': stages['B']['B']['settled'], 'B_after_return': stages['return']['B']['settled'],
            'A_first_after_B': stages['B']['A']['first'],
            'A_route_switch_rate': stages['B']['A']['switch_rate'],
            'branches_after_A': stages['A']['branches'], 'branches_after_B': stages['B']['branches']}


def main():
    rows = []
    for mode in ['shared', 'competitive']:
        for alpha in [1, .9, .75, .5, 0, -.5, -1]:
            for noise in [0, .03, .15]:
                for threshold in [.01, .1, .25]:
                    for seed in range(20):
                        rows.append(run(seed, mode, alpha, noise, threshold))
    output = Path('results/local_prediction_sweep')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (row['mode'], row['alpha'], row['noise'], row['threshold'])
        grouped.setdefault(key, []).append(row)
    summaries = []
    metrics = list(rows[0])[5:]
    for key, group in grouped.items():
        summary = dict(zip(['mode', 'alpha', 'noise', 'threshold'], key))
        for metric in metrics:
            values = [r[metric] for r in group]
            summary[metric] = float(np.mean(values))
            summary[metric + '_sd'] = float(np.std(values, ddof=1))
        summaries.append(summary)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    report = [
        '# Local prediction sweep', '',
        'A predicts tanh(2a); B predicts alpha*tanh(2a). Training: 400 noisy observations '
        'each of A, B, then A again. Twenty paired seeds per condition; 200 evaluation '
        'steps per context. Evaluation freezes weights and scores against noiseless '
        'expected consequences, while routing uses noisy feedback from the preceding step.', '',
        'Recruitment thresholds are squared-error thresholds, not error magnitudes. '
        'Two branches, fixed nonlinear feature, engineered novelty allocation and '
        'one-observation initialization; no task labels or replay. Inputs contain no '
        'current-context cue. This is a mechanism ablation, not biological evidence.', '',
        '## Means at noise standard deviation 0.03 and threshold 0.25', '',
        '| alpha | model | A error after B | B error after B | B error after returning to A | branches after A / B |',
        '|---|---|---|---|---|---|']
    for s in summaries:
        if s['noise'] == .03 and s['threshold'] == .25:
            report.append(f"| {s['alpha']} | {s['mode']} | {s['A_after']:.6f} | "
                          f"{s['B_after']:.6f} | {s['B_after_return']:.6f} | "
                          f"{s['branches_after_A']:.2f} / {s['branches_after_B']:.2f} |")
    report += ['', 'All condition means and sample standard deviations: summary.json. '
               'Per-seed metrics: runs.csv. The first prediction after switching is '
               'reported separately; settled scores exclude it. With noisy feedback '
               'routing can continue switching, so settled does not imply perfect identification.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    for s in summaries:
        if s['mode'] == 'competitive' and s['noise'] in [.03, .15]:
            print(f"alpha={s['alpha']:5g} noise={s['noise']:.2f} threshold={s['threshold']:.2f} "
                  f"A={s['A_after']:.5f} B={s['B_after']:.5f} branches={s['branches_after_B']:.2f}")
    print(f'Saved {len(rows)} runs, {len(summaries)} conditions to {output}')


if __name__ == '__main__':
    main()
