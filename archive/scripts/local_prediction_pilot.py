"""Toy consequence prediction: no task IDs, reward, replay, or biological claims.

Explicit novelty allocation is an engineered mechanism. Routing uses the last
observed consequence, never the current evaluation target. Nonlinear branch
features are fixed; trainable branch readouts receive local prediction errors.
"""
import json
from pathlib import Path
import numpy as np


class Predictor:
    def __init__(self, mode, novelty_threshold=.25):
        self.mode = mode
        self.novelty_threshold = novelty_threshold
        self.weights = np.zeros(2)
        self.used = np.zeros(2, dtype=bool)
        self.active = 0

    def predict(self, action):
        return self.weights[self.active] * np.tanh(2 * action)

    def observe(self, action, consequence):
        feature = np.tanh(2 * action)
        errors = consequence - self.weights * feature
        if self.mode == 'shared':
            chosen = 0
        elif self.mode == 'all_branches':
            self.weights += .15 * errors * feature
            self.used[:] = True
            return
        else:
            candidates = np.where(self.used)[0]
            chosen = int(candidates[np.argmin(errors[candidates] ** 2)]) if len(candidates) else 0
            # Reserve unused capacity when all existing predictions fit poorly.
            if not len(candidates) or (errors[chosen] ** 2 > self.novelty_threshold and not self.used.all()):
                chosen = int(np.where(~self.used)[0][0])
                # Explicit one-sample initialization avoids allocating duplicate
                # branches while a newly recruited branch is still warming up.
                self.weights[chosen] = consequence / feature
                errors[chosen] = 0
        self.weights[chosen] += .15 * errors[chosen] * feature
        self.used[chosen] = True
        self.active = chosen


def evaluate(model, sign, seed, switching=False):
    # Evaluation routing can adapt, but weights never change during evaluation.
    rng = np.random.default_rng(seed)
    active = model.active
    losses, correct = [], []
    for i in range(200):
        context = rng.choice([-1, 1]) if switching else sign
        action = rng.choice([-1, 1]) * rng.uniform(.7, 1)
        target = context * np.tanh(2 * action)
        prediction = model.weights[active] * np.tanh(2 * action)
        losses.append((target - prediction) ** 2)
        correct.append(np.sign(prediction) == np.sign(target))
        if model.mode == 'competitive':
            candidates = np.where(model.used)[0]
            errors = target - model.weights[candidates] * np.tanh(2 * action)
            active = int(candidates[np.argmin(errors ** 2)])
    return {'first_mse': float(losses[0]), 'after_first_mse': float(np.mean(losses[1:])),
            'sign_accuracy': float(np.mean(correct))}


def run(seed, mode):
    rng = np.random.default_rng(seed)
    model = Predictor(mode)
    checkpoints = {}
    for name, sign in [('A', 1), ('B', -1), ('A_return', 1)]:
        for _ in range(400):
            action = rng.choice([-1, 1]) * rng.uniform(.7, 1)
            target = sign * np.tanh(2 * action) + rng.normal(0, .03)
            model.observe(action, target)
        checkpoints[name] = {'weights': model.weights.tolist(),
                             'A': evaluate(model, 1, seed + 1000),
                             'B': evaluate(model, -1, seed + 2000)}
    checkpoints['random_contexts'] = evaluate(model, 1, seed + 3000, switching=True)
    return checkpoints


def main():
    results = {mode: [run(seed, mode) for seed in range(20)]
               for mode in ['shared', 'all_branches', 'competitive']}
    summary = {}
    for mode, runs in results.items():
        summary[mode] = {
            'A_mse_after_A': float(np.mean([r['A']['A']['after_first_mse'] for r in runs])),
            'A_mse_after_B': float(np.mean([r['B']['A']['after_first_mse'] for r in runs])),
            'B_mse_after_B': float(np.mean([r['B']['B']['after_first_mse'] for r in runs])),
            'A_first_mse_after_B': float(np.mean([r['B']['A']['first_mse'] for r in runs])),
            'random_context_accuracy': float(np.mean([r['random_contexts']['sign_accuracy'] for r in runs])),
        }
    output = Path('results/local_prediction_pilot')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps({'summary': summary, 'runs': results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
