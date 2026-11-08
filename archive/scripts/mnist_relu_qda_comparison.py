"""Compare branch activations and QDA on identical frozen MNIST features."""
import json
from pathlib import Path

import numpy as np
import torch

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_nullspace_pilot import run

OUT = Path("results/mnist_relu_qda_20261104")
ORDERS = [list(range(5)), list(reversed(range(5)))]


def qda(features, labels, evaluated, targets, order, shrink):
    scores = torch.full((len(evaluated), 10), -torch.inf, dtype=torch.float64)
    evaluated = evaluated.double()
    acquisition = []
    for pair in order:
        for digit in (2 * pair, 2 * pair + 1):
            x = features[labels == digit].double()
            centered = x - x.mean(0)
            covariance = centered.T @ centered / (len(x) - 1)
            covariance = (1 - shrink) * covariance + shrink * covariance.diag().mean() * torch.eye(x.shape[1])
            factor = torch.linalg.cholesky(covariance)
            residual = torch.linalg.solve_triangular(factor, (evaluated - x.mean(0)).T, upper=False)
            scores[:, digit] = -.5 * residual.square().sum(0) - factor.diag().log().sum() + np.log(len(x))
        mask = targets // 2 == pair
        acquisition.append(float((scores[mask].argmax(1) == targets[mask]).float().mean()))
    predictions = scores.argmax(1)
    final_pairs = [float((predictions[targets // 2 == pair] == targets[targets // 2 == pair]).float().mean())
                   for pair in order]
    return dict(cumulative_accuracy=float((predictions == targets).float().mean()),
                acquisition=acquisition, final_pairs=final_pairs,
                earlier_drop=float(np.mean(np.array(acquisition[:-1]) - final_pairs[:-1])))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=False)
    raw = load_raw("data_cache/mnist")
    x = torch.tensor(raw["train_images"].copy()).float().reshape(-1, 784) / 255
    y = torch.tensor(raw["train_labels"].copy()).long()
    test = torch.tensor(raw["test_images"].copy()).float().reshape(-1, 784) / 255
    targets = torch.tensor(raw["test_labels"].copy()).long()
    rng = np.random.default_rng(20261101)
    held = np.concatenate([rng.permutation(np.flatnonzero(y.numpy() == c))[:500] for c in range(10)])
    fit = np.setdiff1d(np.arange(len(y)), held)
    trials, results = [], []
    for width in (50, 256):
        for seed in (113, 101, 102, 103):
            projection = torch.randn(784, 256, generator=torch.Generator().manual_seed(seed))[:, :width] / 8
            features = (x[fit] @ projection).tanh()
            evaluated, labels = ((x[held] @ projection).tanh(), y[held]) if seed == 113 else (
                (test @ projection).tanh(), targets)
            for method, candidates in [("linear", [0, .01, .1, 1, 10]),
                                       ("relu", [0, .01, .1, 1, 10]),
                                       ("qda", [.001, .01, .1, .5])]:
                if seed != 113:
                    candidates = [max(candidates, key=lambda c: np.mean([
                        z["cumulative_accuracy"] for z in trials
                        if (z["width"], z["method"], z["strength"]) == (width, method, c)]))]
                for strength in candidates:
                    for order in ORDERS:
                        row = (qda(features, y[fit], evaluated, labels, order, strength) if method == "qda"
                               else run(features, y[fit], evaluated, labels, order, seed, method, strength))
                        row.update(width=width, seed=seed, order=order, method=method, strength=strength)
                        (trials if seed == 113 else results).append(row)
    for name, rows in [("validation", trials), ("results", results)]:
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
