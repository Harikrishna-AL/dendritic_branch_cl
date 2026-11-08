"""Offline capacity diagnostic using the same frozen features and five rank-8 branches."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw

OUT = Path("results/mnist_joint_branch_diagnostic")
RATES = (.001, .003, .01)
EPOCHS = (2, 10, 30)


def train(x, y, evaluation, targets, seed, relu, rate, epochs):
    generator = torch.Generator().manual_seed(seed)
    left = nn.Parameter(F.normalize(torch.randn(256, 40, generator=generator), dim=0))
    right = nn.Parameter(torch.zeros(40, 10))
    optimizer = torch.optim.Adam([left, right], lr=rate)

    def forward(inputs):
        hidden = inputs @ left
        return (hidden.relu() if relu else hidden) @ right

    scores = {}
    for epoch in range(1, max(epochs) + 1):
        for ids in torch.randperm(len(y), generator=generator).split(128):
            optimizer.zero_grad()
            F.cross_entropy(forward(x[ids]), y[ids]).backward()
            optimizer.step()
        if epoch in epochs:
            with torch.no_grad():
                scores[epoch] = float((forward(evaluation).argmax(1) == targets).float().mean())
    return scores


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
    assert len(fit) == 55000 and len(held) == 5000

    def project(images, seed):
        weights = torch.randn(784, 256, generator=torch.Generator().manual_seed(seed)) / 8
        return (images @ weights).tanh()

    features = project(x, 113)
    trials = []
    for relu in (False, True):
        for rate in RATES:
            scores = train(features[fit], y[fit], features[held], y[held], 113, relu, rate, EPOCHS)
            trials.extend(dict(relu=relu, rate=rate, epochs=e, accuracy=acc) for e, acc in scores.items())
    chosen = [
        max((t for t in trials if t["relu"] == relu and (budget is None or t["epochs"] == budget)),
            key=lambda t: t["accuracy"])
        for relu in (False, True) for budget in (2, None)
    ]
    (OUT / "validation.json").write_text(json.dumps(dict(trials=trials, selected=chosen), indent=2))
    results = []
    for seed in (101, 102, 103):
        features = project(x[fit], seed)
        evaluated = project(test, seed)
        for setting in chosen:
            e = setting["epochs"]
            accuracy = train(features, y[fit], evaluated, targets, seed, setting["relu"], setting["rate"], [e])[e]
            results.append(dict(setting, seed=seed, accuracy=accuracy))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
