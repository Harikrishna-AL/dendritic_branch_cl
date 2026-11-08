"""Compare strongest-response training with the original error-trace training."""
import json
from pathlib import Path

import numpy as np
import torch

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import NEURONS, SOURCE, model_with_branches, train_pair
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier

OUT = Path("results/mnist_train_strongest")


class StrongestBranchClassifier(BranchClassifier):
    def forward(self, x, counterfactual=False):
        branches = self.first(x).reshape(-1, self.neurons, self.branches)
        if self.use_tanh:
            branches = branches.tanh()
        winner = branches.abs().argmax(-1, keepdim=True)
        hidden = branches.gather(-1, winner).squeeze(-1)
        logits = self.head(hidden)
        if not counterfactual:
            return logits
        delta = branches - hidden[..., None]
        return logits, logits[:, None, None, :] + delta[..., None] * self.head.weight.T[None, :, None, :]

    def observe(self, error):
        # Branch selection uses the input response and never reads an error trace.
        pass


@torch.no_grad()
def evaluate(model, x, y, order, use_tanh):
    correct = []
    for batch in x.split(256):
        responses = model.first(batch).reshape(-1, NEURONS, 2)
        if use_tanh:
            responses = responses.tanh()
        winner = responses.abs().argmax(-1, keepdim=True)
        logits = model.head(responses.gather(-1, winner).squeeze(-1))
        correct.append(logits.argmax(-1))
    hits = torch.cat(correct) == y
    return dict(overall=float(hits.float().mean()),
                pairs=[float(hits[y // 2 == pair].float().mean()) for pair in order])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    raw = load_raw("data_cache/mnist")
    split = np.load(SOURCE / "split_indices.npz")

    def dataset(images, labels):
        x = (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780
        return x, torch.tensor(labels, dtype=torch.long)

    fit = split["train"]
    x, y = dataset(raw["train_images"][fit], raw["train_labels"][fit])
    tx, ty = dataset(raw["test_images"], raw["test_labels"])
    results = []
    for seed in (101, 102, 103):
        for order in (list(range(5)), list(reversed(range(5)))):
            for arm in ("trace", "strongest_tanh", "strongest_linear"):
                torch.manual_seed(seed)
                if arm == "trace":
                    model = model_with_branches(2)
                else:
                    model = StrongestBranchClassifier("local", neurons=NEURONS, branches=2, gate="hard")
                    model.use_tanh = arm == "strongest_tanh"
                optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
                stages = []
                for stage, pair in enumerate(order):
                    mask = y // 2 == pair
                    train_pair(model, optimizer, x[mask], y[mask], seed, stage)
                    stages.append({activation: evaluate(model, tx, ty, order, activation == "tanh")
                                   for activation in ("linear", "tanh")})
                row = dict(seed=seed, order=order, arm=arm, stages=stages)
                results.append(row)
                (OUT / "results.json").write_text(json.dumps(results, indent=2))
                print(seed, order, arm, {k: round(v["overall"] * 100, 2) for k, v in stages[-1].items()},
                      flush=True)


if __name__ == "__main__":
    main()
