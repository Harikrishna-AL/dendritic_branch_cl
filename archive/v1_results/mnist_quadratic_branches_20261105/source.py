"""Matched pilot of low-rank quadratic interactions inside dendritic branches."""
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE, train_pair
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier

OUT = Path("results/mnist_quadratic_branches_20261105")
ARMS = ("original", "normalized", "extra_linear", "quadratic")


class InteractionBranches(BranchClassifier):
    def __init__(self, arm):
        super().__init__("local", neurons=168, branches=2, gate="hard")
        self.arm_name = arm
        if arm in ("extra_linear", "quadratic"):
            self.u = nn.Linear(784, 336, bias=False)
            self.v = nn.Linear(784, 336, bias=False)
            self.alpha = nn.Parameter(torch.zeros(336))

    def responses(self, x, bounded=True):
        if self.arm_name != "original":
            x = F.layer_norm(x, (784,))
        response = self.first(x)
        if self.arm_name == "quadratic":
            response = response + self.alpha * self.u(x) * self.v(x)
        elif self.arm_name == "extra_linear":
            response = response + self.alpha * (self.u(x) + self.v(x)) / math.sqrt(2)
        response = response.reshape(-1, self.neurons, self.branches)
        return response.tanh() if bounded else response

    def forward(self, x, counterfactual=False):
        branches = self.responses(x)
        hidden = (branches * self.gates().detach().clone()).sum(-1)
        logits = self.head(hidden)
        if not counterfactual:
            return logits
        delta = branches - hidden[..., None]
        candidates = logits[:, None, None, :] + delta[..., None] * self.head.weight.T[None, :, None, :]
        return logits, candidates


@torch.no_grad()
def evaluate(model, x, y, order, routes):
    counts = {mode: torch.zeros(5) for mode in ("strongest_linear", "strongest_tanh", "oracle")}
    total = torch.bincount(y // 2, minlength=5)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for bx, by in zip(x.split(256), y.split(256)):
        raw = model.responses(bx, bounded=False)
        for mode in counts:
            responses = raw if mode == "strongest_linear" else raw.tanh()
            choices = routes[by // 2].long() if mode == "oracle" else responses.abs().argmax(-1)
            hidden = responses.gather(-1, choices[..., None]).squeeze(-1)
            predicted = model.head(hidden).argmax(-1)
            counts[mode] += torch.bincount((by // 2)[predicted == by], minlength=5)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    return {mode: dict(overall=float(hits.sum() / len(y)),
                       pairs=[float((hits / total)[pair]) for pair in order])
            for mode, hits in counts.items()}


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    raw = load_raw("data_cache/mnist")
    split = np.load(SOURCE / "split_indices.npz")
    assert not np.intersect1d(split["train"], split["validation"]).size

    def dataset(images, labels):
        inputs = (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780
        return inputs, torch.tensor(labels.copy(), dtype=torch.long)

    x, y = dataset(raw["train_images"][split["train"]], raw["train_labels"][split["train"]])
    tx, ty = dataset(raw["test_images"], raw["test_labels"])
    protocol = dict(
        arms=ARMS, seeds=[101, 102, 103], orders=["forward", "reverse"],
        epochs_per_pair=2, batch_size=128, learning_rate=.0005,
        optimizer="Continuous Adam, clip5", training_loss="Selected CE + .5 mean single-neuron substitution CE",
        inference="Image-only strongest absolute branch response, preserving sign; linear and tanh responses reported. Oracle uses true pair only to select saved branch indices.",
        normalization="Per-image LayerNorm without affine parameters before first projections, for every arm except original.",
        interaction="alpha * (u dot x) * (v dot x), alpha starts at zero. Extra-linear control uses alpha * (u dot x + v dot x)/sqrt2 with same parameter count.",
        limits="One fixed training recipe; no per-arm tuning. Low-rank quadratic adaptation of the paper, raw-pixel MLP, not its CNN implementation. Training uses known boundaries for archives, which only oracle inference accesses.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())
    results = []
    for seed in (101, 102, 103):
        for order in (list(range(5)), list(reversed(range(5)))):
            for arm in ARMS:
                torch.manual_seed(seed)
                model = InteractionBranches(arm)
                optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
                routes = torch.zeros(5, 168, dtype=torch.uint8)
                stages = []
                for stage, pair in enumerate(order):
                    mask = y // 2 == pair
                    train_pair(model, optimizer, x[mask], y[mask], seed, stage)
                    routes[pair] = model.gates().argmax(-1).to(torch.uint8)
                    scores = evaluate(model, tx, ty, order, routes)
                    assert all(np.isfinite(values["overall"]) for values in scores.values())
                    stages.append(scores)
                row = dict(seed=seed, order=order, arm=arm, stages=stages,
                           parameters=sum(p.numel() for p in model.parameters()))
                results.append(row)
                (OUT / "results.json").write_text(json.dumps(results, indent=2))
                print(seed, order, arm,
                      {k: round(v["overall"] * 100, 2) for k, v in stages[-1].items()}, flush=True)
    summary = []
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        for mode in ("strongest_linear", "strongest_tanh", "oracle"):
            seed_scores = [statistics.mean(row["stages"][-1][mode]["overall"] * 100
                                           for row in rows if row["seed"] == seed)
                           for seed in (101, 102, 103)]
            drops = [statistics.mean(row["stages"][stage][mode]["pairs"][stage] -
                                     row["stages"][-1][mode]["pairs"][stage]
                                     for stage in range(4)) * 100 for row in rows]
            summary.append(dict(arm=arm, mode=mode, accuracy=statistics.mean(seed_scores),
                                seed_sd=statistics.stdev(seed_scores),
                                mean_acquisition_to_final_drop=statistics.mean(drops),
                                parameters=rows[0]["parameters"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
