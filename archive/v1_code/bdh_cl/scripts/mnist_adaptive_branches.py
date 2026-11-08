"""Grow the Split MNIST branch model only when held-out accuracy justifies it."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier

OUT = Path("results/mnist_adaptive_branches_continuous_adam")
SOURCE = Path("results/matched_numenta_20261031")
NEURONS = 168
MAX_BRANCHES = 4
MIN_GAIN = .01


def model_with_branches(count):
    return BranchClassifier("local", neurons=NEURONS, branches=count, gate="hard")


@torch.no_grad()
def grow(model, optimizer, seed):
    torch.manual_seed(seed)
    old = model.branches
    new = model_with_branches(old + 1)
    new.first.weight.view(NEURONS, old + 1, 784)[:, :old].copy_(
        model.first.weight.view(NEURONS, old, 784))
    new.first.bias.view(NEURONS, old + 1)[:, :old].copy_(model.first.bias.view(NEURONS, old))
    new.head.load_state_dict(model.head.state_dict())
    new.error[:, :old].copy_(model.error)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=.0005)
    for (name, old_parameter), new_parameter in zip(model.named_parameters(), new.parameters()):
        if old_parameter not in optimizer.state:
            continue
        state = {}
        for key, value in optimizer.state[old_parameter].items():
            if name.startswith("first.") and value.shape == old_parameter.shape:
                expanded = torch.zeros_like(new_parameter)
                expanded.view(NEURONS, old + 1, -1)[:, :old].copy_(value.view(NEURONS, old, -1))
                state[key] = expanded
            else:
                state[key] = value.clone()
        new_optimizer.state[new_parameter] = state
    return new, new_optimizer


def train_pair(model, optimizer, images, labels, seed, stage):
    for epoch in range(2):
        generator = torch.Generator().manual_seed(seed * 100000 + stage * 100 + epoch)
        for ids in torch.randperm(len(labels), generator=generator).split(128):
            optimizer.zero_grad()
            logits, alternatives = model(images[ids], counterfactual=True)
            targets = labels[ids, None, None].expand(alternatives.shape[:-1]).reshape(-1)
            errors = F.cross_entropy(alternatives.reshape(-1, 10), targets, reduction="none")
            errors = errors.reshape(alternatives.shape[:-1])
            loss = F.cross_entropy(logits, labels[ids]) + .5 * errors.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            model.observe(errors.detach().mean(0))


@torch.no_grad()
def scores(model, images, labels, pairs):
    correct = []
    for x, y in zip(images.split(256), labels.split(256)):
        branches = model.first(x).reshape(len(x), NEURONS, model.branches)
        selected = branches.abs().argmax(-1, keepdim=True)
        logits = model.head(branches.gather(-1, selected).squeeze(-1))
        correct.append(logits.argmax(-1) == y)
    correct = torch.cat(correct)
    return [float(correct[labels // 2 == pair].float().mean()) for pair in pairs], float(correct.float().mean())


def run(arm, seed, order, fit, validation, test):
    torch.manual_seed(seed)
    model = model_with_branches(1 if arm == "adaptive" else int(arm))
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    x, y = fit
    records = []
    for stage, pair in enumerate(order):
        current = y // 2 == pair
        seen = order[:stage + 1]
        candidate, candidate_optimizer = copy.deepcopy((model, optimizer))
        train_pair(candidate, candidate_optimizer, x[current], y[current], seed, stage)
        same = float(np.mean(scores(candidate, *validation, seen)[0]))
        trials = [{"branches": candidate.branches, "seen_validation": same}]
        # Use the first size that yields a meaningful gain. Capacity cannot shrink.
        if arm == "adaptive":
            expanded, expanded_optimizer = model, optimizer
            while expanded.branches < MAX_BRANCHES:
                expanded, expanded_optimizer = grow(expanded, expanded_optimizer, 100000 + seed * 100 + stage * 10 + expanded.branches)
                trial, trial_optimizer = copy.deepcopy((expanded, expanded_optimizer))
                train_pair(trial, trial_optimizer, x[current], y[current], seed, stage)
                expanded_score = float(np.mean(scores(trial, *validation, seen)[0]))
                trials.append({"branches": trial.branches, "seen_validation": expanded_score})
                if expanded_score >= same + MIN_GAIN:
                    candidate, candidate_optimizer = trial, trial_optimizer
                    break
        model, optimizer = candidate, candidate_optimizer
        pair_scores, overall = scores(model, *test, order)
        records.append(dict(pair=pair, branches=model.branches, validation_trials=trials,
                            pair_accuracy=pair_scores, overall=overall,
                            parameters=sum(p.numel() for p in model.parameters())))
    return dict(arm=arm, seed=seed, order=order, stages=records)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    raw = load_raw("data_cache/mnist")
    split = np.load(SOURCE / "split_indices.npz")

    def dataset(images, labels):
        x = (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780
        return x, torch.tensor(labels, dtype=torch.long)

    fit, validation = [
        dataset(raw["train_images"][split[key]], raw["train_labels"][split[key]])
        for key in ("train", "validation")
    ]
    test = dataset(raw["test_images"], raw["test_labels"])
    results = []
    for seed in (101, 102, 103):
        for order in (list(range(5)), list(reversed(range(5)))):
            for arm in ("adaptive", "1", "2", "3", "4"):
                row = run(arm, seed, order, fit, validation, test)
                results.append(row)
                print(arm, seed, order, [s["branches"] for s in row["stages"]],
                      round(100 * row["stages"][-1]["overall"], 2), flush=True)
                (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
