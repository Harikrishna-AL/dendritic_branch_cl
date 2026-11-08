"""Post-training readout and activation swaps on frozen Split MNIST checkpoints."""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from bdh_cl.data.mnist import load_raw

SOURCE = Path("results/matched_numenta_20261031")
OUT = Path("results/mnist_posthoc_soma")
ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu,
               "identity": lambda x: x, "softsign": torch.nn.functional.softsign}
READOUTS = ("current", "mean_branches", "max_branch", "largest_abs_branch",
            "mean_routes", "confident_route", "probability_mixture", "prototype")


@torch.no_grad()
def predict(state, routes, contexts, x):
    preactivation = torch.nn.functional.linear(x, state["first.weight"], state["first.bias"])
    preactivation = preactivation.reshape(len(x), 168, 4)
    neurons = torch.arange(168)
    route_ids = torch.cdist(x, contexts).argmin(1)
    for name, activation in ACTIVATIONS.items():
        responses = activation(preactivation)

        def head(hidden):
            return torch.nn.functional.linear(hidden, state["head.weight"], state["head.bias"])

        current = responses[:, neurons, state["error"].argmin(-1)]
        maximum = responses.max(-1).values
        strongest = responses.gather(2, responses.abs().argmax(-1, keepdim=True)).squeeze(-1)
        bank = torch.stack([head(responses[:, neurons, route.long()]) for route in routes], dim=1)
        probabilities = bank.softmax(-1)
        confident = probabilities.amax(-1).argmax(-1)
        rows = torch.arange(len(x))
        logits = (head(current), head(responses.mean(-1)), head(maximum), head(strongest),
                  bank.mean(1), bank[rows, confident], probabilities.mean(1), bank[rows, route_ids])
        for readout, values in zip(READOUTS, logits):
            yield name, readout, values.argmax(-1)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    raw = load_raw("data_cache/mnist")
    split = np.load(SOURCE / "split_indices.npz")
    val = split["validation"]
    contexts = torch.load(SOURCE / "contexts.pt", weights_only=True)

    def inputs(images):
        return (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780

    datasets = [
        (inputs(raw["train_images"][val]), torch.tensor(raw["train_labels"][val])),
        (inputs(raw["test_images"]), torch.tensor(raw["test_labels"])),
    ]
    checkpoints = []
    for seed in (101, 102, 103):
        for order in ("forward", "reverse"):
            ck = torch.load(SOURCE / f"{seed}_{order}_branch_aux.pt", weights_only=True)
            checkpoints.append((seed, order, ck))
    results = {}
    for phase, (x, y) in zip(("validation", "test"), datasets):
        grouped = defaultdict(list)
        for seed, order, ck in checkpoints:
            for activation, readout, predicted in predict(ck["model"], ck["routes"], contexts, x):
                key = f"{activation}/{readout}"
                if phase == "test" and key not in selected:
                    continue
                grouped[key].append(dict(seed=seed, order=order,
                                         accuracy=float((predicted == y).float().mean())))
        results[phase] = dict(grouped)
        if phase == "validation":
            best = max((k for k in grouped if not k.endswith("/prototype")),
                       key=lambda k: np.mean([z["accuracy"] for z in grouped[k]]))
            best_tanh = max((k for k in grouped if k.startswith("tanh/") and not k.endswith("/prototype")),
                            key=lambda k: np.mean([z["accuracy"] for z in grouped[k]]))
            selected = [best, best_tanh, "tanh/prototype", "tanh/current"]
            results["selected"] = list(dict.fromkeys(selected))
            (OUT / "selection.json").write_text(json.dumps(results, indent=2))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
