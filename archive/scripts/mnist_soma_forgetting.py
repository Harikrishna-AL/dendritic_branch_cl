"""Measure forgetting with a fixed, prototype-free soma at every stage."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.matched_numenta_comparison import train

SOURCE = Path("results/matched_numenta_20261031")
OUT = Path("results/mnist_soma_forgetting")


@torch.no_grad()
def evaluate(model, images, labels, order):
    predictions = []
    for batch in images.split(256):
        responses = model.first(batch).reshape(len(batch), 168, 4)
        winners = responses.abs().argmax(-1, keepdim=True)
        predictions.append(model.head(responses.gather(-1, winners).squeeze(-1)).argmax(-1))
    correct = torch.cat(predictions) == labels
    return [float(correct[labels // 2 == pair].float().mean()) for pair in order], float(correct.float().mean())


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    raw = load_raw("data_cache/mnist")
    split = np.load(SOURCE / "split_indices.npz")
    fit, val = split["train"], split["validation"]

    def standardize(images):
        return (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - .13062755) / .30810780

    x, vx, tx = [standardize(images) for images in
                 (raw["train_images"][fit], raw["train_images"][val], raw["test_images"])]
    y, vy, ty = [torch.tensor(labels, dtype=torch.long) for labels in
                 (raw["train_labels"][fit], raw["train_labels"][val], raw["test_labels"])]
    indices = [torch.where(y // 2 == pair)[0] for pair in range(5)]
    contexts = torch.load(SOURCE / "contexts.pt", weights_only=True)
    rate = json.loads((SOURCE / "selected_hyperparameters.json").read_text())["branch_aux"]
    runs = []
    for seed in (101, 102, 103):
        for order in (list(range(5)), list(reversed(range(5)))):
            stages = []

            def record(model, stage):
                pair_accuracy, overall = evaluate(model, tx, ty, order)
                stages.append(dict(stage=stage + 1, pair_accuracy=pair_accuracy, overall=overall))

            train("branch_aux", seed, rate, order, x, y, vx, vy, contexts, indices, on_stage=record)
            runs.append(dict(seed=seed, order=order, stages=stages))
            print(seed, order, stages[-1]["overall"], flush=True)
    (OUT / "results.json").write_text(json.dumps(runs, indent=2))

    plot(runs)


def plot(runs):
    matrix = 100 * np.array([[stage["pair_accuracy"] for stage in row["stages"]] for row in runs])
    seed_means = matrix.reshape(3, 2, 5, 5).mean(1)
    mean = seed_means.mean(0)
    sd = seed_means.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in range(5):
        steps = np.arange(pair + 1, 6)
        values = mean[pair:, pair]
        ax.plot(steps, values, "o-", label=f"Pair learned {pair + 1}")
        ax.fill_between(steps, np.clip(values - sd[pair:, pair], 0, 100),
                        np.clip(values + sd[pair:, pair], 0, 100), alpha=.12)
    ax.set(xlabel="Number of digit pairs trained", ylabel="Digit accuracy on each pair (%)",
           title="Split MNIST forgetting with input-dependent branch selection",
           xticks=range(1, 6), ylim=(0, 103))
    ax.legend(loc="lower left", ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "forgetting_curve.png", dpi=180)
    print("Acquisition", mean.diagonal(), "Final", mean[-1])
    print("Mean earlier-pair drop", (mean.diagonal()[:-1] - mean[-1, :-1]).mean())
    print("Final overall", 100 * np.mean([r["stages"][-1]["overall"] for r in runs]))


if __name__ == "__main__":
    main()
