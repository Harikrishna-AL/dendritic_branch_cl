"""Paired Split MNIST pilot of additive low-rank adapters on frozen features."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw

OUT = Path("results/mnist_nullspace_pilot_geometry_20261103")
WIDTH, RANK, BASIS = 256, 8, 64


def run(features, labels, test, targets, order, seed, arm, penalty=0.):
    width = features.shape[1]
    gram = torch.zeros(width, width, dtype=torch.float64)
    old_count = 0
    weights = torch.zeros(width, 10)
    nonlinear_branches = []
    prefixes, centers, acquisition, spectra = [], [], [], []
    for stage, pair in enumerate(order):
        x, y = features[labels // 2 == pair], labels[labels // 2 == pair]
        generator = torch.Generator().manual_seed(seed * 100 + stage)
        initial = torch.randn(width, RANK, generator=generator)
        basis = None
        if arm == "random_basis" and stage:
            basis = torch.linalg.qr(torch.randn(WIDTH, BASIS, generator=torch.Generator().manual_seed(10000 + seed * 100 + stage))).Q
            initial = basis.T @ initial
        if arm == "null_space" and stage:
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            basis = eigenvectors[:, :BASIS].float()
            spectra.append(float(eigenvalues[:BASIS].sum() / eigenvalues.sum()))
            initial = basis.T @ initial
        a = torch.nn.Parameter(F.normalize(initial, dim=0))
        b = torch.nn.Parameter(torch.zeros(RANK, 10))
        old_second_moment = (gram / old_count).float() if penalty and old_count else None
        optimizer = torch.optim.Adam([a, b], lr=.003)
        for _ in range(2):
            for indices in torch.randperm(len(x), generator=generator).split(128):
                optimizer.zero_grad()
                inputs = x[indices]
                projected = inputs if basis is None else inputs @ basis
                hidden = projected @ a
                if arm == "relu":
                    with torch.no_grad():
                        prior = sum((inputs @ left).relu() @ right for left, right in nonlinear_branches)
                    loss = F.cross_entropy(prior + hidden.relu() @ b, y[indices])
                else:
                    loss = F.cross_entropy(inputs @ weights + hidden @ b, y[indices])
                if old_second_moment is not None:
                    # Mean squared score change across earlier inputs and classes.
                    update = a @ b
                    if arm == "relu":
                        # ReLU cannot increase the hidden vector norm. The
                        # squared Frobenius norm bounds the output amplification.
                        drift = (a * (old_second_moment @ a)).sum() * b.square().sum()
                    else:
                        drift = (update * (old_second_moment @ update)).sum()
                    loss = loss + penalty * drift / 10
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            weights = weights + (a if basis is None else basis @ a) @ b
            if arm == "relu":
                nonlinear_branches.append((a.detach(), b.detach()))
                prefixes.append(sum((test @ left).relu() @ right for left, right in nonlinear_branches))
            else:
                prefixes.append(test @ weights)
            centers.append(x.mean(0))
            mask = targets // 2 == pair
            acquisition.append(float((prefixes[-1][mask].argmax(1) == targets[mask]).float().mean()))
            gram += x.double().T @ x.double()
            old_count += len(x)
    logits = torch.stack(prefixes, dim=1)
    rows = torch.arange(len(test))
    # Each prefix differs from its predecessor by exactly one adapter.
    responses = torch.diff(logits, dim=1, prepend=torch.zeros_like(logits[:, :1]))
    geometry = responses.norm(dim=2).argmax(dim=1)
    oracle = torch.tensor(order).argsort()[targets // 2]
    inferred = torch.cdist(test, torch.stack(centers)).argmin(1)
    final_pairs = [float((logits[targets // 2 == pair, -1].argmax(1)
                          == targets[targets // 2 == pair]).float().mean()) for pair in order]
    return dict(seed=seed, order=order, arm=arm, penalty=penalty, acquisition=acquisition,
                final_pairs=final_pairs, spectrum_fraction=spectra,
                cumulative_accuracy=float((logits[:, -1].argmax(1) == targets).float().mean()),
                prototype_accuracy=float((logits[rows, inferred].argmax(1) == targets).float().mean()),
                geometry_accuracy=float((logits[rows, geometry].argmax(1) == targets).float().mean()),
                geometry_route_accuracy=float((geometry == oracle).float().mean()),
                prototype_route_accuracy=float((inferred == oracle).float().mean()),
                oracle_prefix_accuracy=float((logits[rows, oracle].argmax(1) == targets).float().mean()),
                earlier_drop=float(np.mean(np.array(acquisition[:-1]) - final_pairs[:-1])))


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", action="store_true", help="Tune and evaluate the old-score penalty.")
    args = parser.parse_args()
    out = Path("results/mnist_shared_features_20261104") if args.soft else OUT
    out.mkdir(parents=True, exist_ok=False)
    raw = load_raw("data_cache/mnist")
    x = torch.tensor(raw["train_images"].copy()).float().reshape(-1, 784) / 255
    y = torch.tensor(raw["train_labels"].copy()).long()
    test = torch.tensor(raw["test_images"].copy()).float().reshape(-1, 784) / 255
    targets = torch.tensor(raw["test_labels"].copy()).long()
    rng = np.random.default_rng(20261101)
    held = np.concatenate([rng.permutation(np.flatnonzero(y.numpy() == c))[:500] for c in range(10)])
    fit = np.setdiff1d(np.arange(len(y)), held)
    orders = [list(range(5)), list(reversed(range(5)))]
    settings = [(arm, 0.) for arm in ["unconstrained", "random_basis", "null_space"]]
    if args.soft:
        seed = 113
        projection = torch.randn(784, WIDTH, generator=torch.Generator().manual_seed(seed)) / 8
        fit_features = (x[fit] @ projection).tanh()
        validation_features = (x[held] @ projection).tanh()
        trials = [
            run(fit_features, y[fit], validation_features, y[held], order, seed, "soft", penalty)
            for penalty in [.1, 1., 10., 100., 1000.] for order in orders
        ]
        selected = max(
            sorted({row["penalty"] for row in trials}),
            key=lambda strength: np.mean([row["cumulative_accuracy"] for row in trials
                                          if row["penalty"] == strength]),
        )
        (out / "validation.json").write_text(json.dumps({"selected": selected, "trials": trials}, indent=2))
        settings.append(("soft", selected))
    results = []
    for seed in [101, 102, 103]:
        projection = torch.randn(784, WIDTH, generator=torch.Generator().manual_seed(seed)) / 8
        features, evaluated = (x[fit] @ projection).tanh(), (test @ projection).tanh()
        for order in orders:
            for arm, penalty in settings:
                row = run(features, y[fit], evaluated, targets, order, seed, arm, penalty)
                results.append(row)
                print(row, flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
