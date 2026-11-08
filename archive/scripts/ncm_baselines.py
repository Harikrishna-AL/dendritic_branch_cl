"""Training-free class-incremental baselines: nearest class mean on fixed features.

Why: the dendritic toy reached 0.94 class-incremental accuracy with a stored
prototype readout, but raw-pixel nearest class mean scored 0.995 on the same
toy. Before claiming a learned population code beats forgetting on a real
benchmark, we need the number a method gets with NO learning at all. With
fixed features there is no drift, so a prototype stored at the end of a class's
task stays exact -- these numbers are what "just store class means" buys.

Feature maps (all fixed, never trained):

``raw``        flattened pixels in [0, 1].
``rand_kwta``  random Gaussian projection to N units, binary top-k
               (FlyModel-style sparse expansion).
``rand_dend``  N neurons x K branches, each branch C random input contacts
               with Gaussian weights, soma = max over branches, binary top-k.
               This is the toy's arm F (fixed random weights) at full scale.

Protocol: tasks are disjoint class groups seen in order. After task t, every
class in tasks 0..t has a prototype (cosine class mean of its training
features); test images of all seen classes are classified by argmax over ALL
seen prototypes -- no task id. Reported: final average accuracy (mean over
tasks after the last task) and average incremental accuracy (mean over steps
of accuracy on all classes seen so far).

    python bdh_cl/scripts/ncm_baselines.py --dataset mnist --seeds 0,1,2 \
        --out bdh_cl/results/ncm_baselines/mnist.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


# --------------------------------------------------------------------- data

def load_mnist(cache: str):
    from bdh_cl.data.mnist import load_raw

    d = load_raw(cache)
    x = torch.from_numpy(d["train_images"].reshape(len(d["train_images"]), -1)).float() / 255
    xt = torch.from_numpy(d["test_images"].reshape(len(d["test_images"]), -1)).float() / 255
    y = torch.from_numpy(d["train_labels"].astype(np.int64))
    yt = torch.from_numpy(d["test_labels"].astype(np.int64))
    tasks = [[2 * i, 2 * i + 1] for i in range(5)]
    return x, y, xt, yt, tasks


def load_cifar100(root: str, n_tasks: int = 10):
    from bdh_cl.data import cifar100

    (x, y), _valid, (xt, yt), _order = cifar100.load(root)
    # Labels are already remapped so class c belongs to task c // per_task.
    x = torch.from_numpy(x.reshape(len(x), -1)).float() / 255
    xt = torch.from_numpy(xt.reshape(len(xt), -1)).float() / 255
    per = 100 // n_tasks
    tasks = [list(range(i * per, (i + 1) * per)) for i in range(n_tasks)]
    return x, torch.from_numpy(y).long(), xt, torch.from_numpy(yt).long(), tasks


# ----------------------------------------------------------------- features

def kwta(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Binary top-k: exactly k ones per row."""
    out = torch.zeros_like(scores)
    out.scatter_(1, scores.topk(k, dim=1).indices, 1.0)
    return out


class RandKWTA:
    def __init__(self, d_in, n, k, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(d_in, n, generator=g).to(device) / d_in ** 0.5
        self.k = k

    def __call__(self, x):
        # Centre each image so the projection is not dominated by brightness.
        x = x - x.mean(dim=1, keepdim=True)
        return kwta(x @ self.w, self.k)


class RandDend:
    def __init__(self, d_in, n, k, n_branch, contacts, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randint(0, d_in, (n, n_branch, contacts), generator=g).to(device)
        self.w = (torch.randn(n, n_branch, contacts, generator=g) / contacts ** 0.5).to(device)
        self.k = k

    def __call__(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        branch = (x[:, self.idx] * self.w).sum(-1)       # B, N, K
        return kwta(branch.amax(-1), self.k)


def featurize(fn, x, device, bs=1024):
    # The dendritic gather materialises B x N x K x C floats; keep it ~64M.
    if isinstance(fn, RandDend):
        bs = max(8, min(bs, int(6.4e7 // fn.idx.numel())))
    return torch.cat([fn(x[i : i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


# --------------------------------------------------------------------- eval

def class_means(f: torch.Tensor, y: torch.Tensor, classes) -> torch.Tensor:
    f = F.normalize(f, dim=-1)
    return torch.stack([f[y == c].mean(0) for c in classes])


def ncm_class_incremental(f, y, ft, yt, tasks):
    """Accuracy matrix acc[t][j]: test accuracy on task j after task t (j <= t).

    Prototypes are added task by task and never revised; with fixed features
    the frozen and refreshed protocols coincide.
    """
    seen, protos, acc = [], [], []
    ftn = F.normalize(ft, dim=-1)
    for t, cls in enumerate(tasks):
        seen += cls
        protos.append(class_means(f, y, cls))
        P = F.normalize(torch.cat(protos), dim=-1)
        pred = torch.tensor(seen)[(ftn @ P.T).argmax(1)]
        row = []
        for j in range(t + 1):
            m = torch.isin(yt, torch.tensor(tasks[j]))
            row.append(float((pred[m] == yt[m]).float().mean()))
        acc.append(row)
    return acc


def summarize(acc):
    final = float(np.mean(acc[-1]))
    avg_inc = float(np.mean([np.mean(r) for r in acc]))
    return {"final_avg_acc": final, "avg_incremental_acc": avg_inc}


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar100"], required=True)
    ap.add_argument("--mnist-cache", default="data_cache/mnist")
    ap.add_argument("--cifar-root", default="data_cache/cifar100")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sizes", default="2000,10000", help="N units for random maps")
    ap.add_argument("--sparsity", type=float, default=0.05, help="k = sparsity * N")
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--contacts", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap train/test size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "mnist":
        x, y, xt, yt, tasks = load_mnist(args.mnist_cache)
    else:
        x, y, xt, yt, tasks = load_cifar100(args.cifar_root)
    if args.limit:
        g = torch.Generator().manual_seed(0)
        i, it = torch.randperm(len(x), generator=g)[: args.limit], torch.randperm(len(xt), generator=g)[: args.limit]
        x, y, xt, yt = x[i], y[i], xt[it], yt[it]
    print(f"{args.dataset}: train {tuple(x.shape)} test {tuple(xt.shape)} tasks {len(tasks)} device {device}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [("raw", None)]
    for n in [int(s) for s in args.sizes.split(",")]:
        k = max(1, int(round(args.sparsity * n)))
        arms.append((f"rand_kwta_N{n}_k{k}", lambda s, n=n, k=k: RandKWTA(x.shape[1], n, k, s, device)))
        arms.append((
            f"rand_dend_N{n}_k{k}_K{args.branches}_C{args.contacts}",
            lambda s, n=n, k=k: RandDend(x.shape[1], n, k, args.branches, args.contacts, s, device),
        ))

    report = {"dataset": args.dataset, "args": vars(args), "tasks": tasks, "arms": {}}
    for name, make in arms:
        runs = []
        for s in seeds if make else seeds[:1]:   # raw is deterministic
            t0 = time.time()
            fn = make(s) if make else (lambda b: b)
            f, ft = featurize(fn, x, device), featurize(fn, xt, device)
            acc = ncm_class_incremental(f, y, ft, yt, tasks)
            runs.append({"seed": s, "acc_matrix": acc, **summarize(acc)})
            print(f"{name} seed{s}: final={runs[-1]['final_avg_acc']:.4f} "
                  f"avg_inc={runs[-1]['avg_incremental_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        fin = [r["final_avg_acc"] for r in runs]
        inc = [r["avg_incremental_acc"] for r in runs]
        report["arms"][name] = {
            "runs": runs,
            "final_mean": float(np.mean(fin)), "final_std": float(np.std(fin)),
            "avg_inc_mean": float(np.mean(inc)), "avg_inc_std": float(np.std(inc)),
        }
        a = report["arms"][name]
        print(f"== {name}: final {a['final_mean']:.4f}±{a['final_std']:.4f} "
              f"avg_inc {a['avg_inc_mean']:.4f}±{a['avg_inc_std']:.4f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
