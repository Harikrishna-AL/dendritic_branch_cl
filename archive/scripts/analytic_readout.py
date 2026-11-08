"""Exp21: untrained population code + analytic (gradient-free) class-incremental readouts.

Why: Exp20d showed every gradient-trained arm on frozen ImageNet ResNet-18
features loses to nearest class mean on those features (0.543 valid). In this
regime the competitors accumulate statistics and never overwrite old classes.
Question: is the binary dendritic population a better EXPANSION for such
readouts than raw features or a ReLU random projection (RanPAC-style)?

Feature maps (fixed, never trained; seed = random weights):
  raw        512-d ResNet features
  relu_M     ReLU(x @ W), W Gaussian 512 x M                   (RanPAC-like)
  pop_N_kK   untrained ``PopLayer`` from pop_cl_cifar: N neurons x K dense
             branches, max over branches, binary top-k          (the user's code)
Readouts (all exact streaming statistics, so after task t each equals its own
joint solution on tasks 0..t -- old classes are never overwritten):
  ncm    cosine nearest class mean
  slda   shared within-class covariance + class means, shrinkage eps
  ridge  W = (G + lam I)^-1 C over [phi, 1], one-hot targets      (RanPAC)
Shrinkage / lambda are picked from a grid by FINAL accuracy on the validation
split; the test matrix at that setting is reported. Test never selects.

    python bdh_cl/scripts/analytic_readout.py --feats data_cache/cifar100_r18_feats.pt \
        --maps raw,relu_8000,pop_8000_k160_K4 --seeds 0,1,2 --out results/exp21.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bdh_cl.scripts.pop_cl_cifar import PopLayer  # noqa: E402

GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # x mean eigenvalue


# ----------------------------------------------------------------- maps

def make_map(spec: str, d_in: int, seed: int, device):
    parts = spec.split("_")
    g = torch.Generator().manual_seed(seed)
    if parts[0] == "raw":
        return lambda x: x
    if parts[0] == "relu":
        m = int(parts[1])
        w = (torch.randn(d_in, m, generator=g) / d_in ** 0.5).to(device)
        return lambda x: F.relu(x @ w)
    if parts[0] == "pop":   # pop_N_kK_KB
        n, k, b = int(parts[1]), int(parts[2][1:]), int(parts[3][1:])
        torch.manual_seed(seed)
        layer = PopLayer(d_in, n, b, k, temp=0.5).to(device).eval()
        # forward is hard + soft - soft.detach(); round away the float residue
        return lambda x: (layer(x).detach() > 0.5).float()
    raise ValueError(spec)


def apply(fn, x, bs=2048):
    with torch.no_grad():
        return torch.cat([fn(x[i : i + bs]) for i in range(0, len(x), bs)])


# ---------------------------------------------------------------- stats

class Stats:
    """Streaming sufficient statistics over [phi, 1], float64."""

    def __init__(self, d, n_classes, device):
        self.G = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
        self.S = torch.zeros(n_classes, d + 1, dtype=torch.float64, device=device)   # class sums
        self.n = torch.zeros(n_classes, dtype=torch.float64, device=device)
        self.cos = torch.zeros(n_classes, d, dtype=torch.float64, device=device)     # sums of unit vectors

    def add(self, phi, y, bs=4096):
        for i in range(0, len(phi), bs):
            p = phi[i : i + bs].double()
            pa = torch.cat([p, torch.ones(len(p), 1, dtype=p.dtype, device=p.device)], 1)
            self.G += pa.T @ pa
            self.S.index_add_(0, y[i : i + bs], pa)
            self.n += torch.bincount(y[i : i + bs], minlength=len(self.n)).double()
            self.cos.index_add_(0, y[i : i + bs], F.normalize(p, dim=1))


def aug(phi):
    return torch.cat([phi.double(), torch.ones(len(phi), 1, dtype=torch.float64, device=phi.device)], 1)


def predict_ncm(st, seen, phi):
    P = F.normalize(st.cos[seen], dim=1)
    return torch.as_tensor(seen, device=phi.device)[(F.normalize(phi.double(), dim=1) @ P.T).argmax(1)]


def ridge_weights(st, seen):
    """W(lam) for each lam in GRID via one eigendecomposition of G."""
    evals, Q = torch.linalg.eigh(st.G)
    QC = Q.T @ st.S[seen].T                       # targets: one-hot -> C = S^T
    scale = evals.clamp(min=0).mean()
    return {a: Q @ (QC / (evals.clamp(min=0) + a * scale)[:, None]) for a in GRID}


def slda_weights(st, seen):
    """Shared within-class covariance over the seen classes (bias column dropped)."""
    d = st.G.shape[0] - 1
    n, mu = st.n[seen], st.S[seen, :d] / st.n[seen, None]
    N = n.sum()
    within = st.G[:d, :d] - (mu.T * n) @ mu                # sum x x^T - sum_c n_c mu mu^T
    cov = within / (N - len(seen))
    evals, Q = torch.linalg.eigh(cov)
    scale = evals.clamp(min=0).mean()
    out = {}
    for a in GRID:
        prec_mu = Q @ ((Q.T @ mu.T) / (evals.clamp(min=0) + a * scale)[:, None])   # d x C
        out[a] = (prec_mu, -0.5 * (mu * prec_mu.T).sum(1))
    return out


def per_task_acc(pred, y, tasks_seen):
    out = []
    for t in tasks_seen:
        m = torch.isin(y, torch.as_tensor(t, device=y.device))
        out.append(float((pred[m] == y[m]).float().mean()))
    return out


def summarize(matrix):
    return {"final_avg_acc": float(np.mean(matrix[-1])),
            "avg_incremental_acc": float(np.mean([np.mean(r) for r in matrix])),
            "forgetting": float(np.mean([max(matrix[i][j] for i in range(j, len(matrix) - 1)) - matrix[-1][j]
                                         for j in range(len(matrix) - 1)])) if len(matrix) > 1 else 0.0}


def run_one(spec, seed, data, tasks, device):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    fn = make_map(spec, xtr.shape[1], seed, device)
    ftr, fva, fte = apply(fn, xtr), apply(fn, xva), apply(fn, xte)
    st = Stats(ftr.shape[1], 100, device)
    mats = {"ncm": {"valid": [], "test": []}}
    for r in ("ridge", "slda"):
        for a in GRID:
            mats[(r, a)] = {"valid": [], "test": []}
    seen = []
    for t, cls in enumerate(tasks):
        m = torch.isin(ytr, torch.as_tensor(cls, device=device))
        st.add(ftr[m], ytr[m])
        seen = seen + cls
        seen_t = torch.as_tensor(seen, device=device)
        rw, sw = ridge_weights(st, seen), slda_weights(st, seen)
        for split, f, y in (("valid", fva, yva), ("test", fte, yte)):
            mk = torch.isin(y, seen_t)
            fs, ys = f[mk], y[mk]
            mats["ncm"][split].append(per_task_acc(predict_ncm(st, seen, fs), ys, tasks[: t + 1]))
            fa = aug(fs)
            for a in GRID:
                pr = seen_t[(fa @ rw[a]).argmax(1)]
                mats[("ridge", a)][split].append(per_task_acc(pr, ys, tasks[: t + 1]))
                pm, b = sw[a]
                ps = seen_t[(fs.double() @ pm + b).argmax(1)]
                mats[("slda", a)][split].append(per_task_acc(ps, ys, tasks[: t + 1]))
    res = {"ncm": {"alpha": None, "valid": summarize(mats["ncm"]["valid"]), "test": summarize(mats["ncm"]["test"]),
                   "test_matrix": mats["ncm"]["test"]}}
    for r in ("ridge", "slda"):
        best = max(GRID, key=lambda a: summarize(mats[(r, a)]["valid"])["final_avg_acc"])
        res[r] = {"alpha": best, "valid": summarize(mats[(r, best)]["valid"]),
                  "test": summarize(mats[(r, best)]["test"]), "test_matrix": mats[(r, best)]["test"],
                  "valid_grid": {str(a): summarize(mats[(r, a)]["valid"])["final_avg_acc"] for a in GRID}}
    res["dim"] = int(ftr.shape[1])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True)
    p.add_argument("--maps", default="raw,relu_2000,relu_8000,pop_2000_k100_K4,pop_8000_k160_K4,pop_8000_k160_K1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    blob = torch.load(a.feats)
    data = tuple((x.float().to(device), y.long().to(device)) for x, y in (blob["train"], blob["valid"], blob["test"]))
    tasks = [list(range(i * 10, i * 10 + 10)) for i in range(10)][: a.n_tasks]
    report = {"args": vars(a), "grid": GRID, "maps": {}}
    for spec in a.maps.split(","):
        seeds = [0] if spec == "raw" else [int(s) for s in a.seeds.split(",")]
        runs = {}
        for s in seeds:
            t0 = time.time()
            runs[s] = run_one(spec, s, data, tasks, device)
            print(spec, f"seed{s}", {r: (runs[s][r]["alpha"], round(runs[s][r]["valid"]["final_avg_acc"], 4),
                                          round(runs[s][r]["test"]["final_avg_acc"], 4)) for r in ("ncm", "ridge", "slda")},
                  f"{time.time() - t0:.0f}s", flush=True)
        agg = {}
        for r in ("ncm", "ridge", "slda"):
            for split in ("valid", "test"):
                for metric in ("final_avg_acc", "avg_incremental_acc"):
                    v = [runs[s][r][split][metric] for s in seeds]
                    agg[f"{r}/{split}/{metric}"] = (float(np.mean(v)), float(np.std(v)))
        report["maps"][spec] = {"runs": {str(s): v for s, v in runs.items()}, "agg": agg}
        print("==", spec, {k: f"{m:.4f}±{sd:.4f}" for k, (m, sd) in agg.items() if "test/final" in k}, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
