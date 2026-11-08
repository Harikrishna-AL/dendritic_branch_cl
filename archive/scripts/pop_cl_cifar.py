"""Exp20: binary branched population code vs standard CL baselines on Split CIFAR-100.

Question: does a wide, binary, top-k population layer with dendritic branches,
read out by class prototypes frozen at the end of each task (optionally with
tiredness-scaled plasticity), beat standard exemplar-free continual learning
on a benchmark where fixed features fail? Training-free floor on this split:
nearest class mean on fixed random dendritic features = 0.151 final accuracy
(job 4385548, NOTES 2026-10-07).

Protocol (shared by every arm):
  * CIFAR-100, 10 tasks x 10 classes, class order from ``data.cifar100.load``.
  * From scratch, reduced ResNet-18 (nf=20, 160-d features), same optimiser,
    epochs, batch and augmentation for every arm. No exemplars stored.
  * Class-incremental evaluation: argmax over ALL classes seen so far, no task
    id. Evaluated after every task on ``--eval-split`` (valid for tuning, test
    for reporting).
  * Two readouts reported for every arm:
      head   the learned linear head, restricted to seen classes;
      ncm    nearest class mean (cosine) with each class's prototype stored at
             the end of ITS OWN task from its training images and never
             revised. Exemplar-free: a prototype is a mean, not data.
    Plus ``ncm_oracle`` after the last task (prototypes recomputed with the
    final model from all training data -- needs old data, so NOT a method; the
    gap to ``ncm`` is representation drift).

Arms:
  finetune        CE over seen classes.
  finetune_local  CE over the current task's classes only (no head bias
                  pressure); the control for ``pop`` without the population.
  lwf             finetune + distillation on old-class logits (Li & Hoiem).
  ewc             finetune + online diagonal-Fisher EWC penalty.
  pop             backbone -> N neurons x K branches -> max -> binary top-k
                  (straight-through sigmoid surrogate) -> head; local CE.
  pop_tired       pop + tiredness-scaled plasticity: gradient through neuron i
                  scaled by (1 - t_i), t_i = f_i / (f_i + tau), f_i = firings
                  in COMPLETED tasks. Scaling the gradient at the neuron's
                  output slows its branch weights AND the backbone paths that
                  feed old tasks' neurons.
  joint           all 100 classes at once, same epochs: upper bound.

    python bdh_cl/scripts/pop_cl_cifar.py --data data_cache/cifar100 --arm pop \
        --seed 0 --eval-split valid --out results/exp20/pop_seed0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bdh_cl.cl import wblog  # noqa: E402

ARMS = ["finetune", "finetune_local", "lwf", "ewc", "pop", "pop_tired", "joint", "joint_pop"]
JOINT = ("joint", "joint_pop")


# ------------------------------------------------------------------ models

class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.sc(x))


class ReducedResNet18(nn.Module):
    """The nf=20 ResNet-18 used across the CL literature (GEM, ER, Mammoth)."""

    def __init__(self, nf=20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU())
        layers, cin = [], nf
        for mult, stride in ((1, 1), (2, 2), (4, 2), (8, 2)):
            layers += [Block(cin, nf * mult, stride), Block(nf * mult, nf * mult, 1)]
            cin = nf * mult
        self.layers = nn.Sequential(*layers)
        self.dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1)


def scale_grad(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Identity forward; backward multiplies the gradient by ``g``."""
    return x * g + (x * (1 - g)).detach()


class PopLayer(nn.Module):
    """N neurons x K dense branches, soma = max branch, binary top-k output.

    Scores are z-scored across neurons per example so the surrogate
    temperature is scale-free. Forward is exactly binary (k ones); backward is
    the sigmoid around the midpoint between the k-th and (k+1)-th score.
    """

    def __init__(self, d, n, branches, k, temp):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, branches, d) / d ** 0.5)
        self.b = nn.Parameter(torch.zeros(n, branches))
        self.k, self.temp = k, temp
        self.register_buffer("grad_scale", torch.ones(n))
        self.register_buffer("fired_task", torch.zeros(n))   # current task
        self.register_buffer("fired_prev", torch.zeros(n))   # completed tasks
        self.tired = False

    def forward(self, h):
        score = torch.einsum("bd,nkd->bnk", h, self.w).add(self.b).amax(-1)
        # Normaliser and threshold are treated as constants: otherwise every
        # neuron's gradient leaks into all others through them, and a neuron
        # whose plasticity is scaled to zero would still be rewritten.
        mu, sd = score.mean(1, keepdim=True).detach(), score.std(1, keepdim=True).detach()
        z = (score - mu) / (sd + 1e-5)
        top = z.detach().topk(self.k + 1, dim=1).values
        hard = (z >= top[:, self.k - 1 : self.k]).float()
        theta = 0.5 * (top[:, self.k - 1] + top[:, self.k])
        soft = torch.sigmoid((z - theta[:, None]) / self.temp)
        a = hard + soft - soft.detach()
        if self.training:
            self.fired_task += hard.detach().sum(0)
            if self.tired:
                a = scale_grad(a, self.grad_scale)
        return a

    @torch.no_grad()
    def end_task(self, tau):
        self.fired_prev += self.fired_task
        self.fired_task.zero_()
        t = self.fired_prev / (self.fired_prev + tau)
        self.grad_scale.copy_(1 - t)
        return t


class Flat(nn.Module):
    """No-conv backbones. ``none``: raw pixels. ``mlp``: one dense ReLU layer
    (the no-population control at matched width)."""

    def __init__(self, kind, width, d_in=3072):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), *([nn.Linear(d_in, width), nn.ReLU()] if kind == "mlp" else []))
        self.dim = width if kind == "mlp" else d_in

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, arm, n_classes=100, nf=20, pop_n=2000, pop_branches=4, pop_k=100, temp=0.5,
                 backbone="conv", pop_layers=1, d_in=3072):
        super().__init__()
        self.backbone = ReducedResNet18(nf) if backbone == "conv" else Flat(backbone, pop_n, d_in)
        d = self.backbone.dim
        self.pops = nn.ModuleList()
        if arm.startswith("pop") or arm == "joint_pop":
            for _ in range(pop_layers):
                layer = PopLayer(d, pop_n, pop_branches, pop_k, temp)
                layer.tired = arm == "pop_tired"
                self.pops.append(layer)
                d = pop_n
        self.head = nn.Linear(d, n_classes)

    def features(self, x):
        h = self.backbone(x)
        for layer in self.pops:
            h = layer(h)
        return h

    def forward(self, x):
        return self.head(self.features(x))


# -------------------------------------------------------------------- data

def images(x, augment=False):
    if x.dtype != torch.uint8:   # precomputed frozen features (--feats): no-op
        return x
    x = x.float() / 255.0
    if augment:
        B = len(x)
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
        off = torch.randint(0, 9, (B, 2), device=x.device)
        ar = torch.arange(32, device=x.device)
        yy = off[:, 0, None, None] + ar[None, :, None]
        xx = off[:, 1, None, None] + ar[None, None, :]
        x = padded.permute(0, 2, 3, 1)[torch.arange(B, device=x.device)[:, None, None], yy, xx].permute(0, 3, 1, 2)
        x = torch.where(torch.rand(B, 1, 1, 1, device=x.device) < 0.5, x.flip(-1), x)
    return (x - 0.5) / 0.5


def subset(x, y, classes):
    m = torch.isin(y, torch.as_tensor(classes, device=y.device))
    return x[m], y[m]


# --------------------------------------------------------------- readouts

@torch.no_grad()
def embed(model, x, bs=500):
    model.eval()
    return torch.cat([model.features(images(x[i : i + bs])) for i in range(0, len(x), bs)])


@torch.no_grad()
def prototypes(model, x, y, classes):
    f = F.normalize(embed(model, x).float(), dim=-1)
    return {c: f[y == c].mean(0) for c in classes}


@torch.no_grad()
def evaluate(model, x, y, seen, protos):
    """Head and NCM predictions over all seen classes."""
    model.eval()
    feats, logits = [], []
    for i in range(0, len(x), 500):
        f = model.features(images(x[i : i + 500]))
        feats.append(f.float())
        logits.append(model.head(f)[:, seen])
    seen_t = torch.as_tensor(seen, device=y.device)
    out = {"head": seen_t[torch.cat(logits).argmax(1)]}
    if protos:
        cls = sorted(protos)
        P = F.normalize(torch.stack([protos[c] for c in cls]), dim=-1)
        out["ncm"] = torch.as_tensor(cls, device=y.device)[(F.normalize(torch.cat(feats), dim=-1) @ P.T).argmax(1)]
    return out


def per_task_acc(pred, y, tasks_seen):
    return [float((pred[torch.isin(y, torch.as_tensor(t, device=y.device))] ==
                   y[torch.isin(y, torch.as_tensor(t, device=y.device))]).float().mean())
            for t in tasks_seen]


def summarize(matrix):
    return {"final_avg_acc": float(np.mean(matrix[-1])),
            "avg_incremental_acc": float(np.mean([np.mean(r) for r in matrix])),
            "forgetting": float(np.mean([max(matrix[i][j] for i in range(j, len(matrix) - 1)) - matrix[-1][j]
                                         for j in range(len(matrix) - 1)])) if len(matrix) > 1 else 0.0}


# ------------------------------------------------------------------- EWC

class EWC:
    def __init__(self, lam):
        self.lam, self.fisher, self.anchor = lam, None, None

    def penalty(self, model):
        if self.fisher is None:
            return 0.0
        return 0.5 * self.lam * sum((self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
                                    for n, p in model.named_parameters())

    def consolidate(self, model, x, y, seen, n_samples=200):
        """Online EWC: Fisher accumulates across tasks (gamma = 1)."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        idx = torch.randperm(len(x), device=x.device)[:n_samples]
        target = torch.searchsorted(seen, y)   # label -> position among seen
        for i in idx:
            model.zero_grad()
            F.cross_entropy(model(images(x[i : i + 1]))[:, seen], target[i : i + 1]).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 / len(idx)
        if self.fisher is not None:
            fisher = {n: fisher[n] + self.fisher[n] for n in fisher}
        self.fisher = fisher
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.zero_grad()


# ------------------------------------------------------------------- train

def train_task(model, a, x, y, seen, task_cls, old_model, ewc, device):
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    steps = a.epochs * ((len(x) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    seen_t = torch.as_tensor(seen, device=device)
    remap = torch.full((100,), -1, dtype=torch.long, device=device)
    lo_cls = torch.as_tensor(task_cls, device=device)
    old_cls = seen_t[~torch.isin(seen_t, lo_cls)]
    local = a.arm in ("finetune_local", "pop", "pop_tired")
    remap[lo_cls if local else seen_t] = torch.arange(len(lo_cls) if local else len(seen_t), device=device)
    losses = []
    for _ in range(a.epochs):
        model.train()
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), a.batch):
            b = perm[i : i + a.batch]
            xb = images(x[b], augment=True)
            logits = model(xb)
            out = logits[:, lo_cls] if local else logits[:, seen_t]
            loss = F.cross_entropy(out, remap[y[b]])
            if a.arm == "lwf" and old_model is not None and len(old_cls):
                with torch.no_grad():
                    old = old_model(xb)[:, old_cls]
                T = 2.0
                loss = loss + a.lwf_lambda * T * T * F.kl_div(
                    F.log_softmax(logits[:, old_cls] / T, 1), F.softmax(old / T, 1), reduction="batchmean")
            if a.arm == "ewc":
                loss = loss + ewc.penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses[-50:]))


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    from bdh_cl.data import cifar100

    if a.feats:   # frozen features from extract_feats.py, same splits/class order
        if a.backbone == "conv":
            raise SystemExit("--feats needs --backbone none or mlp")
        blob = torch.load(a.feats)
        (xtr, ytr), (xva, yva), (xte, yte) = blob["train"], blob["valid"], blob["test"]
        xdt = torch.float32
    else:
        (xtr, ytr), (xva, yva), (xte, yte), order = cifar100.load(a.data)
        xdt = torch.uint8
    xev, yev = (xva, yva) if a.eval_split == "valid" else (xte, yte)
    to = lambda arr, dt: torch.as_tensor(arr, dtype=dt, device=device)
    xtr, ytr, xev, yev = to(xtr, xdt), to(ytr, torch.long), to(xev, xdt), to(yev, torch.long)
    if a.limit:   # smoke: few images per class
        keep = torch.cat([torch.nonzero(ytr == c).flatten()[: a.limit] for c in range(100)])
        xtr, ytr = xtr[keep], ytr[keep]

    tasks = [list(range(i * 10, i * 10 + 10)) for i in range(10)][: a.n_tasks]
    if a.arm in JOINT:
        tasks_train = [sum(tasks, [])]
    else:
        tasks_train = tasks

    model = Net(a.arm, nf=a.nf, pop_n=a.pop_n, pop_branches=a.pop_branches,
                pop_k=a.pop_k, temp=a.temp, backbone=a.backbone, pop_layers=a.pop_layers,
                d_in=xtr[0].numel()).to(device)
    ewc = EWC(a.ewc_lambda) if a.arm == "ewc" else None
    name = f"{a.arm}_seed{a.seed}" + (f"_{a.tag}" if a.tag else "")
    wblog.init({"enabled": not a.no_wandb, "project": "bdh-cl", "group": a.group, "job_type": a.eval_split},
               name=name, config=vars(a))

    seen, protos, old_model = [], {}, None
    mats = {"head": [], "ncm": []}
    log = []
    t0 = time.time()
    for t, cls in enumerate(tasks_train):
        seen = seen + cls
        x, y = subset(xtr, ytr, cls)
        loss = train_task(model, a, x, y, seen, cls, old_model, ewc, device)
        if a.arm not in JOINT:
            protos.update(prototypes(model, x, y, cls))   # frozen at task end
        if ewc is not None:
            ewc.consolidate(model, x, y, torch.as_tensor(seen, device=device))
        tired = None
        if len(model.pops):
            tau = a.tau_tasks * a.pop_k / a.pop_n * len(x) * a.epochs
            tired = {}
            for li, layer in enumerate(model.pops):
                tvec = layer.end_task(tau)
                sfx = "" if li == 0 else f"_L{li + 1}"
                tired |= {"mean" + sfx: float(tvec.mean()), "frac_gt_0.9" + sfx: float((tvec > 0.9).float().mean()),
                          "never_fired" + sfx: float((layer.fired_prev == 0).float().mean())}
        if a.arm == "lwf":
            old_model = copy.deepcopy(model).eval()
            for p in old_model.parameters():
                p.requires_grad_(False)

        xe, ye = subset(xev, yev, seen)
        tasks_seen = tasks if a.arm in JOINT else tasks[: t + 1]
        pred = evaluate(model, xe, ye, seen, protos)
        row = {k: per_task_acc(v, ye, tasks_seen) for k, v in pred.items()}
        for k in row:
            mats[k].append(row[k])
        entry = {"task": t, "loss": loss, "elapsed_s": time.time() - t0,
                 **{f"{k}_acc": float(np.mean(v)) for k, v in row.items()}, "tired": tired}
        log.append(entry)
        print(json.dumps(entry), flush=True)
        wblog.log({"cl/stage": t, **{f"cl/{k}_acc": float(np.mean(v)) for k, v in row.items()},
                   "cl/loss": loss, **({f"tired/{k}": v for k, v in tired.items()} if tired else {})})

    # Oracle: every prototype recomputed with the final model (needs old data).
    oracle = prototypes(model, xtr, ytr, sum(tasks, []))
    xe, ye = subset(xev, yev, sum(tasks, []))
    pred_o = evaluate(model, xe, ye, sum(tasks, []), oracle)["ncm"]
    oracle_acc = float(np.mean(per_task_acc(pred_o, ye, tasks)))

    result = {"args": vars(a), "log": log, "matrices": {k: v for k, v in mats.items() if v},
              "summary": {k: summarize(v) for k, v in mats.items() if v},
              "ncm_oracle_final": oracle_acc, "seconds": time.time() - t0}
    if result["summary"].get("ncm"):
        result["drift"] = oracle_acc - result["summary"]["ncm"]["final_avg_acc"]
    wblog.summary({"final/" + f"{k}_{m}": v for k, s in result["summary"].items() for m, v in s.items()}
                  | {"final/ncm_oracle": oracle_acc})
    wblog.finish()
    print("SUMMARY", json.dumps({k: v for k, v in result.items() if k in ("summary", "ncm_oracle_final", "drift", "seconds")}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--nf", type=int, default=20)
    p.add_argument("--backbone", choices=["conv", "mlp", "none"], default="conv")
    p.add_argument("--pop-layers", type=int, default=1)
    p.add_argument("--feats", default=None, help="precomputed features (.pt from extract_feats.py)")
    p.add_argument("--clip", type=float, default=0.0, help="grad-norm clip; 0 = off")
    p.add_argument("--pop-n", type=int, default=2000)
    p.add_argument("--pop-branches", type=int, default=4)
    p.add_argument("--pop-k", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--tau-tasks", type=float, default=1.0,
                   help="tau in units of an average neuron's firings over one task")
    p.add_argument("--lwf-lambda", type=float, default=1.0)
    p.add_argument("--ewc-lambda", type=float, default=100.0)
    p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="smoke: train images per class")
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", default="exp20_pop_cifar100")
    p.add_argument("--tag", default="")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out", default=None)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
