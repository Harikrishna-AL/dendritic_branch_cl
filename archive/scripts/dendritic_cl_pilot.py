"""CPU factorial pilot: competition, online consolidation, and teacher replay.

Fixed-capacity dendritic MLP on raw 14x14 Split MNIST. This isolates a mechanism;
it is not a BDH scaling experiment or a no-replay/boundary-free learner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bdh_cl.data.mnist import build_pools

ARMS = ("sum", "competitive", "protected", "protected_teacher", "protected_replay",
        "sum_teacher", "competitive_teacher")


class Student(nn.Module):
    def __init__(self, inputs=196, neurons=64, branches=4, pooling="max"):
        super().__init__()
        self.pooling = pooling
        self.weight = nn.Parameter(torch.randn(neurons, branches, inputs) / math.sqrt(inputs))
        self.bias = nn.Parameter(torch.zeros(neurons, branches))
        self.readout = nn.Linear(neurons, 10)

    def activity(self, x):
        a = F.relu(F.linear(x, self.weight.flatten(0, 1), self.bias.flatten()))
        a = a.reshape(len(x), *self.weight.shape[:2])
        return a

    def forward(self, x):
        a = self.activity(x)
        h = a.sum(-1) / math.sqrt(a.shape[-1]) if self.pooling == "sum" else a.max(-1).values
        return self.readout(h)


def distill(student, teacher, temperature=2.0):
    """Teacher is a detached target; KL is averaged over examples."""
    return temperature**2 * F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher.detach() / temperature, dim=-1), reduction="batchmean",
    )


class Consolidation:
    """Online empirical-Fisher anchoring, including shared readout parameters.

    Importance is normalized once per tensor, capped, accumulated, and anchored
    at each known task boundary. This is EWC-like, not a biological spike rule.
    """
    def __init__(self):
        self.omega = {}
        self.anchor = {}

    def penalty(self, model):
        return sum((self.omega[n] * (p - self.anchor[n]).square()).sum()
                   for n, p in model.named_parameters()) if self.omega else next(model.parameters()).sum()*0

    def consolidate(self, model, x, y, samples=32):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        count = min(samples, len(y))
        model.eval()
        for i in range(count):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
            for n, p in model.named_parameters():
                fisher[n].add_(p.grad.detach().square() / count)
        for n, p in model.named_parameters():
            f = fisher[n]
            f = (f / f.mean().clamp_min(1e-12)).clamp(max=10)
            self.omega[n] = (self.omega.get(n, torch.zeros_like(f)) + f).clamp(max=10)
            self.anchor[n] = p.detach().clone()
        model.zero_grad(set_to_none=True)


def subset(labels, classes, total, rng):
    assert total % len(classes) == 0
    ix = np.concatenate([rng.choice(np.flatnonzero(labels == c), total//len(classes),
                                    replace=False) for c in classes])
    return rng.permutation(ix)


def data_for_seed(pools, seed, ntrain, ntest):
    rng = np.random.default_rng(seed)
    result, indices = [], []
    for p in pools:
        tr = subset(p.train_labels, p.classes, ntrain, rng)
        te = subset(p.test_labels, p.classes, ntest, rng)
        def tensors(images, labels, ix):
            return (torch.tensor(images[ix].copy(), dtype=torch.float32).flatten(1)/255,
                    torch.tensor(labels[ix].astype(np.int64)))
        result.append((*tensors(p.train_images, p.train_labels, tr),
                       *tensors(p.test_images, p.test_labels, te)))
        indices.append({"train": tr.tolist(), "test": te.tolist()})
    return result, indices


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    acc, routes = [], []
    for _, _, x, y in data:
        logits, activity = model(x), model.activity(x)
        acc.append(float((logits.argmax(-1) == y).float().mean()))
        # For sum this is an activity diagnostic, not its actual update route.
        active = activity.max(-1).values > 0
        winners = activity.argmax(-1)
        usage = (F.one_hot(winners, activity.shape[-1])*active[..., None]).float().mean(0)
        routes.append((winners, active, usage))
    return acc, routes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--lr", type=float, default=.05)
    ap.add_argument("--protection", type=float, default=.1)
    ap.add_argument("--replay-budget", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.ntrain, args.ntest, args.seeds = 1, 64, 20, "0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    seeds = [int(s) for s in args.seeds.split(",")]
    protocol = {**vars(args), "seeds": seeds, "arms": ARMS,
                "schedule": [[0,1],[2,3],[4,5],[6,7],[8,9]],
                "architecture": "196 inputs, 64 neurons, 4 branches, trainable 10-class head",
                "evaluation": "global ten-class prediction; no task IDs; official test data",
                "optimizer": "SGD, no momentum or weight decay, constant rate",
                "protection": "task-boundary empirical Fisher (32 labeled examples), per-tensor normalized/capped online anchors; protects branch and readout weights",
                "teacher": "frozen preceding student snapshot; T=2; unit KL coefficient; no future teacher training",
                "replay": "100-example total class-balanced training reservoir; 16 old examples per new minibatch; same examples/order in teacher and replay control; replay control uses labels",
                "limitations": "boundary-aware, raw-pixel MLP, fixed hyperparameters, small pilot; teacher/replay arms use additional data presentations and storage",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch_version": torch.__version__}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    pools = build_pools(ROOT/"data_cache/mnist", side=14, task_tags=False)
    start = time.perf_counter()
    for seed in seeds:
        data, indices = data_for_seed(pools, seed, args.ntrain, args.ntest)
        (out/f"indices_seed{seed}.json").write_text(json.dumps(indices))
        torch.manual_seed(seed)
        initial = Student()
        report = {"seed": seed, "arms": {}}
        for arm in ARMS:
            model = copy.deepcopy(initial)
            model.pooling = "sum" if arm in ("sum", "sum_teacher") else "max"
            protected = arm.startswith("protected")
            replaying = arm.endswith("teacher") or arm == "protected_replay"
            memory, teacher, cons = None, None, Consolidation()
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            stages, references = [], []
            rng = torch.Generator().manual_seed(seed+1000)
            replay_rng = torch.Generator().manual_seed(seed+2000)
            arm_start = time.perf_counter()
            for task, (x,y,_,_) in enumerate(data):
                before, _ = evaluate(model, data)
                model.train()
                losses, replays = [], 0
                for _ in range(args.epochs):
                    for ix in torch.randperm(len(y), generator=rng).split(64):
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(x[ix]), y[ix])
                        if protected:
                            loss = loss + .5*args.protection*cons.penalty(model)
                        if replaying and memory is not None:
                            oldx, oldy = memory
                            ri = torch.randint(len(oldy), (16,), generator=replay_rng)
                            old_logits = model(oldx[ri])
                            if arm.endswith("teacher"):
                                with torch.no_grad():
                                    targets = teacher(oldx[ri])
                                loss = loss + distill(old_logits, targets)
                            else:
                                loss = loss + F.cross_entropy(old_logits, oldy[ri])
                            replays += len(ri)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        opt.step()
                        losses.append(float(loss.detach()))
                if protected:
                    cons.consolidate(model, x, y)
                scores, routes = evaluate(model, data)
                references.append((routes[task][0].clone(), routes[task][1].clone()))
                agreement = []
                for i, (winner, active, _) in enumerate(routes[:task+1]):
                    old_winner, old_active = references[i]
                    both = active & old_active
                    agreement.append(float((winner[both] == old_winner[both]).float().mean())
                                     if both.any() else None)
                stage = {"task": task, "before_accuracy": before, "accuracy": scores,
                         "seen_accuracy": float(np.mean(scores[:task+1])),
                         "old_accuracy": float(np.mean(scores[:task])) if task else None,
                         "new_accuracy": scores[task], "mean_training_loss": float(np.mean(losses)),
                         "route_agreement_on_jointly_active_units": agreement,
                         "branch_usage_by_task": [r[2].tolist() for r in routes],
                         "new_presentations": args.epochs*len(y), "old_presentations": replays}
                stages.append(stage)
                if replaying:
                    # Rebalance only from retained old samples and current task.
                    cx, cy = (torch.cat((memory[0], x)), torch.cat((memory[1], y))) if memory else (x,y)
                    per_class = args.replay_budget//(2*(task+1))
                    pick = []
                    for label in range(2*(task+1)):
                        candidates = (cy == label).nonzero().flatten()
                        pick.append(candidates[torch.randperm(len(candidates), generator=replay_rng)[:per_class]])
                    pick = torch.cat(pick)
                    memory = (cx[pick].clone(), cy[pick].clone())
                    teacher = copy.deepcopy(model).eval().requires_grad_(False)
                print(f"seed={seed} arm={arm} task={task} new={scores[task]:.3f} seen={stage['seen_accuracy']:.3f}", flush=True)
            matrix = np.array([s["accuracy"] for s in stages])
            forgetting = [float(matrix[j:,j].max()-matrix[-1,j]) for j in range(4)]
            count = sum(p.numel() for p in model.parameters())
            report["arms"][arm] = {"parameters": count, "stages": stages,
                "final_accuracy": float(matrix[-1].mean()), "mean_forgetting": float(np.mean(forgetting)),
                "seconds": time.perf_counter()-arm_start,
                "protection_bytes": 2*count*4 if protected else 0,
                "teacher_parameter_bytes": count*4 if arm.endswith("teacher") else 0,
                "replay_bytes": sum(t.numel()*t.element_size() for t in memory) if memory else 0}
            (out/f"seed{seed}.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"seed {seed} complete", flush=True)
    aggregate = {}
    for arm in ARMS:
        rows = [json.loads((out/f"seed{s}.json").read_text())["arms"][arm] for s in seeds]
        aggregate[arm] = {k: {"mean":float(np.mean([r[k] for r in rows])),
                             "sd":float(np.std([r[k] for r in rows], ddof=1)) if len(rows)>1 else None,
                             "per_seed":[r[k] for r in rows]}
                          for k in ("final_accuracy", "mean_forgetting", "seconds")}
    (out/"summary.json").write_text(json.dumps(aggregate, indent=2)+"\n")
    print(f"COMPLETE {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
