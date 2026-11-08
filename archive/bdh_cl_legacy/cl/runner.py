"""Sequential training over a list of tasks (Step 3).

One entry point, :func:`run_sequence`, drives every strategy in
:mod:`bdh_cl.cl.baselines`:

* train on task 0, task 1, ... task k-1 in order
* after each task, evaluate **all** tasks' held-out sets and fill row ``i`` of
  the accuracy matrix ``R``
* before task 0 and before each task j, also record the zero-shot entry needed
  for forward transfer
* collect an activation profile per task after each task, for the Step 3
  sparsity / overlap / repurposing probes
* checkpoint after each task

Everything is written out with the full config and the git hash.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from ..config import TrainSpec, as_dict
from . import probes as probes_mod
from .consolidation import build_consolidation
from . import wblog
from .metrics import CLResult, plasticity

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Optimizer / schedule
# --------------------------------------------------------------------------


def make_optimizer(model, train: TrainSpec):
    """AdamW with no weight decay on 1-D parameters (norms, biases, embeddings).

    Upstream decays everything; separating the groups is standard practice and
    applied identically to both models, so it cannot favour either.
    """
    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train.lr,
        betas=(0.9, 0.95),
    )


def lr_at(step: int, train: TrainSpec) -> float:
    """Linear warmup then cosine decay to 10% of peak, restarted per task."""
    if step < train.warmup_steps:
        return train.lr * (step + 1) / max(train.warmup_steps, 1)
    prog = (step - train.warmup_steps) / max(train.steps_per_task - train.warmup_steps, 1)
    prog = min(max(prog, 0.0), 1.0)
    return train.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# --------------------------------------------------------------------------
# One task's training
# --------------------------------------------------------------------------


def train_one_task(
    model,
    task,
    optimizer,
    train: TrainSpec,
    device,
    rng: np.random.Generator,
    *,
    batch_fn: Callable | None = None,
    curve_metric: Callable | None = None,
    log_prefix: str = "",
    verbose: bool = True,
    global_step_offset: tuple[int, int] = (0, 0),
    consolidation=None,
) -> dict:
    """Train ``train.steps_per_task`` steps on one task.

    ``batch_fn(rng, batch_size)`` overrides how batches are produced, which is
    how the replay and joint baselines inject data from other tasks.
    ``curve_metric(model)`` is sampled every ``eval_every`` steps to build the
    learning curve that plasticity is read off.
    """
    batch_fn = batch_fn or task.train_batch
    # Structural plasticity (e.g. rewiring) runs between optimizer steps.
    post_step_hooks = [
        m.post_step for m in model.modules() if callable(getattr(m, "post_step", None))
    ]
    model.train()
    # Losses are accumulated *on device*. Calling .item() every step forces a
    # GPU->CPU sync, and on MPS under memory pressure that sync becomes an
    # uninterruptible wait on faulted-in buffers -- it hung a 2.5h run dead.
    # We sync only at log/eval boundaries and once at the end.
    loss_buf = torch.zeros(train.steps_per_task, device=device)
    # Penalty tracked separately: the logged loss stays the DATA loss so the
    # curves remain comparable with runs that have no penalty, while the
    # penalty's size is still visible (if it is ~0 the mechanism is inert).
    pen_buf = torch.zeros(train.steps_per_task, device=device)
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    tokens = 0

    for step in range(train.steps_per_task):
        set_lr(optimizer, lr_at(step, train))
        batch = batch_fn(rng, train.batch_size).to(device)
        _logits, loss = model(
            batch.x, batch.y, loss_mask=getattr(batch, "loss_mask", None)
        )
        data_loss = loss
        if consolidation is not None:
            # Quadratic anchor penalty, added BEFORE backward. This is the part
            # that actually survives AdamW: rescaling a gradient by a constant
            # cancels in m/sqrt(v), whereas a penalty contributes a restoring
            # gradient of its own. Measured: pure gradient scaling left the
            # naive floor untouched (0.197 -> 0.199 across lam = 1..1000).
            pen = consolidation.penalty(model)
            if pen is not None:
                loss = loss + pen
                pen_buf[step] = pen.detach()
        loss.backward()
        # Importance-weighted gradient masking goes here: after the gradients
        # exist, BEFORE clipping, so the clip sees the gradients the optimizer
        # will actually apply rather than the unmasked ones.
        if consolidation is not None:
            consolidation.mask_grads(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for hook in post_step_hooks:
            hook(optimizer)
        loss_buf[step] = data_loss.detach()
        tokens += batch.x.numel()

        if curve_metric is not None and (
            step % train.eval_every == 0 or step == train.steps_per_task - 1
        ):
            cv = float(curve_metric(model))
            curve.append((step, cv))
            wblog.log(
                {f"curve/{task.name}": cv, "curve/current_task": cv},
                step=global_step_offset[0] + step,
            )
            model.train()

        if (step % train.log_every == 0 or step == train.steps_per_task - 1):
            lo = max(0, step - train.log_every + 1)
            # One sync, at a boundary that already syncs. Never per step.
            window = float(loss_buf[lo : step + 1].mean())
            wblog.log(
                {
                    f"train/{task.name}/loss": window,
                    "train/loss": window,
                    "train/lr": lr_at(step, train),
                    "train/task_index": global_step_offset[1],
                },
                step=global_step_offset[0] + step,
            )
            if verbose:
                print(
                    f"    {log_prefix}step {step:5d}/{train.steps_per_task} "
                    f"loss {window:.4f}"
                    + (f" curve {curve[-1][1]:.3f}" if curve else ""),
                    flush=True,
                )

    losses = loss_buf.detach().cpu().tolist()
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        # If this is ~0 the consolidation penalty is inert and any "no effect"
        # result is about the wiring, not about the idea.
        "penalty_mean": float(pen_buf.mean()),
        "curve": curve,
        "elapsed_sec": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens / elapsed if elapsed else float("nan"),
        "loss_first": float(np.mean(losses[:20])) if losses else float("nan"),
        "loss_last": float(np.mean(losses[-20:])) if losses else float("nan"),
    }


# --------------------------------------------------------------------------
# Evaluation of the whole task set
# --------------------------------------------------------------------------


def evaluate_all(
    model, tasks: Sequence, device, *, verbose: bool = True, with_longgen: bool = False
) -> list[dict]:
    out = []
    for t in tasks:
        r = t.evaluate(model, device, with_longgen=with_longgen)
        out.append(r)
        if verbose:
            print(f"      {t.name:10s} {t.metric_name}={r['value']:.4f}", flush=True)
    return out


def collect_profiles(model, tasks: Sequence, device, *, max_batches: int = 4) -> dict:
    """Activation profile per task, under the model's current weights."""
    return {
        t.name: probes_mod.collect_profile(
            model, t.eval_batches(), device, max_batches=max_batches
        )
        for t in tasks
    }


def collect_branch_profiles(model, tasks: Sequence, device, *, max_batches: int = 2) -> dict:
    """Per-task branch usage, for dendritic models. Empty dict otherwise."""
    out = {}
    for t in tasks:
        prof = probes_mod.collect_branch_profile(
            model, t.eval_batches(), device, max_batches=max_batches
        )
        if prof:
            out[t.name] = prof
    return out


def summarise_branches(per_task: dict, task_names: list[str]) -> dict:
    """Branch usage per task plus all-pairs 'same dominant branch' fractions."""
    if not per_task:
        return {}
    usage = {
        task: {stream: pr.to_dict() for stream, pr in streams.items()}
        for task, streams in per_task.items()
    }
    pairs = {}
    for i, a in enumerate(task_names):
        for b in task_names[i + 1 :]:
            if a in per_task and b in per_task:
                pairs[f"{a}|{b}"] = probes_mod.branch_overlap(per_task[a], per_task[b])
    return {"usage": usage, "same_branch": pairs}


# --------------------------------------------------------------------------
# The sequence
# --------------------------------------------------------------------------


def run_sequence(
    build_model: Callable[[], torch.nn.Module],
    tasks: Sequence,
    train: TrainSpec,
    device,
    *,
    strategy: str = "naive",
    strategy_batch_fn: Callable | None = None,
    label: str = "",
    out_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    collect_probes: bool = True,
    probe_batches: int = 4,
    random_init_scores: list[float] | None = None,
    verbose: bool = True,
    consolidation_cfg: dict | None = None,
) -> dict:
    """Train through ``tasks`` in order and return everything Step 3 needs.

    ``strategy_batch_fn(task_index, tasks, buffer)`` returns the ``batch_fn``
    used for task ``task_index``; :mod:`bdh_cl.cl.baselines` supplies these.
    """
    k = len(tasks)
    task_names = [t.name for t in tasks]
    metric_name = tasks[0].metric_name
    higher = tasks[0].higher_is_better
    assert all(t.metric_name == metric_name for t in tasks), "mixed metrics in one sequence"

    torch.manual_seed(train.seed)
    np.random.seed(train.seed)
    model = build_model().to(device)
    optimizer = make_optimizer(model, train)
    rng = np.random.default_rng(train.seed)
    # Built here rather than by the caller: the model does not exist until now,
    # and the importance tensors are shaped by its units.
    consolidation = build_consolidation(consolidation_cfg, model, device)

    R = [[float("nan")] * k for _ in range(k)]
    per_task_records: list[dict] = []
    profiles_after_task: dict[str, dict] = {}
    buffer: list = []

    # Row -1: the freshly initialised model on every task. Doubles as the FWT
    # baseline when the caller does not supply one.
    if verbose:
        print(f"  [{label}] random-init evaluation", flush=True)
    init_evals = evaluate_all(model, tasks, device, verbose=verbose)
    init_scores = [e["value"] for e in init_evals]

    # Branch usage at random init. Without it, a trained model's
    # "same dominant branch" number has no meaningful reference: even an
    # untrained dendritic unit has per-neuron branch preferences, because the
    # branch weight norms differ.
    branch_init = (
        summarise_branches(
            collect_branch_profiles(model, tasks, device, max_batches=probe_batches),
            task_names,
        )
        if collect_probes
        else {}
    )

    for i, task in enumerate(tasks):
        if verbose:
            print(f"  [{label}] task {i}: {task.name}", flush=True)

        if train.reset_optimizer_between_tasks and i > 0:
            optimizer = make_optimizer(model, train)

        batch_fn = (
            strategy_batch_fn(i, tasks, buffer) if strategy_batch_fn else task.train_batch
        )

        def curve_metric(m, _t=task):
            # Cheap subset: a full evaluation every eval_every steps would cost
            # more than the training being measured.
            return _t.evaluate(m, device, quick=True)["value"]

        rec = train_one_task(
            model, task, optimizer, train, device, rng,
            batch_fn=batch_fn,
            curve_metric=curve_metric,
            log_prefix=f"{task.name} ",
            verbose=verbose,
            consolidation=consolidation,
            # Monotonic x-axis across the whole sequence, so the W&B chart
            # reads as one continuous run with task boundaries visible.
            global_step_offset=(i * train.steps_per_task, i),
        )
        rec["task"] = task.name
        rec["index"] = i

        # Estimate importance on the task just finished, then fold it into the
        # running Omega. Done here, at the boundary, with the trained model --
        # not during training, where early steps would dilute the estimate with
        # activity from a model that had not learned the task yet.
        if consolidation is not None:
            crec = consolidation.estimate_and_merge(
                model, task, device, rng, task_index=i
            )
            rec["consolidation"] = crec
            if verbose and crec:
                for stream, st in crec.get("per_stream", {}).items():
                    print(
                        f"    [consolidate] {stream} protected="
                        f"{st['protected_frac']:.1%} omega_mean={st['omega_mean']:.3f}",
                        flush=True,
                    )

        # Grow the replay buffer with items from the task just finished.
        if strategy == "replay":
            n_keep = max(1, int(round(0.05 * train.steps_per_task * train.batch_size)))
            buffer.extend([(i, it) for it in task.sample_items(rng, n_keep)])
            rec["buffer_size"] = len(buffer)

        if verbose:
            print(f"    evaluating all {k} tasks", flush=True)
        # Length generalization is only reported for the final model, and it
        # costs about as much again as the main evaluation.
        evals = evaluate_all(
            model, tasks, device, verbose=verbose, with_longgen=(i == k - 1)
        )
        for j, e in enumerate(evals):
            R[i][j] = e["value"]
        rec["evals"] = [
            {kk: vv for kk, vv in e.items() if kk != "samples"} for e in evals
        ]
        per_task_records.append(rec)

        at = (i + 1) * train.steps_per_task
        wblog.log(
            {f"eval/{t.name}": e["value"] for t, e in zip(tasks, evals)}
            | {
                "eval/mean_over_tasks": float(
                    np.nanmean([e["value"] for e in evals])
                ),
                "eval/current_task": evals[i]["value"],
                "progress/tasks_done": i + 1,
            },
            step=at,
        )
        wblog.log_matrix("matrix/R", R, task_names, step=at)

        if collect_probes:
            profs = collect_profiles(model, tasks, device, max_batches=probe_batches)
            profiles_after_task[task.name] = profs

        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task_index": i,
                    "task": task.name,
                    "model": model.state_dict(),
                    "label": label,
                    "seed": train.seed,
                },
                ckpt_dir / f"{label}_seed{train.seed}_after{i}_{task.name}.pt",
            )

        # Persist after EVERY task. The first version of this only wrote at the
        # end of the sequence, so when a run hung on task 3 of 5 it left three
        # completed accuracy-matrix rows on the floor.
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{label}_seed{train.seed}.partial.json").write_text(
                json.dumps(
                    as_dict({
                        "label": label,
                        "seed": train.seed,
                        "complete": False,
                        "tasks_done": i + 1,
                        "task_names": task_names,
                        "metric": metric_name,
                        "higher_is_better": higher,
                        "R": R,
                        "random_init_own": init_scores,
                        "per_task": per_task_records,
                    }),
                    indent=2, default=str,
                ) + "\n"
            )

        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- probes: overlap between tasks, measured at the end of the sequence
    probe_out: dict = {}
    if collect_probes and profiles_after_task:
        final = profiles_after_task[task_names[-1]]
        probe_out["overlap_matrix_final"] = probes_mod.overlap_matrix(final, k_frac=0.05)
        probe_out["sparsity_per_task_final"] = {
            n: p.to_dict(k=256) for n, p in final.items()
        }
        # Repurposing: each task's own neurons, right after learning it vs at the end.
        rep = {}
        for i, n in enumerate(task_names):
            before = profiles_after_task[n][n]
            after = final[n]
            rep[n] = probes_mod.repurposing(before, after, k_frac=0.05)
        probe_out["repurposing"] = rep
        branch_final = summarise_branches(
            collect_branch_profiles(model, tasks, device, max_batches=probe_batches),
            task_names,
        )
        if branch_final or branch_init:
            probe_out["branches"] = {"at_init": branch_init, "final": branch_final}
        probe_out["sparsity_trajectory"] = {
            after_task: {n: p.to_dict(k=64) for n, p in profs.items()}
            for after_task, profs in profiles_after_task.items()
        }

    result = CLResult(
        task_names=task_names,
        metric_name=metric_name,
        higher_is_better=higher,
        R=R,
        random_init=random_init_scores or init_scores,
    )

    summary = result.summary()
    summary.update(
        {
            "label": label,
            "strategy": strategy,
            "seed": train.seed,
            "device": str(device),
            "train_spec": as_dict(train),
            "git_hash": git_hash(),
            "random_init_own": init_scores,
            "per_task": per_task_records,
            "probes": probe_out,
            "total_tokens": sum(r["tokens"] for r in per_task_records),
            "total_train_sec": sum(r["elapsed_sec"] for r in per_task_records),
        }
    )

    wblog.summary(
        {
            "final/average": summary["final"]["average"],
            "final/forgetting": summary["forgetting"]["average_excluding_last"],
            "final/bwt": summary["bwt"],
            "final/fwt": summary["fwt"],
            "final/total_tokens": summary["total_tokens"],
            "final/train_sec": summary["total_train_sec"],
        }
    )
    if probe_out:
        flat = {}
        for stream, d in (probe_out.get("overlap_matrix_final") or {}).items():
            J = np.array(d["jaccard"], dtype=float)
            C = np.array(d["cosine"], dtype=float)
            off = ~np.eye(J.shape[0], dtype=bool)
            flat[f"probe/{stream}/jaccard_offdiag"] = float(np.nanmean(J[off]))
            flat[f"probe/{stream}/cosine_offdiag"] = float(np.nanmean(C[off]))
        dens: dict[str, list] = {}
        for _task, streams in (probe_out.get("sparsity_per_task_final") or {}).items():
            for stream, d in streams.items():
                dens.setdefault(stream, []).extend(
                    layer["density"] for layer in d["per_layer"]
                )
        for stream, vals in dens.items():
            flat[f"probe/{stream}/density"] = float(np.mean(vals))
            flat[f"probe/{stream}/sparsity"] = 1.0 - float(np.mean(vals))
        wblog.summary(flat)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{label}_seed{train.seed}.json"
        path.write_text(json.dumps(as_dict(summary), indent=2, default=str) + "\n")
        summary["path"] = str(path)
        (out_dir / f"{label}_seed{train.seed}.partial.json").unlink(missing_ok=True)

    return summary


def add_plasticity(
    summary: dict,
    fresh_curves: dict[str, list],
    targets: dict[str, float] | float,
) -> dict:
    """Attach plasticity numbers, given per-task curves from fresh models.

    ``fresh_curves[task_name]`` is the learning curve of a model trained on that
    task from random init, i.e. the "how fast can this be learned at all"
    reference.

    ``targets`` should be **per task**. A single shared target is accepted but
    is usually wrong: tasks have very different ceilings (BDH-tiny reaches 1.00
    on `copy` but only ~0.30 on `add`), so one global threshold is trivial for
    the easy tasks and unreachable for the hard ones, and an unreachable target
    reports as "never learned" rather than "learned slowly".
    """
    higher = summary["higher_is_better"]
    out = {}
    for rec in summary["per_task"]:
        name = rec["task"]
        if name not in fresh_curves:
            continue
        tgt = targets[name] if isinstance(targets, dict) else targets
        out[name] = plasticity(rec["curve"], fresh_curves[name], tgt, higher)
        out[name]["task_ceiling_target"] = tgt
    summary["plasticity"] = out
    return summary
