"""Cross-task interference: does an update learned on task B disturb task A?

The hypothesis under test ties two results together. Ben-Kish et al. (2026,
"Local Support Learning") show that a weight update learned on input x_new
changes the model's output on an old input x_old by

    dv_old = -alpha * dL/dv * (x_new^T x_old)

so forgetting is driven by how much the update REACHES old inputs. Two
architectural properties could make BDH-family updates more local:

* across INPUTS: BDH's decoder reads the conjunction code xy_sparse, measured
  at ~5-8% density and anti-correlated beyond independence, so x_new^T x_old
  may be small;
* across PARAMETERS: a dendritic branch is private to one neuron, while the
  depth control (bdh_deep) routes everything through one SHARED hidden layer,
  so its updates should spread to every neuron. exp13 found dendritic 0.948
  vs depth 0.787 retained accuracy at equal parameters and neuron count.

Protocol per (model, seed, task pair A->B): train on A exactly as the runner
does, then measure, before and after a short burst of real optimizer steps on
B (fresh optimizer, as the runner resets it between tasks):

1. INTERFERENCE RATIO -- the headline. Rise in A's held-out loss per unit of
   fall in B's held-out loss. How much A is damaged per unit of B learned.
   Lower = more local. Reported at several step counts because at step 1 the
   denominator is tiny and the ratio is unstable.
2. GRADIENT ALIGNMENT -- cos(grad L_A, grad L_B), overall and per parameter
   group. Negative = a descent step on B ascends A.
3. LOCALITY OF THE CHANGE -- on A's inputs, how the code the output layer
   reads changes after the B steps: relative size of the change, and the
   fraction of active units that carry 90% of its energy. Small fraction =
   the update touched few units = local.
4. INPUT OVERLAP -- mean cosine between codes of random A tokens and random B
   tokens (LSL's x_new^T x_old term), against the A-vs-A baseline.

Predictions if the hypothesis is right: dendritic < baseline < depth on (1)
and (3), and BDH below GPT on (4). Any of these can fail, which is the point.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl.runner import make_optimizer, set_lr, train_one_task  # noqa: E402
from bdh_cl.data.bytes import PAD  # noqa: E402


# ------------------------------------------------------------------ helpers

@torch.no_grad()
def task_loss(model, batches, device) -> float:
    """Mean masked loss over held-out batches, dropout off."""
    model.eval()
    tot = 0.0
    for b in batches:
        b = b.to(device)
        _, loss = model(b.x, b.y, loss_mask=b.loss_mask)
        tot += float(loss)
    return tot / max(len(batches), 1)


def grads(model, batches, device) -> dict[str, torch.Tensor]:
    """Gradient of the mean held-out loss, dropout off so it is deterministic."""
    model.eval()
    model.zero_grad(set_to_none=True)
    for b in batches:
        b = b.to(device)
        _, loss = model(b.x, b.y, loss_mask=b.loss_mask)
        (loss / len(batches)).backward()
    out = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return out


def group_of(name: str) -> str:
    """Coarse parameter groups that mean the same thing across models.

    encoder_v must be tested before encoder, which is its prefix.
    """
    if "neuron_unit_y" in name or "encoder_v" in name:
        return "enc_y"
    if "neuron_unit_x" in name or "encoder" in name:
        return "enc_x"
    if "decoder" in name:
        return "dec"
    if "embed" in name or "lm_head" in name or "wte" in name:
        return "io"
    if "attn" in name:
        return "attn"
    if "mlp" in name or "fc" in name:
        return "mlp"
    return "other"


def cosines(ga: dict, gb: dict) -> dict[str, float]:
    groups: dict[str, list[str]] = {}
    for n in ga:
        if n in gb:
            groups.setdefault(group_of(n), []).append(n)

    def cos(names):
        a = torch.cat([ga[n].flatten() for n in names])
        b = torch.cat([gb[n].flatten() for n in names])
        d = float(a.norm() * b.norm())
        return float(a @ b) / d if d > 0 else float("nan")

    out = {g: cos(ns) for g, ns in groups.items()}
    out["all"] = cos([n for ns in groups.values() for n in ns])
    return out


@torch.no_grad()
def codes(model, batch, device):
    """The code the output layer reads, per layer, as (L, B, T, U), plus mask.

    BDH: xy_sparse, the conjunction the decoder consumes. GPT: post-GELU MLP
    hidden, which its down-projection consumes. Same role in both.
    """
    model.eval()
    b = batch.to(device)
    out = model(b.x, None, return_activations=True)
    acts = out[2]
    layers = getattr(acts, "xy_sparse", None) or getattr(acts, "hidden", None)
    if not layers:
        raise RuntimeError(f"no usable activations on {type(acts).__name__}")
    vecs = []
    for t in layers:
        if t.dim() == 4:  # B, nh, T, N  ->  B, T, nh*N
            t = t.permute(0, 2, 1, 3).reshape(t.shape[0], t.shape[2], -1)
        vecs.append(t.float())
    return torch.stack(vecs), (b.x != PAD)


def change_locality(before, after, mask) -> dict:
    """How concentrated the change in A's code is, after the B steps."""
    m = mask[None, :, :, None].float()
    d2 = ((after - before).pow(2) * m).sum(dim=(0, 1, 2))          # per unit
    active = (((before > 0) | (after > 0)).float() * m).sum(dim=(0, 1, 2)) > 0
    n_active = int(active.sum())
    tot = float(d2.sum())
    base = float((before.pow(2) * m).sum())
    if tot <= 0:
        return {"units": d2.numel(), "active_units": n_active, "rel_change": 0.0,
                "frac_units_90": 0.0, "frac_active_90": 0.0}
    s, _ = torch.sort(d2, descending=True)
    k90 = int((torch.cumsum(s, 0) / tot < 0.9).sum()) + 1
    return {
        "units": d2.numel(),
        "active_units": n_active,
        "rel_change": math.sqrt(tot / max(base, 1e-12)),
        "frac_units_90": k90 / d2.numel(),
        "frac_active_90": k90 / max(n_active, 1),
    }


@torch.no_grad()
def overlap(ca, ma, cb, mb, n_pairs: int, seed: int) -> dict:
    """Mean cosine of random A-B token code pairs, and the A-A baseline."""
    g = torch.Generator().manual_seed(seed)
    ab, aa = [], []
    for layer in range(ca.shape[0]):
        A = ca[layer][ma]
        B = cb[layer][mb]
        if len(A) < 2 or len(B) < 1:
            continue
        i = torch.randint(len(A), (n_pairs,), generator=g).to(A.device)
        j = torch.randint(len(B), (n_pairs,), generator=g).to(A.device)
        k = torch.randint(len(A), (n_pairs,), generator=g).to(A.device)
        ab.append(float(F.cosine_similarity(A[i], B[j], dim=-1).mean()))
        aa.append(float(F.cosine_similarity(A[i], A[k], dim=-1).mean()))
    oab = float(np.mean(ab)) if ab else float("nan")
    oaa = float(np.mean(aa)) if aa else float("nan")
    return {"cross": oab, "within_A": oaa, "cross_over_within": oab / oaa if oaa else float("nan")}


# --------------------------------------------------------------------- core

def probe_pair(build_model, task_a, task_b, train, device, seed,
               b_steps=(1, 5, 20, 50), grad_batches=2, n_pairs=2000) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().to(device)
    rng = np.random.default_rng(seed)

    # Train on A exactly as the runner would.
    opt = make_optimizer(model, train)
    train_one_task(model, task_a, opt, train, device, rng, verbose=False)

    ev_a, ev_b = task_a.eval_batches(), task_b.eval_batches()
    cos = cosines(grads(model, ev_a[:grad_batches], device),
                  grads(model, ev_b[:grad_batches], device))
    la0, lb0 = task_loss(model, ev_a, device), task_loss(model, ev_b, device)
    before, mask_a = codes(model, ev_a[0], device)
    cb, mask_b = codes(model, ev_b[0], device)
    ov = overlap(before, mask_a, cb, mask_b, n_pairs, seed)

    # A short burst on B. Fresh optimizer: the runner resets it between tasks
    # (reset_optimizer_between_tasks defaults to True), and carrying A's Adam
    # moments into B would measure a different regime than the CL runs used.
    opt = make_optimizer(model, train)
    traj, done = [], 0
    for target in sorted(b_steps):
        model.train()
        while done < target:
            set_lr(opt, train.lr)
            b = task_b.train_batch(rng, train.batch_size).to(device)
            _, loss = model(b.x, b.y, loss_mask=b.loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
        la, lb = task_loss(model, ev_a, device), task_loss(model, ev_b, device)
        traj.append({
            "steps": target, "loss_a": la, "loss_b": lb,
            "dA": la - la0, "dB": lb0 - lb,
            "ratio": (la - la0) / max(lb0 - lb, 1e-6),
        })
    after, _ = codes(model, ev_a[0], device)
    return {
        "loss_a0": la0, "loss_b0": lb0,
        "grad_cos": cos, "overlap": ov,
        "trajectory": traj,
        "locality": change_locality(before, after, mask_a),
    }


# --------------------------------------------------------------------- main

def main() -> None:
    from bdh_cl.cl.tasks import build_synthetic_sequence
    from bdh_cl.config import build_from_yaml
    from bdh_cl.scripts.run_cl import model_builders, pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="bdh,bdh_den_sum,bdh_den_gated,bdh_deep,gpt")
    ap.add_argument("--pairs", default="copy:reverse,reverse:sort")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--b-steps", default="1,5,20,50")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = build_from_yaml(args.config)
    device = pick_device(args.device)
    builders = model_builders(cfg)
    train0 = cfg["train_spec"]
    # Same task list as the CL runs, so seq_len (and eval sets) match exp13.
    tasks = {t.name: t for t in build_synthetic_sequence(
        cfg["tasks"], n_eval=cfg.get("n_eval", 100),
        eval_seed=cfg.get("eval_seed", 0),
        batch_size=cfg.get("eval_batch_size", 32))}
    b_steps = tuple(int(x) for x in args.b_steps.split(","))

    runs = {}
    for mname in args.models.split(","):
        if mname not in builders:
            raise SystemExit(f"unknown model {mname!r}; have {sorted(builders)}")
        for pair in args.pairs.split(","):
            a, b = pair.split(":")
            for seed in [int(s) for s in args.seeds.split(",")]:
                train = dataclasses.replace(train0, seed=seed)
                r = probe_pair(builders[mname], tasks[a], tasks[b], train, device,
                               seed, b_steps=b_steps)
                key = f"{mname}|{a}->{b}|seed{seed}"
                runs[key] = r
                t = {x["steps"]: x for x in r["trajectory"]}
                last = max(t)
                print(f"{key:36s} cos_all {r['grad_cos']['all']:+.3f}  "
                      f"ratio@{last} {t[last]['ratio']:.3f}  dA@{last} {t[last]['dA']:+.3f}  "
                      f"frac90 {r['locality']['frac_active_90']:.3f}  "
                      f"overlap {r['overlap']['cross']:.3f}/{r['overlap']['within_A']:.3f}",
                      flush=True)

    # Aggregate over seeds AND pairs per model.
    def agg(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return "-"
        return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

    last = max(b_steps)
    print("\n=== summary (mean ± sd over seeds x pairs) ===")
    print(f"{'model':14s} {'ratio@20':>15s} {f'ratio@{last}':>15s} {'grad cos':>15s} "
          f"{'frac90 active':>15s} {'rel change':>15s} {'overlap A-B':>15s} {'overlap A-A':>15s}")
    summary = {}
    for mname in args.models.split(","):
        rs = [v for k, v in runs.items() if k.startswith(mname + "|")]
        pick = lambda s: [next(x["ratio"] for x in r["trajectory"] if x["steps"] == s) for r in rs]
        row = {
            "ratio@20": pick(20) if 20 in b_steps else [],
            f"ratio@{last}": pick(last),
            "grad_cos_all": [r["grad_cos"]["all"] for r in rs],
            "frac_active_90": [r["locality"]["frac_active_90"] for r in rs],
            "rel_change": [r["locality"]["rel_change"] for r in rs],
            "overlap_cross": [r["overlap"]["cross"] for r in rs],
            "overlap_within": [r["overlap"]["within_A"] for r in rs],
        }
        summary[mname] = row
        print(f"{mname:14s} " + " ".join(f"{agg(v):>15s}" for v in row.values()))

    if args.out:
        Path(args.out).write_text(json.dumps({"runs": runs, "summary": summary}, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
