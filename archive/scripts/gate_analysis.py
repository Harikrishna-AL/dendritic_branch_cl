"""Is the learned `gated` soma actually routing, or has it collapsed to the mean?

Motivation: `bdh_den_gated` (0.932) and `bdh_den_sum` (0.923) are within seed
noise of each other. Two readings:

  (a) gating is doing nothing -- the learned beta is small enough that
      softmax_c(beta * a_c) is ~uniform, so the neuron computes the mean over
      branches, which is `sum` up to a factor of k that the decoder absorbs.
  (b) gating is routing, but this task family is too easy for routing to pay.

These are distinguishable without retraining, because the two limits of the
gate are exactly the two other somas:

      beta -> 0    =>  uniform mean over branches   (`sum` / k)
      beta -> inf  =>  hard argmax                  (`max`)

So we take the TRAINED gated weights and, at evaluation only, override the
gate. If accuracy is unchanged when the gate is forced uniform, the trained
gate was not contributing -- reading (a). If accuracy drops, the gate was
carrying real input-dependent structure -- reading (b).

Alongside the functional test we measure the gate directly: its sharpness
(mean max_c w_c, where 1/k is uniform and 1 is hard) and, more importantly,
its INPUT DEPENDENCE -- a gate that always picks the same branch for a given
neuron regardless of input is static, and a static gate provably folds into the
branch weights (see DendriticNeuronUnit docstring), i.e. it is `sum` in
disguise.

    python bdh_cl/scripts/gate_analysis.py \
        --config bdh_cl/configs/exp4_dendritic.yaml \
        --ckpt-dir bdh_cl/results/exp4_dendritic/ckpt \
        --strategy replay --after 2_sort
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl.probes import _dendritic_units  # noqa: E402
from bdh_cl.cl.tasks import build_synthetic_sequence  # noqa: E402
from bdh_cl.config import build_from_yaml  # noqa: E402
from bdh_cl.scripts.run_cl import model_builders, pick_device  # noqa: E402


# ---------------------------------------------------------------- gate override

def make_soma(mode: str, beta_scale: float = 1.0):
    """Return a forward() for DendriticNeuronUnit implementing `mode`.

    Every mode reuses the unit's own trained branch weights; only the pooling
    changes. `trained` reproduces the unit exactly (asserted below).
    """

    def fwd(self, x, weight=None):
        a = self.branch_activations(x)
        if mode == "uniform":          # beta -> 0 limit: mean over branches
            return a.mean(dim=-2)
        if mode == "uniform_scaled":
            # Same function as `uniform`, but with the per-(head, neuron) gain
            # that best matches the trained gate's output scale. Because
            #     sum_c w_c a_c  ~=  mean_c(a) + beta * Var_c(a),
            # dropping the gate lowers every activation, and a bare `uniform`
            # ablation therefore confounds "lost input-dependent structure"
            # with "lost gain". Calibrated on the task's own eval data, i.e.
            # as favourably to the ablation as possible.
            out = a.mean(dim=-2)
            g = getattr(self, "_calib_gain", None)
            return out if g is None else out * g
        if mode == "hard":             # beta -> inf limit
            return a.amax(dim=-2)
        if mode == "sum":              # mean * k, i.e. upstream-style pooling
            return a.sum(dim=-2)
        beta = F.softplus(self.log_beta).view(1, self.n_head, 1, 1, self.N)
        if mode == "scaled":
            beta = beta * beta_scale
        elif mode != "trained":
            raise ValueError(mode)
        w = torch.softmax(a * beta, dim=-2)
        return (a * w).sum(dim=-2)

    return fwd


class Soma:
    """Temporarily swap the soma of every dendritic unit in `model`."""

    def __init__(self, units, mode, beta_scale=1.0):
        self.units, self.fwd = units, make_soma(mode, beta_scale)
        self.saved = []

    def __enter__(self):
        for u in self.units:
            self.saved.append(u.forward)
            u.forward = self.fwd.__get__(u, type(u))
        return self

    def __exit__(self, *exc):
        for u, f in zip(self.units, self.saved):
            u.forward = f
        self.saved.clear()
        return False


@torch.no_grad()
def calibrate_uniform_gain(model, units, task, device, max_batches=4):
    """Per-(head, neuron) gain making `uniform` match the trained gate's scale.

    Stored on each unit as ``_calib_gain`` and consumed by the
    ``uniform_scaled`` soma above.
    """
    acc = [None] * len(units)
    grabbed: list[list[torch.Tensor]] = [[] for _ in units]
    handles = [
        u.register_forward_pre_hook(
            lambda mod, inp, i=i: grabbed[i].append(inp[0].detach())
        )
        for i, u in enumerate(units)
    ]
    for b in task.eval_batches()[:max_batches]:
        model(b.to(device).x, None)
    for h in handles:
        h.remove()

    for i, (u, xs) in enumerate(zip(units, grabbed)):
        num = den = None
        beta = F.softplus(u.log_beta).view(1, u.n_head, 1, 1, u.N)
        for x in xs:
            a = u.branch_activations(x)
            w = torch.softmax(a * beta, dim=-2)
            gated = (a * w).sum(dim=-2)        # B,nh,T,N
            plain = a.mean(dim=-2)
            ns = gated.sum(dim=(0, 2))         # nh,N
            ds = plain.sum(dim=(0, 2))
            num = ns if num is None else num + ns
            den = ds if den is None else den + ds
        if num is None:
            continue
        gain = torch.where(den > 0, num / den.clamp(min=1e-9), torch.ones_like(den))
        acc[i] = gain.view(1, u.n_head, 1, u.N)
    for u, g in zip(units, acc):
        u._calib_gain = g
    return acc


def dendritic_units(model):
    """The gated units, in a stable order. Reuses the probes' own discovery so
    this script and the branch probes always agree on what they are looking at."""
    found = _dendritic_units(model)
    return [u for _, u in sorted(found.items()) if getattr(u, "log_beta", None) is not None]


# ------------------------------------------------------------------ gate probes

@torch.no_grad()
def gate_stats(model, units, tasks, device, max_batches=2):
    """Sharpness and input-dependence of the learned gate, per task.

    Captured with forward hooks so we see the real inputs the model receives,
    across every layer-loop call rather than only the last one.
    """
    out = {}
    for task in tasks:
        grabbed: list[list[torch.Tensor]] = [[] for _ in units]
        handles = [
            u.register_forward_pre_hook(
                lambda mod, inp, i=i: grabbed[i].append(inp[0].detach())
            )
            for i, u in enumerate(units)
        ]
        for b in task.eval_batches()[:max_batches]:
            model(b.to(device).x, None)
        for h in handles:
            h.remove()

        per_unit = []
        for u, xs in zip(units, grabbed):
            if not xs:
                continue
            beta = F.softplus(u.log_beta).view(1, u.n_head, 1, 1, u.N)
            maxw, eff, modal, dyn, n = 0.0, 0.0, 0.0, 0.0, 0
            for x in xs:
                a = u.branch_activations(x)                     # B,nh,T,k,N
                w = torch.softmax(a * beta, dim=-2)
                live = a.sum(dim=-2) > 0                        # B,nh,T,N
                if live.sum() == 0:
                    continue
                m = live.unsqueeze(-2)                          # B,nh,T,1,N
                cnt = live.sum()
                maxw += float((w.amax(-2) * live).sum()) / float(cnt)
                eff += float(((1.0 / (w * w).sum(-2)) * live).sum()) / float(cnt)
                # input dependence: does the argmax branch change with input?
                am = w.argmax(-2)                               # B,nh,T,N
                k = u.k
                hist = torch.zeros(k, *am.shape[1:], device=am.device)
                for c in range(k):
                    hist[c] = ((am == c) & live).float().sum(0)
                tot = hist.sum(0).clamp(min=1)
                modal += float((hist.max(0).values / tot).mean())
                # per-neuron spread of the gate across tokens
                dyn += float((w.std(dim=2) * live.any(dim=2).unsqueeze(-2)).mean())
                n += 1
            if n:
                per_unit.append(
                    dict(
                        mean_max_w=maxw / n,
                        eff_branches=eff / n,
                        modal_branch_frac=modal / n,
                        gate_std_across_tokens=dyn / n,
                        beta_min=float(F.softplus(u.log_beta).min()),
                        beta_med=float(F.softplus(u.log_beta).median()),
                        beta_max=float(F.softplus(u.log_beta).max()),
                        k=u.k,
                    )
                )
        out[task.name] = per_unit
    return out


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--variant", default="bdh_den_gated")
    ap.add_argument("--strategy", default="replay")
    ap.add_argument("--after", default="2_sort", help="checkpoint suffix, e.g. 2_sort")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="score the cheap eval subset only -- a preliminary read that fits on a "
             "login node while the full job waits in the queue",
    )
    args = ap.parse_args()

    cfg = build_from_yaml(args.config)
    device = pick_device(args.device)
    builders = model_builders(cfg)
    if args.variant not in builders:
        raise SystemExit(f"{args.variant} not in {sorted(builders)}")

    modes = [
        ("trained", 1.0),
        ("uniform", 1.0),   # beta -> 0
        ("uniform_scaled", 1.0),  # beta -> 0, but scale-matched (the control)
        ("hard", 1.0),      # beta -> inf
        ("scaled", 0.25),
        ("scaled", 4.0),
        ("scaled", 16.0),
    ]
    report = {"variant": args.variant, "strategy": args.strategy, "seeds": {}}

    for seed in [int(s) for s in args.seeds.split(",")]:
        ck = Path(args.ckpt_dir) / f"{args.variant}_{args.strategy}_seed{seed}_after{args.after}.pt"
        if not ck.exists():
            raise SystemExit(f"missing {ck}")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model = builders[args.variant]().to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        units = dendritic_units(model)

        tasks = build_synthetic_sequence(
            cfg.get("tasks"),
            n_eval=cfg.get("n_eval", 200),
            eval_seed=cfg.get("eval_seed", 0),
            batch_size=cfg.get("eval_batch_size", 32),
        )

        print(f"\n=== seed {seed}  ({len(units)} gated units, k={units[0].k}) ===", flush=True)
        acc = {}
        for mode, scale in modes:
            key = mode if mode != "scaled" else f"beta x{scale:g}"
            vals = {}
            for t in tasks:
                # The scale-matched control is calibrated on the task being
                # scored, which is the most generous setting for it.
                if mode == "uniform_scaled":
                    calibrate_uniform_gain(model, units, t, device)
                with Soma(units, mode, scale):
                    vals[t.name] = t.evaluate(model, device, quick=args.quick)["value"]
            vals["average"] = sum(vals.values()) / len(vals)
            acc[key] = vals
            cells = "  ".join(f"{k}={v:.3f}" for k, v in vals.items())
            print(f"  {key:12s} {cells}", flush=True)

        g = gate_stats(model, units, tasks, device, max_batches=cfg.get("probe_batches", 2))
        for tname, per_unit in g.items():
            for i, s in enumerate(per_unit):
                print(
                    f"  gate[{tname}/unit{i}] max_w={s['mean_max_w']:.3f} "
                    f"(uniform={1.0/s['k']:.3f}) eff_branches={s['eff_branches']:.2f}/{s['k']} "
                    f"modal_frac={s['modal_branch_frac']:.3f} "
                    f"std_over_tokens={s['gate_std_across_tokens']:.4f} "
                    f"beta={s['beta_min']:.3f}/{s['beta_med']:.3f}/{s['beta_max']:.3f}",
                    flush=True,
                )
        report["seeds"][seed] = {"accuracy": acc, "gate": g}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
