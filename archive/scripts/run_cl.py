"""Step 3 entry point: weight-level continual learning, BDH vs GPT.

Runs, for each model and each strategy, a sequential pass over the task list
with 3 seeds, and writes per-seed JSON plus a seed-aggregated summary.

    python bdh_cl/scripts/run_cl.py --config bdh_cl/configs/exp1_synthetic_tiny.yaml

Also computes the fresh-model references the metrics need:

* ``random_init`` -- every task scored by an untrained model, for FWT.
* ``fresh_curves`` -- each task trained alone from random init, which is both
  the plasticity reference and the "how good can this get" ceiling per task.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl import baselines, wblog  # noqa: E402
from bdh_cl.cl.metrics import aggregate_seeds  # noqa: E402
from bdh_cl.cl.runner import (  # noqa: E402
    add_plasticity,
    evaluate_all,
    git_hash,
    make_optimizer,
    run_sequence,
    train_one_task,
)
from bdh_cl.cl.tasks import (  # noqa: E402
    build_mnist_sequence,
    build_synthetic_sequence,
)
from bdh_cl.config import as_dict, build_from_yaml  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.gpt_baseline import GPTBaseline  # noqa: E402
from bdh_cl.models.param_accounting import compare, count_module  # noqa: E402


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CONSOL_SUFFIX = "+consol"


def base_strategy(name: str) -> str:
    """'naive+consol' -> 'naive'. The suffix selects consolidation, not a
    different batch-sampling strategy, so the base name drives the batches."""
    return name[: -len(CONSOL_SUFFIX)] if name.endswith(CONSOL_SUFFIX) else name


def consolidation_cfg_for(cfg, name: str) -> dict | None:
    """The consolidation block, but only for strategies asking for it.

    Keeping it on the strategy axis means naive, naive+consol, replay and
    replay+consol run in one sweep with everything else held fixed, which is
    what makes the comparison interpretable.
    """
    if not name.endswith(CONSOL_SUFFIX):
        return None
    block = dict(cfg.get("consolidation") or {})
    block.update(cfg.get("_consol_overrides") or {})
    if not block:
        raise SystemExit(
            f"strategy {name!r} asks for consolidation but the config has no "
            "'consolidation:' block"
        )
    block["enabled"] = True
    return block


def model_builders(cfg):
    """Model name -> constructor.

    ``bdh_dendritic`` is baseline BDH with the DendriticNeuronUnit swapped in.
    It reuses the same harness, tasks, seeds and probes as the other two, so
    the comparison is like-for-like rather than two separately-tuned setups.
    """
    builders = {
        "bdh": lambda: BDHWrapper(cfg["bdh_spec"]),
        "gpt": lambda: GPTBaseline(cfg["gpt_spec"]),
    }
    dspec = cfg.get("dendritic_spec")
    if dspec is not None:
        builders["bdh_dendritic"] = lambda: BDHWrapper(dspec)
    for name, spec in (cfg.get("memory_specs") or {}).items():
        builders[f"bdh_mem_{name}"] = (lambda sp: (lambda: BDHWrapper(sp)))(spec)
    dpspec = cfg.get("deep_spec")
    if dpspec is not None:
        builders["bdh_deep"] = lambda: BDHWrapper(dpspec)
    nspec = cfg.get("narrow_spec")
    if nspec is not None:
        builders["bdh_narrow"] = lambda: BDHWrapper(nspec)
    for name, spec in (cfg.get("dendritic_specs") or {}).items():
        builders[f"bdh_den_{name}"] = (lambda sp: (lambda: BDHWrapper(sp)))(spec)
    return builders


def fresh_task_curves(build_model, tasks, train, device, *, verbose=True) -> dict:
    """Train each task alone from random init: the plasticity / ceiling reference."""
    curves, finals = {}, {}
    for t in tasks:
        if verbose:
            print(f"  fresh reference: {t.name}", flush=True)
        torch.manual_seed(train.seed)
        m = build_model().to(device)
        opt = make_optimizer(m, train)
        rng = np.random.default_rng(train.seed)
        rec = train_one_task(
            m, t, opt, train, device, rng,
            curve_metric=lambda mm, _t=t: _t.evaluate(mm, device, quick=True)["value"],
            log_prefix=f"fresh/{t.name} ",
            verbose=verbose,
        )
        curves[t.name] = rec["curve"]
        finals[t.name] = t.evaluate(m, device)["value"]
        del m, opt
        if device.type == "mps":
            torch.mps.empty_cache()
    return {"curves": curves, "single_task_final": finals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", default=None, help="override, e.g. 'bdh' or 'bdh,gpt'")
    ap.add_argument("--strategies", default=None)
    ap.add_argument("--seeds", default=None, help="e.g. '0,1,2'")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-probes", action="store_true")
    # Consolidation overrides, so one config can drive the whole sweep over
    # signal / granularity / strength without a config file per cell.
    ap.add_argument("--consol-mode", default=None, choices=["scale", "penalty", "mask"])
    ap.add_argument("--consol-signal", default=None, choices=["activity", "synaptic"])
    ap.add_argument("--consol-granularity", default=None, choices=["branch", "neuron"])
    ap.add_argument("--consol-lam", type=float, default=None)
    ap.add_argument("--consol-cap", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = build_from_yaml(args.config)
    over = {
        k: v
        for k, v in (
            ("mode", args.consol_mode),
            ("signal", args.consol_signal),
            ("granularity", args.consol_granularity),
            ("lam", args.consol_lam),
            ("cap", args.consol_cap),
        )
        if v is not None
    }
    if over:
        cfg["_consol_overrides"] = over
    device = pick_device(args.device)
    verbose = not args.quiet

    tag = args.tag or cfg.get("name", Path(args.config).stem)
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"

    models = (args.models or ",".join(cfg.get("models", ["bdh", "gpt"]))).split(",")
    strategies = (args.strategies or ",".join(cfg.get("strategies", ["naive"]))).split(",")
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg.get("seeds", [0, 1, 2]))]

    task_names = cfg.get("tasks")
    family = cfg.get("family", "synthetic")
    if family not in ("synthetic", "mnist"):
        raise SystemExit(f"family {family!r} not wired into this script yet")

    def build_tasks():
        """Fresh task objects per (model, seed) so eval sets match across models."""
        if family == "mnist":
            from bdh_cl.data import mnist as _m

            splits = cfg.get("mnist_splits")
            return build_mnist_sequence(
                cfg["mnist_cache"],
                side=cfg.get("mnist_side", 14),
                n_eval=cfg.get("n_eval", 200),
                eval_seed=cfg.get("eval_seed", 0),
                batch_size=cfg.get("eval_batch_size", 32),
                task_tags=cfg.get("mnist_task_tags", False),
                splits=tuple(tuple(c) for c in splits) if splits else _m.SPLITS,
            )
        return build_synthetic_sequence(
            task_names,
            n_eval=cfg.get("n_eval", 200),
            eval_seed=cfg.get("eval_seed", 0),
            batch_size=cfg.get("eval_batch_size", 32),
        )

    train = cfg["train_spec"]

    if family == "mnist":
        # The rendered length is fixed by the image size, so the model's context
        # must be at least that wide or collate() raises. Derive it rather than
        # leaving it to be set by hand in the YAML and silently mismatched.
        from bdh_cl.data import mnist as _mnist

        side = cfg.get("mnist_side", 14)
        need = _mnist.rendered_len(side) - 1
        if train.block_size < need:
            raise SystemExit(
                f"block_size={train.block_size} too small for {side}x{side} "
                f"images; need >= {need}"
            )
        # The pools define the task list, so the YAML's `tasks:` is advisory.
        task_names = [t.name for t in build_tasks()]
    builders = model_builders(cfg)

    param_report = compare(cfg["bdh_spec"], cfg["gpt_spec"])
    print(
        f"device={device}  tasks={task_names}  models={models}  "
        f"strategies={strategies}  seeds={seeds}\n"
        f"BDH {param_report['bdh']['params']['total']:,} params | "
        f"GPT {param_report['gpt']['params']['total']:,} params | "
        f"rel {param_report['rel_error_gpt_vs_bdh']:+.2%} "
        f"(matched: {param_report['matched_within_10pct']})",
        flush=True,
    )

    manifest = {
        "tag": tag,
        "config_path": str(args.config),
        "config": as_dict({k: v for k, v in cfg.items() if not k.endswith("_spec")}),
        "bdh_spec": as_dict(cfg["bdh_spec"]),
        "gpt_spec": as_dict(cfg["gpt_spec"]),
        "train_spec": as_dict(train),
        "param_report": param_report,
        "device": str(device),
        "git_hash": git_hash(),
        "models": models,
        "strategies": strategies,
        "seeds": seeds,
        "tasks": task_names,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    all_runs: dict[str, list[dict]] = {}
    references: dict[str, dict] = {}
    t_start = time.perf_counter()

    # Seed is the OUTER loop so that seed 0 yields a complete comparison across
    # every model and strategy early; later seeds only add error bars. With the
    # model outermost, an interrupted sweep leaves one model entirely unrun.
    for seed in seeds:
        # One task list per (model, seed) so eval sets are identical across models.
        for mname in models:
            tspec = dataclasses.replace(train, seed=seed)
            tasks = build_tasks()

            # Fresh references, computed once per (model, seed) and cached on
            # disk. They cost as much as a whole sequential run, so a rerun
            # after a crash must not pay for them twice. The cache is keyed by
            # the settings that actually determine the references.
            ref_key = f"{mname}_seed{seed}"
            ref_file = out_dir / f"reference_{ref_key}.json"
            ref_sig = {
                "tasks": task_names,
                "steps_per_task": tspec.steps_per_task,
                "batch_size": tspec.batch_size,
                "block_size": tspec.block_size,
                "lr": tspec.lr,
                "n_eval": cfg.get("n_eval", 200),
            }
            if ref_key not in references and ref_file.exists():
                cached = json.loads(ref_file.read_text())
                if cached.get("signature") == ref_sig:
                    print(f"[{mname} seed{seed}] reusing cached references "
                          f"from {ref_file.name}", flush=True)
                    references[ref_key] = cached
                else:
                    print(f"[{mname} seed{seed}] cached references are stale "
                          f"(config changed); recomputing", flush=True)

            if ref_key not in references:
                print(f"[{mname} seed{seed}] fresh references", flush=True)
                torch.manual_seed(seed)
                m0 = builders[mname]().to(device)
                rand_init = [e["value"] for e in evaluate_all(m0, tasks, device, verbose=verbose)]
                del m0
                if device.type == "mps":
                    torch.mps.empty_cache()
                fr = fresh_task_curves(builders[mname], tasks, tspec, device, verbose=verbose)
                fr["random_init"] = rand_init
                fr["signature"] = ref_sig
                references[ref_key] = fr
                ref_file.write_text(json.dumps(as_dict(fr), indent=2, default=str) + "\n")
            ref = references[ref_key]

            for sname in strategies:
                label = f"{mname}_{sname}"
                print(f"\n=== {label} seed={seed} ===", flush=True)
                wblog.init(
                    cfg.get("wandb"),
                    name=f"{tag}-{label}-s{seed}",
                    config={
                        "model": mname, "strategy": sname, "seed": seed,
                        "tasks": task_names, "preset": cfg.get("preset"),
                        "params": count_module(builders[mname]())["total"],
                        **as_dict(train),
                    },
                    tags=[mname, sname, tag, f"seed{seed}"],
                )
                strat = baselines.get_strategy(base_strategy(sname))
                summary = run_sequence(
                    builders[mname],
                    tasks,
                    tspec,
                    device,
                    strategy=sname,
                    strategy_batch_fn=strat,
                    label=label,
                    out_dir=out_dir,
                    ckpt_dir=ckpt_dir if cfg.get("save_checkpoints", True) else None,
                    collect_probes=not args.no_probes,
                    probe_batches=cfg.get("probe_batches", 4),
                    random_init_scores=ref["random_init"],
                    verbose=verbose,
                    consolidation_cfg=consolidation_cfg_for(cfg, sname),
                )
                # Plasticity target: a fraction of what a fresh model achieves
                # on *that same task* alone, so the target is attainable for
                # every task by construction. A single shared target would be
                # trivial for the easy tasks and unreachable for the hard ones.
                frac = cfg.get("plasticity_target_frac", 0.5)
                targets = {
                    name: frac * v for name, v in ref["single_task_final"].items()
                }
                summary = add_plasticity(summary, ref["curves"], targets)
                summary["plasticity_target"] = {
                    "frac_of_single_task_ceiling": frac,
                    "per_task_target": targets,
                    "single_task_final": ref["single_task_final"],
                }
                summary["model"] = mname
                summary["params"] = count_module(builders[mname]())["total"]
                wblog.finish()
                all_runs.setdefault(label, []).append(summary)
                # Rewrite with plasticity attached.
                (out_dir / f"{label}_seed{seed}.json").write_text(
                    json.dumps(as_dict(summary), indent=2, default=str) + "\n"
                )

    # ---- aggregate over seeds
    agg = {}
    for label, runs in all_runs.items():
        slim = [
            {
                k: v
                for k, v in r.items()
                if k in {"R", "final", "forgetting", "bwt", "fwt", "plasticity", "params"}
            }
            for r in runs
        ]
        agg[label] = {
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "aggregate": aggregate_seeds(slim),
            "task_names": runs[0]["task_names"],
            "metric": runs[0]["metric"],
            "higher_is_better": runs[0]["higher_is_better"],
        }

    elapsed = time.perf_counter() - t_start
    out = {
        "manifest": manifest,
        "elapsed_sec": elapsed,
        "runs": agg,
        "references": {k: as_dict(v) for k, v in references.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(as_dict(out), indent=2, default=str) + "\n")

    print(f"\n{'=' * 64}\nDone in {elapsed / 60:.1f} min. Wrote {out_dir}/summary.json")
    for label, a in agg.items():
        f = a["aggregate"]["final"]["average"]
        g = a["aggregate"]["forgetting"]["average_excluding_last"]
        b = a["aggregate"]["bwt"]
        print(
            f"  {label:16s} final {f['mean']:+.3f}+-{f['std']:.3f}  "
            f"forget {g['mean']:+.3f}+-{g['std']:.3f}  bwt {b['mean']:+.3f}+-{b['std']:.3f}"
        )


if __name__ == "__main__":
    main()
