"""Recompute plasticity from saved curves, without retraining.

Plasticity is derived entirely from data already in the result JSONs --
each run's per-task learning curve and the fresh-model reference curves -- so
a change to the *definition* of the target does not require re-running
anything.

This exists because the first Step 3 run used a single shared target
(a fraction of the mean single-task ceiling). That is wrong when tasks have
very different ceilings: BDH-tiny reaches 1.00 on `copy` but ~0.30 on `add`,
so one global threshold is trivial for the easy tasks and unreachable for the
hard ones -- and an unreachable target reports as "never learned" rather than
"learned slowly". The target is now a fraction of *that task's own* ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl.metrics import aggregate_seeds  # noqa: E402
from bdh_cl.cl.runner import add_plasticity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--frac", type=float, default=None,
                    help="override the fraction-of-ceiling target")
    args = ap.parse_args()

    res = Path(args.results)
    summary_path = res / "summary.json"
    summary = json.loads(summary_path.read_text())
    refs = {p.stem.replace("reference_", ""): json.loads(p.read_text())
            for p in res.glob("reference_*.json")}
    if not refs:
        raise SystemExit(f"no reference_*.json in {res}")

    changed = []
    per_label: dict[str, list] = {}
    for run_path in sorted(res.glob("*_seed*.json")):
        run = json.loads(run_path.read_text())
        label, model = run.get("label"), run.get("model")
        seed = run.get("seed")
        if label is None:
            continue
        ref = refs.get(f"{model}_seed{seed}")
        if ref is None:
            print(f"  ! no reference for {run_path.name}; skipped")
            continue

        frac = args.frac
        if frac is None:
            frac = (run.get("plasticity_target") or {}).get(
                "frac_of_single_task_ceiling", 0.5
            )
        targets = {n: frac * v for n, v in ref["single_task_final"].items()}

        before = run.get("plasticity")
        run = add_plasticity(run, ref["curves"], targets)
        run["plasticity_target"] = {
            "frac_of_single_task_ceiling": frac,
            "per_task_target": targets,
            "single_task_final": ref["single_task_final"],
            "recomputed": True,
        }
        run_path.write_text(json.dumps(run, indent=2, default=str) + "\n")
        per_label.setdefault(label, []).append(run)
        if before != run["plasticity"]:
            changed.append(run_path.name)
        print(f"  {run_path.name}: targets " +
              ", ".join(f"{n}={t:.3f}" for n, t in targets.items()))

    # Re-aggregate so summary.json agrees with the per-seed files.
    for label, runs in per_label.items():
        if label not in summary["runs"]:
            continue
        slim = [
            {k: v for k, v in r.items()
             if k in {"R", "final", "forgetting", "bwt", "fwt", "plasticity", "params"}}
            for r in runs
        ]
        summary["runs"][label]["aggregate"] = aggregate_seeds(slim)
    summary["plasticity_recomputed"] = True
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(f"\nrewrote {len(per_label)} run label(s); "
          f"{len(changed)} file(s) had their plasticity change")


if __name__ == "__main__":
    main()
