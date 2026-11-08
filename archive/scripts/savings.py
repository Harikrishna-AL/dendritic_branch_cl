"""Savings: how much faster an environment is relearned when it recurs.

Final accuracy is the wrong headline for a lifelong stream. A system with a
persistent memory can look degraded the instant it returns to an old
environment and still recover in a fraction of the steps, and that recovery is
what the memory is for. Savings is the classic measure of exactly that, it
needs no task identity, and it survives smooth drift and absent boundaries --
all the things that made the accuracy matrix awkward here.

Three numbers per environment, read off the learning curves the runner already
records:

``retention``  accuracy on the environment at the moment of return, BEFORE any
               retraining. This is what the accuracy matrix already reports.
``steps_1`` / ``steps_2``
               steps to reach a target on the first and second visit.
``savings``   ``1 - steps_2 / steps_1``. Positive means the second visit was
               faster; 0 means the revisit was no cheaper than learning it new;
               negative means the model was actively worse off for having seen
               it before.

The target is a fraction of what a FRESH model reaches on that environment
alone, so it is attainable by construction and does not silently become
unreachable for a harder environment -- the same per-task-ceiling fix the
plasticity metric needed.

Curves are sampled every ``eval_every`` steps, so steps-to-target is quantised
to that grid. With 8 points per task the resolution is coarse; the reported
number is the first sample at or above target, and ``resolution`` is included
so nobody reads more precision into it than is there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def steps_to_target(curve: list, target: float, sustained: bool = True) -> tuple[int | None, int]:
    """First sampled step at or above ``target``, and the sampling interval.

    ``sustained`` requires the NEXT sample to clear the target as well, and it
    matters a lot here. The curves are quick-eval scores over 32 examples and
    carry the ~0.04 decode noise recorded elsewhere in NOTES, so they are
    non-monotone -- e.g. 0.19, 0.19, 0.50, 0.41, 0.81, 0.66, 0.75, 0.97. A bare
    first-crossing latches onto a single lucky sample, which makes
    steps-to-target noisy and savings noisier still, since savings is a ratio of
    two of them. Requiring two consecutive samples costs at most one interval of
    resolution and removes most of that.

    The last sample cannot be confirmed by a successor, so it is accepted on its
    own rather than being reported as "never reached".
    """
    if not curve:
        return None, 0
    step_gap = curve[1][0] - curve[0][0] if len(curve) > 1 else 0
    for i, (step, value) in enumerate(curve):
        if value < target:
            continue
        if not sustained or i + 1 >= len(curve) or curve[i + 1][1] >= target:
            return int(step), int(step_gap)
    return None, int(step_gap)


def savings_for_run(run: dict, frac: float = 0.8, sustained: bool = True) -> dict:
    """Per environment visited more than once, its savings on the later visit."""
    per_task = run.get("per_task", [])
    ceilings = (run.get("plasticity_target") or {}).get("single_task_final", {})

    visits: dict[str, list[dict]] = {}
    for rec in per_task:
        name = rec.get("task", "")
        base = name.split("#")[0]
        visits.setdefault(base, []).append(rec)

    out = {}
    for base, recs in visits.items():
        if len(recs) < 2:
            continue
        ceiling = ceilings.get(base) or max(
            (v for _s, v in recs[0].get("curve", [])), default=1.0
        )
        target = frac * ceiling
        first, gap1 = steps_to_target(recs[0].get("curve", []), target, sustained)
        second, gap2 = steps_to_target(recs[-1].get("curve", []), target, sustained)

        # Accuracy on this environment when we came back, before retraining.
        # Read from the accuracy matrix, not from the eval records: those are
        # positional and carry no task name, so matching on one silently
        # returned None for every row.
        retention = None
        revisit_idx = recs[-1].get("index")
        first_idx = recs[0].get("index")
        R = run.get("R") or []
        if revisit_idx and first_idx is not None and revisit_idx - 1 < len(R):
            row = R[revisit_idx - 1]
            if first_idx < len(row):
                retention = row[first_idx]

        sav = None
        if first is not None and second is not None and first > 0:
            sav = 1.0 - (second / first)
        elif first is not None and second is None:
            sav = None  # never re-reached target; savings undefined, not zero

        out[base] = {
            "target": target,
            "ceiling": ceiling,
            "steps_first_visit": first,
            "steps_second_visit": second,
            "savings": sav,
            "retention_at_return": retention,
            "resolution": max(gap1, gap2),
            "n_visits": len(recs),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory of *_seed*.json")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument(
        "--raw", action="store_true",
        help="bare first-crossing instead of requiring two consecutive samples",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = Path(args.results)
    report: dict[str, dict] = {}
    for f in sorted(res.glob("*_seed*.json")):
        if "reference_" in f.name or "partial" in f.name:
            continue
        run = json.loads(f.read_text())
        s = savings_for_run(run, args.frac, sustained=not args.raw)
        if s:
            report[f.stem] = s

    by_label: dict[str, list] = {}
    for key, envs in report.items():
        label = key.rsplit("_seed", 1)[0]
        for env, d in envs.items():
            by_label.setdefault(f"{label} :: {env}", []).append(d)

    print(f"{'run :: environment':44s} {'retention':>10s} {'v1':>6s} {'v2':>6s} {'savings':>9s}")
    for key, ds in sorted(by_label.items()):
        sav = [d["savings"] for d in ds if d["savings"] is not None]
        ret = [d["retention_at_return"] for d in ds if d["retention_at_return"] is not None]
        s1 = [d["steps_first_visit"] for d in ds if d["steps_first_visit"] is not None]
        s2 = [d["steps_second_visit"] for d in ds if d["steps_second_visit"] is not None]
        fmt = lambda xs, f="%.3f": (f % (sum(xs) / len(xs))) if xs else "  -"
        print(
            f"{key:44s} {fmt(ret):>10s} {fmt(s1, '%.0f'):>6s} "
            f"{fmt(s2, '%.0f'):>6s} {fmt(sav):>9s}   (n={len(ds)})"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
