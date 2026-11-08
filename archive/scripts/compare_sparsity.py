"""Compare BDH vs GPT density at MATCHED training steps, across seeds.

Comparing endpoints is not enough: the two models train at very different
speeds (BDH ~13x slower per token at this config), so a wall-clock or
"latest value" comparison would be meaningless. Every number here is aligned
on optimizer step.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(res: Path) -> dict:
    runs = defaultdict(list)
    for f in sorted(res.glob("*.json")):
        d = json.loads(f.read_text())
        runs[d["model"]].append(d)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    runs = load(Path(args.results))
    if not runs:
        raise SystemExit("no results")

    print("=== final (step 2999) ===")
    print(f"{'model':6s} {'seeds':>5s} {'bpb(val)':>16s} {'bpb(train)':>11s} "
          f"{'tok/s':>10s}  density per stream")
    for model, ds in sorted(runs.items()):
        bpb = [d.get("bits_per_byte_val", float("nan")) for d in ds]
        bpb_tr = [d.get("bits_per_byte_train", d.get("bits_per_byte_final", float("nan")))
                  for d in ds]
        tps = [d["tokens_per_sec"] for d in ds]
        streams = sorted(ds[0]["density_final"])
        parts = []
        for s in streams:
            v = [d["density_final"][s]["mean"] for d in ds]
            n = ds[0]["density_final"][s]["n_neurons"]
            act = np.mean(v) * n
            parts.append(f"{s} {np.mean(v)*100:.1f}%±{np.std(v)*100:.1f} ({act:.0f}/{n})")
        print(f"{model:6s} {len(ds):5d} {np.nanmean(bpb):8.4f}±{np.nanstd(bpb):.4f} "
              f"{np.nanmean(bpb_tr):11.4f} {np.mean(tps):10,.0f}  " + "  ".join(parts))

    print("\n=== density at matched steps (mean ± std over seeds) ===")
    steps = sorted({h["step"] for ds in runs.values() for d in ds for h in d["history"]})
    show = [s for s in steps if s % 500 == 0 or s == steps[-1]]
    cols = []
    for model, ds in sorted(runs.items()):
        for s in sorted(ds[0]["density_final"]):
            cols.append((model, s))
    header = "step   " + "  ".join(f"{m}/{s:10s}" for m, s in cols)
    print(header)
    for st in show:
        row = [f"{st:5d} "]
        for model, stream in cols:
            vals = []
            for d in runs[model]:
                h = next((x for x in d["history"] if x["step"] == st), None)
                if h and stream in h["density"]:
                    vals.append(h["density"][stream]["mean"])
            row.append(f"{np.mean(vals)*100:5.1f}±{np.std(vals)*100:4.1f}" if vals else "    -    ")
        print("  ".join(row))

    print("\nNote: 'active' counts differ because layer widths differ "
          "(BDH n=4*N per layer, GPT 4*n_embd).")


if __name__ == "__main__":
    main()
