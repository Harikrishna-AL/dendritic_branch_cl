"""Neuron-set overlap *through* a sequence, not just at the end.

The overlap matrix stored by the runner is computed on the **final** model.
Under naive finetuning that model has forgotten every task but the last, so it
processes all input distributions with the same machinery almost by
definition -- high overlap there may be a *consequence* of forgetting rather
than evidence about the mechanism.

The trajectory probe stores each task's top-k neuron set under the model state
after every task, so we can ask the better-posed question: at the moment the
model actually knows a task, are its neurons distinct from the other tasks'?

The cleanest test of all is the `joint` strategy, where one model knows all
tasks at once -- there, "do different tasks use different neurons?" has an
unambiguous answer.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else float("nan")


def analyse(path: Path) -> dict:
    d = json.loads(path.read_text())
    tr = (d.get("probes") or {}).get("sparsity_trajectory")
    if not tr:
        return {}
    names = d["task_names"]
    out = {}
    for state in names:  # model state after training `state`
        per_stream = {}
        for stream in ("x_sparse", "y_sparse", "mlp_hidden"):
            pairs = {}
            for a, b in combinations(names, 2):
                if stream not in tr[state].get(a, {}):
                    continue
                la = tr[state][a][stream]["per_layer"]
                lb = tr[state][b][stream]["per_layer"]
                vals = [jaccard(x["top_k"], y["top_k"]) for x, y in zip(la, lb)]
                pairs[f"{a}|{b}"] = sum(vals) / len(vals)
            if pairs:
                per_stream[stream] = {
                    "pairs": pairs,
                    "mean": sum(pairs.values()) / len(pairs),
                }
        out[state] = per_stream
    return {"label": d.get("label"), "task_names": names, "by_state": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)

    rows = []
    for f in sorted(res.glob("*_seed*.json")):
        if f.name.endswith(".partial.json"):
            continue
        a = analyse(f)
        if a:
            rows.append((f.name, a))

    for name, a in rows:
        print(f"\n=== {a['label']} ({name}) ===")
        for state, streams in a["by_state"].items():
            for stream, v in streams.items():
                pairs = "  ".join(f"{k} {x:.3f}" for k, x in v["pairs"].items())
                print(f"  after {state:8s} {stream:10s} mean {v['mean']:.3f}   {pairs}")

    out = res / "overlap_trajectory.json"
    out.write_text(json.dumps([{ "file": n, **a} for n, a in rows], indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
