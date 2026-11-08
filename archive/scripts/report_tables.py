"""Render result tables straight from summary.json.

Every number in REPORT.md comes through here rather than being copied by hand,
so the report cannot drift from the runs that produced it. Values that were
never measured print as `n/a`, never as 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def fmt(node, digits: int = 3, *, pm: bool = True) -> str:
    """A scalar or an aggregated {mean, std, n} leaf, as `mean ± std`."""
    if node is None:
        return "n/a"
    if isinstance(node, dict) and "mean" in node:
        m, s, n = node["mean"], node.get("std", 0.0), node.get("n", 1)
        if m is None or (isinstance(m, float) and math.isnan(m)):
            return "n/a"
        if pm and n and n > 1:
            return f"{m:.{digits}f} ± {s:.{digits}f}"
        return f"{m:.{digits}f}"
    if isinstance(node, (int, float)):
        return "n/a" if (isinstance(node, float) and math.isnan(node)) else f"{node:.{digits}f}"
    return str(node)


def _rows(node):
    """Aggregated k x k matrix -> list of (mean_row, std_row)."""
    out = []
    for row in node:
        if isinstance(row, dict):
            out.append((row["mean"], row.get("std")))
        else:
            out.append((row, None))
    return out


def table_r_matrix(run: dict, label: str) -> str:
    names = run["task_names"]
    agg = run["aggregate"]["R"]
    n_seeds = run["n_seeds"]
    lines = [
        f"**{label}** — {run['metric']} "
        f"({'higher' if run['higher_is_better'] else 'lower'} is better), "
        f"{n_seeds} seed{'s' if n_seeds > 1 else ''}",
        "",
        "| after training | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 1),
    ]
    for i, (mean_row, std_row) in enumerate(_rows(agg)):
        cells = []
        for j, v in enumerate(mean_row):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("n/a")
            elif std_row and n_seeds > 1:
                cells.append(f"{v:.3f} ± {std_row[j]:.3f}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(f"| **{names[i]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_headline(runs: dict) -> str:
    lines = [
        "| run | params | final avg | forgetting | BWT | FWT |",
        "|---|---|---|---|---|---|",
    ]
    for label, run in runs.items():
        a = run["aggregate"]
        params = a.get("params")
        p = f"{int(params['mean']):,}" if isinstance(params, dict) else (
            f"{params:,}" if params else "n/a"
        )
        lines.append(
            f"| `{label}` | {p} | {fmt(a['final']['average'])} | "
            f"{fmt(a['forgetting']['average_excluding_last'])} | "
            f"{fmt(a['bwt'])} | {fmt(a['fwt'])} |"
        )
    return "\n".join(lines)


def table_forgetting(runs: dict) -> str:
    first = next(iter(runs.values()))
    names = first["task_names"]
    lines = ["| run | " + " | ".join(names) + " | mean (excl. last) |",
             "|" + "---|" * (len(names) + 2)]
    for label, run in runs.items():
        per = run["aggregate"]["forgetting"]["per_task"]
        cells = [fmt(per[n]) for n in names]
        lines.append(
            f"| `{label}` | " + " | ".join(cells) + " | "
            + fmt(run["aggregate"]["forgetting"]["average_excluding_last"]) + " |"
        )
    return "\n".join(lines)


def table_plasticity(runs: dict) -> str:
    first = next(iter(runs.values()))
    names = first["task_names"]
    lines = ["| run | " + " | ".join(f"{n} (seq/fresh)" for n in names) + " |",
             "|" + "---|" * (len(names) + 1)]
    for label, run in runs.items():
        pl = run["aggregate"].get("plasticity") or {}
        cells = []
        for n in names:
            e = pl.get(n)
            if not e:
                cells.append("n/a")
                continue
            s, f = e.get("steps_sequential"), e.get("steps_fresh")
            cells.append(f"{fmt(s, 0, pm=False)} / {fmt(f, 0, pm=False)}")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_sparsity(per_seed: dict) -> str:
    """Mean activation density per model/stream, over tasks, layers and seeds.

    ``sparsity_per_task_final`` is keyed [task][stream], not [stream].
    """
    import numpy as np

    acc: dict[tuple, list] = {}
    dead: dict[tuple, list] = {}
    width: dict[tuple, int] = {}
    for label, runs in per_seed.items():
        model = label.split("_")[0]
        for r in runs:
            sp = (r.get("probes") or {}).get("sparsity_per_task_final") or {}
            for _task, streams in sp.items():
                for stream, d in streams.items():
                    key = (model, stream)
                    for layer in d["per_layer"]:
                        acc.setdefault(key, []).append(layer["density"])
                        dead.setdefault(key, []).append(layer["dead_fraction"])
                        width[key] = layer["n_neurons"]
    if not acc:
        return "_no probe data_"

    lines = [
        "| model | stream | units/layer | mean density | mean sparsity | "
        "active units/layer | dead |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(acc):
        model, stream = key
        m = float(np.mean(acc[key]))
        n = width[key]
        lines.append(
            f"| {model} | `{stream}` | {n:,} | {m * 100:.1f}% | {(1 - m) * 100:.1f}% | "
            f"{m * n:,.0f} | {np.mean(dead[key]) * 100:.1f}% |"
        )
    return "\n".join(lines)


def table_overlap(per_seed: dict) -> str:
    """Mean off-diagonal neuron-set overlap between distinct tasks."""
    import numpy as np

    lines = ["| run | stream | mean top-5% Jaccard (off-diagonal) | mean profile cosine |",
             "|---|---|---|---|"]
    found = False
    for label, runs in per_seed.items():
        per_stream: dict[str, list] = {}
        per_stream_cos: dict[str, list] = {}
        for r in runs:
            om = (r.get("probes") or {}).get("overlap_matrix_final") or {}
            for stream, d in om.items():
                J = np.array(d["jaccard"], dtype=float)
                C = np.array(d["cosine"], dtype=float)
                off = ~np.eye(J.shape[0], dtype=bool)
                per_stream.setdefault(stream, []).append(np.nanmean(J[off]))
                per_stream_cos.setdefault(stream, []).append(np.nanmean(C[off]))
        for stream, vals in sorted(per_stream.items()):
            found = True
            lines.append(
                f"| `{label}` | `{stream}` | {np.mean(vals):.3f} | "
                f"{np.mean(per_stream_cos[stream]):.3f} |"
            )
    return "\n".join(lines) if found else "_no probe data_"


def load(res: Path):
    """Rebuild from per-seed files so tables work mid-sweep, as plots.py does."""
    from bdh_cl.scripts.plots import build_runs_from_seed_files, order_labels

    runs = build_runs_from_seed_files(res)
    runs = {k: runs[k] for k in order_labels(runs)}
    per_seed = {}
    for label in runs:
        per_seed[label] = [
            json.loads(f.read_text())
            for f in sorted(res.glob(f"{label}_seed*.json"))
            if not f.name.endswith(".partial.json")
        ]
    summary_path = res / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "elapsed_sec": float("nan"), "manifest": {}
    }
    summary["runs"] = runs
    return summary, per_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    summary, per_seed = load(res)
    runs = summary["runs"]

    print("## Headline\n")
    print(table_headline(runs))
    print("\n## Accuracy matrices\n")
    for label, run in runs.items():
        print(table_r_matrix(run, label), "\n")
    print("## Forgetting per task\n")
    print(table_forgetting(runs))
    print("\n## Plasticity (steps to target)\n")
    print(table_plasticity(runs))
    print("\n## Activation density\n")
    print(table_sparsity(per_seed))
    print("\n## Neuron-set overlap between tasks\n")
    print(table_overlap(per_seed))
    print(f"\n_elapsed {summary['elapsed_sec'] / 60:.1f} min; "
          f"git {summary['manifest'].get('git_hash', 'n/a')}_")


if __name__ == "__main__":
    main()
