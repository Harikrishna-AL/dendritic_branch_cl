"""Step 5 plots, generated from the JSON written by run_cl.py.

    python bdh_cl/scripts/plots.py --results bdh_cl/results/<tag>

Forms, chosen by the data's job:

* accuracy matrix R  -> heatmap (magnitude over two categorical axes),
  one-hue sequential ramp, every cell directly labelled
* forgetting         -> grouped bars (magnitude by category, 2 models),
  error bars = std over seeds, every bar directly labelled
* sparsity per layer -> grouped bars over an ordinal layer index
* neuron overlap     -> heatmap per model, same sequential ramp
* learning curves    -> lines, one per task

Every chart carries direct value labels, which is the relief the palette's
contrast WARN requires, and REPORT.md repeats the numbers as tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import matplotlib.pyplot as plt  # noqa: E402

from bdh_cl.scripts import vizstyle as vs  # noqa: E402


#: Strategies read lower-bound -> mitigation -> upper-bound, and models group
#: together. Alphabetical order would put `joint` first, which is backwards.
STRATEGY_ORDER = {"naive": 0, "replay": 1, "joint": 2}
MODEL_ORDER = {
    "bdh": 0, "bdh_narrow": 1,
    "bdh_den_gated": 2, "bdh_den_sum": 3, "bdh_den_max": 4,
    "gpt": 5,
}


def split_label(lb: str) -> tuple[str, str]:
    """('bdh_den_gated_replay') -> ('bdh_den_gated', 'replay').

    Model names contain underscores, so splitting on the first one silently
    mapped every dendritic variant to 'bdh' and painted them all one colour.
    """
    model = max(
        (m for m in MODEL_ORDER if lb.startswith(m + "_")),
        key=len, default=lb.partition("_")[0],
    )
    strat = lb[len(model) + 1:] if lb.startswith(model + "_") else ""
    return model, strat


def order_labels(labels) -> list:
    def key(lb: str):
        model, strat = split_label(lb)
        return (MODEL_ORDER.get(model, 9), STRATEGY_ORDER.get(strat, 9), lb)
    return sorted(labels, key=key)


def is_joint(label: str) -> bool:
    """Joint is a performance ceiling, not a continual-learning method.

    Its batches contain every task from step 0, so its rows are progress
    through mixed training rather than a forgetting matrix, and its BWT/FWT
    are not comparable with the sequential strategies'.
    """
    return label.endswith("_joint")


def _mean(node):
    """Unwrap an aggregate leaf ({'mean':..,'std':..}) or a bare number."""
    if isinstance(node, dict) and "mean" in node:
        return node["mean"]
    return node


def _std(node):
    if isinstance(node, dict) and "std" in node:
        return node["std"]
    return 0.0


def _matrix(node, key: str = "mean") -> np.ndarray:
    """Unwrap an aggregated matrix.

    ``aggregate_seeds`` turns a k x k list of lists into a list of k row-dicts,
    each ``{"mean": [...], "std": [...]}``. A single-seed summary is still a
    plain list of lists.
    """
    rows = []
    for row in node:
        rows.append(row[key] if isinstance(row, dict) else row)
    return np.array(rows, dtype=float)


# --------------------------------------------------------------------------
# 1. Accuracy / bpb matrix heatmaps
# --------------------------------------------------------------------------


def plot_r_matrix(runs: dict, out_dir: Path, metric: str, higher_is_better: bool) -> None:
    labels = order_labels(runs)
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(1.6 + 2.7 * n, 3.4), squeeze=False)

    mats = {lb: _matrix(runs[lb]["aggregate"]["R"]) for lb in labels}
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()])
    vmin, vmax = float(finite.min()), float(finite.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    for col, (ax, lb) in enumerate(zip(axes[0], labels)):
        M = mats[lb]
        names = runs[lb]["task_names"]
        ax.imshow(M, cmap=vs.SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        # Row labels only on the leftmost panel, else they overlap the panel
        # to their left.
        ax.set_yticks(
            range(len(names)),
            [f"after {t}" for t in names] if col == 0 else [""] * len(names),
        )
        ax.set_title(lb + ("\n(ceiling: all tasks mixed from step 0)" if is_joint(lb) else ""),
                     pad=8, fontsize=10)
        ax.grid(False)
        vs.strip_spines(ax, keep=())
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=vs.cell_ink(v, vmin, vmax),
                )
        # A faint separator grid reads better than cell borders.
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=vs.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    direction = "higher is better" if higher_is_better else "lower is better"
    fig.suptitle(
        f"Performance on task j (columns) after training through task i (rows)\n"
        f"{metric} — {direction}; mean over seeds",
        fontsize=10.5, color=vs.INK, y=1.06,
    )
    vs.save(fig, out_dir / "r_matrix.png")


# --------------------------------------------------------------------------
# 2. Forgetting bars
# --------------------------------------------------------------------------


def plot_forgetting(runs: dict, out_dir: Path, metric: str) -> None:
    # Joint has nothing to forget by construction; showing it invites a
    # comparison that does not mean anything.
    labels = [lb for lb in order_labels(runs) if not is_joint(lb)]
    names = runs[labels[0]]["task_names"]
    x = np.arange(len(names))
    width = min(0.8 / len(labels), 0.34)

    fig, ax = plt.subplots(figsize=(1.6 + 1.35 * len(names), 3.6))
    for idx, lb in enumerate(labels):
        per = runs[lb]["aggregate"]["forgetting"]["per_task"]
        vals = [_mean(per[t]) for t in names]
        errs = [_std(per[t]) for t in names]
        # Colour carries the MODEL (the primary comparison). Strategy is a
        # second dimension, so it gets a texture rather than another hue --
        # two series sharing a colour would make the legend ambiguous, and
        # identity must never rest on colour alone.
        model, strat = split_label(lb)
        color = vs.MODEL_COLOR.get(model, vs.SERIES[idx % len(vs.SERIES)])
        hatch = {"naive": None, "replay": "///", "joint": "..."}.get(strat)
        off = (idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            x + off, vals, width * 0.92, label=lb, color=color,
            hatch=hatch, edgecolor=vs.SURFACE, linewidth=0.6,
            yerr=errs, capsize=3,
            error_kw={"ecolor": vs.INK_SECONDARY, "elinewidth": 1.1},
        )
        for b, v, e in zip(bars, vals, errs):
            top_of_bar = v + (e if np.isfinite(e) else 0.0)
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, top_of_bar),
                textcoords="offset points",
                xytext=(0, 4 if v >= 0 else -11),
                ha="center", fontsize=7.5, color=vs.INK_SECONDARY,
            )

    ax.axhline(0, color=vs.INK_MUTED, linewidth=0.9)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel(f"forgetting ({metric})")
    ax.set_title("Forgetting per task — best score ever reached minus final score", pad=10)
    ax.grid(axis="x", visible=False)
    vs.strip_spines(ax)
    # Legend below the axes: inside it collided with the tallest bars and
    # their value labels, which is exactly where the interesting numbers are.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncols=min(len(labels), 4), frameon=False,
    )
    # Headroom for value labels and the top of the error bars.
    top = 0.0
    for lb in labels:
        per = runs[lb]["aggregate"]["forgetting"]["per_task"]
        for t in names:
            v, e = _mean(per[t]), _std(per[t])
            if np.isfinite(v):
                top = max(top, v + (e if np.isfinite(e) else 0.0))
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), top * 1.18 + 0.02)
    fig.text(
        0.01, -0.22,
        "Higher = more forgetting. Error bars: std over seeds. "
        "A bar with a very large error bar is bimodal across seeds, not merely noisy.",
        fontsize=8, color=vs.INK_MUTED,
    )
    vs.save(fig, out_dir / "forgetting.png")


# --------------------------------------------------------------------------
# 3. Headline metrics: final performance, BWT, FWT
# --------------------------------------------------------------------------


def plot_headline(runs: dict, out_dir: Path, metric: str) -> None:
    labels = order_labels(runs)
    fields = [
        ("final average", lambda a: (_mean(a["final"]["average"]), _std(a["final"]["average"]))),
        ("forgetting (mean)", lambda a: (
            _mean(a["forgetting"]["average_excluding_last"]),
            _std(a["forgetting"]["average_excluding_last"]),
        )),
        ("BWT\n(sequential only)", lambda a: (_mean(a["bwt"]), _std(a["bwt"]))),
        ("FWT\n(sequential only)", lambda a: (_mean(a["fwt"]), _std(a["fwt"]))),
    ]
    fig, axes = plt.subplots(1, len(fields), figsize=(3.4 * len(fields), 4.2),
                             layout="constrained")
    for ax, (title, get) in zip(axes, fields):
        # BWT/FWT are undefined for joint (every task is present from step 0),
        # so those panels drop it rather than plot a meaningless bar.
        panel = [lb for lb in labels if not (is_joint(lb) and "only" in title)]
        vals, errs, cols, hatches = [], [], [], []
        for idx, lb in enumerate(panel):
            v, e = get(runs[lb]["aggregate"])
            vals.append(0.0 if v is None or not np.isfinite(v) else v)
            errs.append(0.0 if e is None or not np.isfinite(e) else e)
            m_, s_ = split_label(lb)
            cols.append(vs.MODEL_COLOR.get(m_, vs.SERIES[idx % len(vs.SERIES)]))
            hatches.append({"naive": None, "replay": "///", "joint": "..."}.get(s_))
        bars = ax.bar(range(len(panel)), vals, 0.6, color=cols, yerr=errs, capsize=3,
                      edgecolor=vs.SURFACE, linewidth=0.6,
                      error_kw={"ecolor": vs.INK_SECONDARY, "elinewidth": 1.1})
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
        for b, v, e in zip(bars, vals, errs):
            tip = v + (e if v >= 0 else -e)
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, tip),
                        textcoords="offset points", xytext=(0, 4 if v >= 0 else -11),
                        ha="center", fontsize=7, color=vs.INK_SECONDARY)
        ax.axhline(0, color=vs.INK_MUTED, linewidth=0.9)
        ax.set_xticks(range(len(panel)), panel, rotation=30, ha="right")
        ax.set_title(title, pad=14, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        vs.strip_spines(ax)

        span = max(abs(min(vals + [0])), abs(max(vals + [0])))
        if span < 1e-9:
            # An all-zero panel otherwise auto-scales to +-0.04 of pure noise
            # and reads as though something varied. Say plainly that it did not.
            ax.set_ylim(-1.0, 1.0)
            ax.text(
                0.5, 0.5, "all zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=vs.INK_MUTED,
            )
        else:
            ax.set_ylim(
                min(0, min(vals)) - 0.18 * span, max(0, max(vals)) + 0.18 * span
            )
    fig.suptitle(f"Continual-learning headline metrics ({metric})", fontsize=10.5, y=1.04)
    vs.save(fig, out_dir / "headline.png")


# --------------------------------------------------------------------------
# 4. Sparsity per layer
# --------------------------------------------------------------------------


def plot_sparsity(per_seed: dict, out_dir: Path) -> None:
    """Density (fraction of active units) per layer, faceted by model.

    Faceted rather than overlaid because "layer 3" is not the same thing in a
    4-layer BDH and an 8-layer Transformer, and the two have different layer
    counts — putting them on one axis would invite a comparison that the x
    axis does not support. The comparison that *is* supported — overall density
    — is stated in the panel subtitles.
    """
    rows = []
    for label, runs in per_seed.items():
        model = label.split("_")[0]
        for r in runs:
            sp = r.get("probes", {}).get("sparsity_per_task_final")
            if not sp:
                continue
            for task, streams in sp.items():
                for stream, d in streams.items():
                    for li, layer in enumerate(d["per_layer"]):
                        rows.append(
                            {"model": model, "task": task, "stream": stream,
                             "layer": li, "density": layer["density"]}
                        )
    if not rows:
        print("  (no probe data; skipping sparsity plot)")
        return

    models = sorted({r["model"] for r in rows})
    widths = [max(r["layer"] for r in rows if r["model"] == m) + 1 for m in models]
    fig, axes = plt.subplots(
        1, len(models), figsize=(2.2 + 0.95 * sum(widths), 3.8),
        squeeze=False, sharey=True,
        gridspec_kw={"width_ratios": widths},
    )

    for ax, model in zip(axes[0], models):
        mrows = [r for r in rows if r["model"] == model]
        streams = sorted({r["stream"] for r in mrows})
        n_layers = max(r["layer"] for r in mrows) + 1
        x = np.arange(n_layers)
        width = min(0.78 / len(streams), 0.36)
        overall = []
        for idx, stream in enumerate(streams):
            vals, errs = [], []
            for li in range(n_layers):
                d = [r["density"] for r in mrows if r["stream"] == stream and r["layer"] == li]
                vals.append(float(np.mean(d)) if d else np.nan)
                errs.append(float(np.std(d)) if d else 0.0)
            overall.extend(v for v in vals if np.isfinite(v))
            off = (idx - (len(streams) - 1) / 2) * width
            bars = ax.bar(
                x + off, vals, width * 0.9, label=stream,
                color=vs.STREAM_COLOR.get(stream, vs.SERIES[idx % len(vs.SERIES)]),
                yerr=errs, capsize=2.5,
                error_kw={"ecolor": vs.INK_SECONDARY, "elinewidth": 1.0},
            )
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.annotate(
                        f"{v * 100:.0f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color=vs.INK_SECONDARY,
                    )
        ax.set_xticks(x, [str(i) for i in range(n_layers)])
        ax.set_xlabel("layer")
        ax.set_title(
            f"{model}  —  mean {np.mean(overall) * 100:.1f}% active",
            pad=8,
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        vs.strip_spines(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncols=len(streams))

    axes[0][0].set_ylabel("fraction of units active (> 0)")
    fig.suptitle(
        "Activation density per layer — BDH latent neurons vs Transformer MLP hidden units",
        fontsize=10.5, y=1.03,
    )
    fig.text(
        0.01, -0.16,
        "Lower = sparser. Bar labels are percentages. Averaged over tasks and "
        "seeds; error bars std.",
        fontsize=8, color=vs.INK_MUTED,
    )
    vs.save(fig, out_dir / "sparsity.png")


# --------------------------------------------------------------------------
# 5. Neuron overlap heatmaps
# --------------------------------------------------------------------------


def plot_overlap(per_seed: dict, out_dir: Path) -> None:
    panels = []
    for label in order_labels(per_seed):
        runs = per_seed[label]
        mats: dict[str, list] = {}
        names = None
        for r in runs:
            om = r.get("probes", {}).get("overlap_matrix_final")
            if not om:
                continue
            for stream, d in om.items():
                mats.setdefault(stream, []).append(np.array(d["jaccard"], dtype=float))
                names = d["task_names"]
        for stream, ms in mats.items():
            panels.append((f"{label}  ·  {stream}", np.nanmean(np.stack(ms), axis=0), names))

    if not panels:
        print("  (no probe data; skipping overlap plot)")
        return

    n = len(panels)
    cols = min(n, 3)
    rws = math.ceil(n / cols)
    # constrained_layout, and real height per row: with a tight figure the
    # panel titles land on top of the previous row's tick labels.
    fig, axes = plt.subplots(
        rws, cols, figsize=(3.3 * cols, 3.9 * rws), squeeze=False,
        layout="constrained",
    )
    vmin, vmax = 0.0, 1.0
    for pi, (ax, (title, M, names)) in enumerate(zip(axes.ravel(), panels)):
        ax.imshow(M, cmap=vs.SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(
            range(len(names)), names if pi % cols == 0 else [""] * len(names)
        )
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.grid(False)
        vs.strip_spines(ax, keep=())
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5, color=vs.cell_ink(M[i, j], vmin, vmax))
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=vs.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Neuron-set overlap between tasks — Jaccard of the top-5% most active units\n"
        "1.00 = the same units serve both tasks; 0.00 = fully separated "
        "(chance ≈ 0.03, independent of layer width)",
        fontsize=10.5,
    )
    vs.save(fig, out_dir / "overlap.png")


# --------------------------------------------------------------------------
# 6. Learning curves
# --------------------------------------------------------------------------


def plot_curves(per_seed: dict, out_dir: Path, metric: str) -> None:
    labels = order_labels(per_seed)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.3),
                             squeeze=False, sharey=True)
    for ax, lb in zip(axes[0], labels):
        runs = per_seed[lb]
        names = runs[0]["task_names"]
        offset = 0
        for ti, name in enumerate(names):
            xs_all, ys_all = [], []
            for r in runs:
                rec = next((p for p in r["per_task"] if p["task"] == name), None)
                if not rec or not rec["curve"]:
                    continue
                xs_all.append([s for s, _ in rec["curve"]])
                ys_all.append([v for _, v in rec["curve"]])
            if not ys_all:
                continue
            xs = np.array(xs_all[0]) + offset
            ys = np.nanmean(np.array(ys_all, dtype=float), axis=0)
            color = vs.SERIES[ti % len(vs.SERIES)]
            ax.plot(xs, ys, color=color, label=name, marker="o", markersize=3.5)
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(4, 0), fontsize=7.5, color=vs.INK_SECONDARY, va="center")
            steps = max(xs_all[0]) + 1
            if ti < len(names) - 1:
                ax.axvline(offset + steps, color=vs.GRID, linewidth=1.2)
            offset += steps
        ax.set_title(lb, pad=8)
        ax.set_xlabel("cumulative training step")
        vs.strip_spines(ax)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(loc="lower right", ncols=2)
    fig.suptitle(
        "Performance on the task currently being trained (vertical rules = task switch)",
        fontsize=10.5, y=1.04,
    )
    vs.save(fig, out_dir / "curves.png")


# --------------------------------------------------------------------------



def build_runs_from_seed_files(res: Path) -> dict:
    """Aggregate per-seed run files into the shape summary.json would hold.

    Lets plots be produced mid-sweep, and keeps a partially-completed sweep
    usable instead of worthless.
    """
    import re
    sys.path.insert(0, str(ROOT.parent))
    from bdh_cl.cl.metrics import aggregate_seeds

    by_label: dict[str, list] = {}
    for f in sorted(res.glob("*_seed*.json")):
        if f.name.endswith(".partial.json"):
            continue
        d = json.loads(f.read_text())
        label = d.get("label")
        if label:
            by_label.setdefault(label, []).append(d)

    runs = {}
    for label, ds in by_label.items():
        slim = [
            {k: v for k, v in d.items()
             if k in {"R", "final", "forgetting", "bwt", "fwt", "plasticity", "params"}}
            for d in ds
        ]
        runs[label] = {
            "n_seeds": len(ds),
            "seeds": [d.get("seed") for d in ds],
            "aggregate": aggregate_seeds(slim),
            "task_names": ds[0]["task_names"],
            "metric": ds[0]["metric"],
            "higher_is_better": ds[0]["higher_is_better"],
        }
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="a results/<tag> directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = Path(args.results)
    out_dir = Path(args.out) if args.out else res / "plots"

    # summary.json is only written when the whole sweep finishes. Rebuild the
    # aggregate from the per-seed files so a sweep can be plotted while it is
    # still running -- and so a sweep that dies part-way is still plottable.
    runs = build_runs_from_seed_files(res)
    if not runs:
        raise SystemExit(f"no completed *_seed*.json runs in {res}")

    first = next(iter(runs.values()))
    metric = first["metric"]
    higher = first["higher_is_better"]

    per_seed: dict[str, list] = {}
    for label in runs:
        files = sorted(res.glob(f"{label}_seed*.json"))
        per_seed[label] = [json.loads(f.read_text()) for f in files]

    vs.apply_style()
    print(f"plotting {res} ({len(runs)} runs, metric={metric})")
    plot_r_matrix(runs, out_dir, metric, higher)
    plot_forgetting(runs, out_dir, metric)
    plot_headline(runs, out_dir, metric)
    plot_curves(per_seed, out_dir, metric)
    plot_sparsity(per_seed, out_dir)
    plot_overlap(per_seed, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
