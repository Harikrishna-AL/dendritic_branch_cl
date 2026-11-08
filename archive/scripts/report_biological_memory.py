"""Summarize seed-level uncertainty without treating paired odors as replicates."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bdh-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bdh-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/biological_memory_20261017"))
    parser.add_argument("--out", type=Path, default=Path("reports/biological_memory_20261017"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.input / "metrics.csv")
    keys = ["seed", "arm", "phase", "count", "reset"]
    metrics = ["susceptible", "restrained", "longterm", "mean_contrast"]
    paired = d.groupby(keys)[metrics].mean().reset_index()
    rng = np.random.RandomState(17)
    summaries = []
    for key, g in paired.groupby(keys[1:]):
        vals = g[metrics].to_numpy()
        boot = vals[rng.randint(len(vals), size=(4000, len(vals)))].mean(axis=1)
        row = dict(zip(keys[1:], key))
        for i, metric in enumerate(metrics):
            row[metric] = vals[:, i].mean()
            row[metric + "_lo"], row[metric + "_hi"] = np.quantile(boot[:, i], [.025, .975])
        summaries.append(row)
    s = pd.DataFrame(summaries)
    s.to_csv(args.out / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, phase in zip(axes, ["blank", "extinction", "reversal"]):
        for metric, color, label in zip(metrics[:3], ["#147d92", "#cc7c22", "#923b91"],
                                         ["Susceptible", "Restrained", "Long-term compartment"]):
            p = s[(s.arm == "full") & (s.reset == "none") & (s.phase == phase)].sort_values("count")
            ax.plot(p["count"], p[metric], label=label, color=color)
            ax.fill_between(p["count"], p[metric + "_lo"], p[metric + "_hi"], color=color, alpha=.2)
        ax.axhline(0, color="grey", linewidth=.6)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 1, 4, 16, 64, 256], labels=["0", "1", "4", "16", "64", "256"])
        ax.set_title(phase.capitalize())
        ax.set_xlabel("Blank trials" if phase == "blank" else "A/B cycles (2 trials each)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Avoidance contrast for original punished odor")
    axes[0].legend(fontsize=8)
    fig.suptitle("Published fly memory circuit: distinct memory compartments", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out / "retention.png", dpi=180)
    plt.close(fig)
    meta = json.loads((args.input / "metadata.json").read_text())
    lines = ["# Fly memory pilot", "", "Completed simulation, not a whole-brain or MaleCNS experiment.", "",
             f"Published Incentive Circuit revision `{meta['revision']}`; {meta['seeds']} noise seeds, both punished-odor assignments per seed.",
             "", "## Protocol and checks", "",
             "Six A/B conditioning cycles; three native timesteps per trial. Shock coincides with the final odor timestep. Original Python equations and default manuscript-example parameters are unchanged. Trials are model units, not biological hours.",
             "", "All three original reversal/extinction/unpaired routines execute. The custom stimulus driver matches their complete activity and weight histories exactly. This verifies the driver, but is not an independent quantitative reproduction of the paper's biological results.",
             "", "Probes use separate clones with plasticity disabled; assertions check frozen weights and source-checkpoint preservation. Positive contrast means stronger avoidance of the original punished odor; negative means the opposite. This is a circuit-response contrast, not percent correct or a calibrated behavioral probability.",
             "", "## Endpoints", "",
             "Mean contrasts across counterbalanced odors and seeds:", "",
             "| Arm | Phase | Susceptible | Restrained | Long-term compartment |",
             "|---|---|---:|---:|---:|"]
    for _, r in s[(s.reset == "none") & ((s.phase == "acquisition") | ((s["count"] == 256) & s.phase.isin(["blank", "extinction", "reversal"])))].iterrows():
        lines.append(f"| {r.arm} | {r.phase} | {r.susceptible:.3f} | {r.restrained:.3f} | {r.longterm:.3f} |")
    lines += ["", "![Retention](retention.png)", "", "## Interpretation and limits", "",
              "Inspect compartments separately: an average can conceal opposite memories after reversal. Removing the long-term pathway also removes a direct contributor to the average, so that average alone is not evidence for a consolidation mechanism.",
              "", "Unpaired shock is a timing manipulation, not a guaranteed zero-memory baseline: it can produce opposite-sign associations in this model. Plasticity-disabled acquisition is the no-learning control.",
              "", "Activity-reset probes retain learned synapses; weight-reset probes preserve current activity. Both use baseline state as defined by the implementation. Reset results and seed-bootstrap 95% intervals are in summary.csv. Intervals describe input-noise realizations of this circuit, not biological animals or uncertainty in the model's mechanism.",
              "", "Blank intervals include the upstream weak spontaneous KC noise and native synaptic updates. Repeated unrewarded odor exposure can change memory, so extinction is not equivalent to passive delay. Blank counts are single trials; extinction/reversal counts are two-trial A/B cycles and should not be compared as equal elapsed time.",
              "", "This first pilot measures learned odor association retention, not learned event order. It uses a reduced biologically motivated circuit, not detailed neuronal biophysics or the Google/Janelia wiring map. Follow-up: validate paper figure trajectories, test matched-duration interference with additional odors, inspect causal pathway interventions, and then attempt AB versus BA learning.",
              "", "Sources: [model and original code](https://github.com/InsectRobotics/IncentiveCircuit), [eLife paper](https://elifesciences.org/articles/75611), [Google MaleCNS release](https://www.research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).",
              "", f"Runtime: {meta['seconds']:.1f} s. Raw metrics, native histories, acquisition checkpoints and provenance are in `{args.input.resolve()}`."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
