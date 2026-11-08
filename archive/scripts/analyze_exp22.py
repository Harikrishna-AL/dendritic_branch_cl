"""Summarise Exp22 (continual RL on MinAtar) result JSONs.

Per arm (mean +- std over seeds):
  c1 / c2       train return on each game in cycle 1 and cycle 2
  fresh         train return of a fresh dqn on that game alone (reference)
  plast         c2 / fresh, averaged over games: < 1 means the agent learns the
                game worse than a fresh network would (loss of plasticity)
  final_eval    greedy return on each game after the last phase / fresh (forgetting)
  dormant       dormant fraction at the end of the run

    python bdh_cl/scripts/analyze_exp22.py bdh_cl/results/exp22
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GAMES = ["breakout", "space_invaders", "freeway", "asterix", "seaquest"]


def load(d: Path):
    runs, fresh = defaultdict(dict), defaultdict(list)
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        if f.name.startswith("fresh_"):
            fresh[a["games"]].append(r["log"][0]["train_return"])
        else:
            runs[a["tag"] or a["arm"]][a["seed"]] = r
    return runs, {g: float(np.mean(v)) for g, v in fresh.items()}


def ms(v):
    v = [x for x in v if x == x]
    return f"{np.mean(v):.2f}±{np.std(v):.2f}" if v else "n/a"


def main():
    d = Path(sys.argv[1])
    runs, fresh = load(d)
    print("fresh dqn (mean over seeds):", {g: round(v, 2) for g, v in fresh.items()})
    hdr = "arm | n | " + " | ".join(f"{g[:6]} c1→c2" for g in GAMES) + " | plast(c2/fresh) | c2/c1 | dormant_end"
    print(hdr)
    for arm, seeds in runs.items():
        cells, plast, ratio, dorm = [], [], [], []
        for g in GAMES:
            c1 = [s["summary"][g]["train_by_cycle"][0] for s in seeds.values()]
            c2 = [s["summary"][g]["train_by_cycle"][1] for s in seeds.values() if len(s["summary"][g]["train_by_cycle"]) > 1]
            cells.append(f"{ms(c1)} → {ms(c2)}")
        for s in seeds.values():
            p = [s["summary"][g]["train_by_cycle"][1] / fresh[g] for g in GAMES if g in fresh and fresh[g] > 0]
            q = [s["summary"][g]["train_by_cycle"][1] / s["summary"][g]["train_by_cycle"][0] for g in GAMES
                 if s["summary"][g]["train_by_cycle"][0] > 0]
            plast.append(np.mean(p) if p else np.nan)
            ratio.append(np.mean(q))
            dorm.append(s["log"][-1]["dormant_last"])
        print(f"{arm} | {len(seeds)} | " + " | ".join(cells) + f" | {ms(plast)} | {ms(ratio)} | {ms(dorm)}")
    print("\nfinal greedy eval / fresh (forgetting), mean over seeds:")
    for arm, seeds in runs.items():
        e = {g: np.mean([s["log"][-1]["eval"][g] for s in seeds.values()]) for g in GAMES}
        print(arm, {g: round(float(v), 2) for g, v in e.items()})


if __name__ == "__main__":
    main()
