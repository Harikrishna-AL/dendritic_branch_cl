"""Sequential conditioning assay of the unmodified published Incentive Circuit.

Upstream is GPLv3+; this experiment harness is distributed under GPLv3+ too.
Run with the repository's Python environment; no upstream installation required.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/IncentiveCircuit"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bdh-biological-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bdh-biological-cache")
sys.path.insert(0, str(UPSTREAM / "src"))
import numpy as np
from incentive.circuit import IncentiveCircuit


def model(seed, arm, trials=700):
    # Same settings as upstream examples/plot_model.py and model-parameters.yml.
    return IncentiveCircuit(
        learning_rule="disabled" if arm == "no_plasticity" else "dpr",
        nb_timesteps=3, nb_trials=trials, nb_kc=10,
        nb_kc_odour_1=7, nb_kc_odour_2=6, nb_active_kcs=5,
        ltm_speed=.3, has_ltm=arm != "no_ltm",
        rng=np.random.RandomState(seed),
    )


def deliver(m, trials):
    """Continue state using upstream __call__; generator owns its time cursor."""
    def routine():
        for odor, reinforcement, paired in trials:
            for step in range(3):
                cs = np.zeros(2) if odor is None or step == 0 else np.eye(2)[odor]
                us = np.zeros(2)
                if reinforcement and ((paired and step == 2) or (not paired and step == 0)):
                    us[1] = 1.  # native shock amplitude and timing
                yield m._t // 3, step, cs, us
                m._t += 1
    m(routine=routine())


def compact_clone(m):
    """Clone only current state into fresh history; preserve independent RNG."""
    c = model(0, "full")
    c._w_m2v = m._w_m2v.copy()
    c._w_d2k = m._w_d2k.copy()
    c._learning_rule = m._learning_rule
    c._v[0] = m._v[m._t].copy()
    c.w_k2m[0] = m.w_k2m[m._t].copy()
    # bias/w_rest alias index 0 in upstream. Detach before writing state above
    # by restoring their baseline values explicitly from the source model.
    c.bias = m.bias.copy()
    c.w_rest = m.w_rest.copy()
    c.rng = copy.deepcopy(m.rng)
    return c


def probe(m, reset="none"):
    """Independent frozen-plasticity probes of A and B from one checkpoint."""
    values = []
    for odor in range(2):
        c = compact_clone(m)
        if reset == "activity":
            c._v[0] = c.bias.copy()
        elif reset == "weights":
            c.w_k2m[0] = np.broadcast_to(c.w_rest, c.w_k2m[0].shape)
        c._learning_rule = "disabled"  # upstream fallback: unchanged weights
        weights = c.w_k2m[0].copy()
        deliver(c, [(odor, False, True)])
        assert np.array_equal(weights, c.w_k2m[c._t])
        v = c._v[c._t, 6:12].reshape(3, 2)
        values.append(v[:, 0] - v[:, 1])
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "results/biological_memory_20261017")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, native = [], {}
    # Execute all three original routines before adapting stimulus schedules.
    for routine in ("reversal", "extinction", "unpaired"):
        m = model(2021, "full", trials=26)
        m(**{routine: True})
        assert m._t == 78 and np.isfinite(m._v).all()
        # Independent custom delivery must exactly match the published schedule,
        # including its initial unreinforced trials and intra-trial timing.
        reference = model(2021, "full", trials=26)
        shock_trials = {3, 5, 7, 9, 11}
        if routine != "extinction":
            shock_trials |= {14, 16, 18, 20, 22, 24}
        schedule = [(i % 2, i in shock_trials,
                     routine != "unpaired" or i in {3, 5, 7, 9, 11})
                    for i in range(26)]
        deliver(reference, schedule)
        np.testing.assert_array_equal(m._v, reference._v)
        np.testing.assert_array_equal(m.w_k2m, reference.w_k2m)
        native[routine + "_v"] = m._v
        native[routine + "_w"] = m.w_k2m
    np.savez_compressed(args.out / "native_routines.npz", **native)

    def record(m, seed, arm, punished, phase, count):
        before_w = m.w_k2m[m._t].copy()
        before_rng = copy.deepcopy(m.rng.get_state())
        for reset in ("none", "activity", "weights"):
            p = probe(m, reset)
            # Positive means avoidance of the originally punished odor.
            contrast = p[1 - punished] - p[punished]
            rows.append(dict(seed=seed, arm=arm, punished=punished, phase=phase,
                             count=count, reset=reset, susceptible=contrast[0],
                             restrained=contrast[1], longterm=contrast[2],
                             mean_contrast=contrast.mean()))
        assert np.array_equal(before_w, m.w_k2m[m._t])
        assert np.array_equal(before_rng[1], m.rng.get_state()[1])

    checkpoints = {}
    for seed in range(args.seeds):
        for punished in (0, 1):
            for arm in ("full", "no_ltm", "no_plasticity", "unpaired"):
                m = model(2021 + seed, arm)
                record(m, seed, arm, punished, "baseline", 0)
                acquisition = [(odor, odor == punished, arm != "unpaired")
                               for _ in range(6) for odor in (0, 1)]
                deliver(m, acquisition)
                record(m, seed, arm, punished, "acquisition", 6)
                checkpoints[f"s{seed}_p{punished}_{arm}_weights"] = m.w_k2m[m._t].copy()
                checkpoints[f"s{seed}_p{punished}_{arm}_activity"] = m._v[m._t].copy()
                for phase in ("blank", "extinction", "reversal"):
                    branch = compact_clone(m)
                    previous = 0
                    for count in (0, 1, 4, 16, 64, 256):
                        trials = []
                        for _ in range(count - previous):
                            if phase == "blank":
                                trials.append((None, False, True))
                            else:
                                trials.extend((odor, phase == "reversal" and odor != punished,
                                               arm != "unpaired") for odor in (0, 1))
                        deliver(branch, trials)
                        record(branch, seed, arm, punished, phase, count)
                        previous = count
        print(f"completed seed {seed + 1}/{args.seeds}", flush=True)
    np.savez_compressed(args.out / "acquisition_checkpoints.npz", **checkpoints)
    with (args.out / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    metadata = dict(upstream="https://github.com/InsectRobotics/IncentiveCircuit",
                    revision=revision, seeds=args.seeds, python=sys.version,
                    numpy=np.__version__, seconds=time.time()-started,
                    units="model trials/cycles; not biological hours",
                    score="mean over three MBON attraction-minus-avoidance contrasts; not accuracy",
                    native_check="original routines execute; no quantitative paper-figure comparison yet",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
