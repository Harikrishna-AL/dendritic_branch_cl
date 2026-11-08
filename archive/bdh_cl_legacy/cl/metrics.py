"""Continual-learning metrics (Step 3).

Everything is derived from one object: the **accuracy matrix**

    R[i][j] = performance on task j's held-out set, measured after training
              through task i

with ``i, j`` in ``0..k-1`` and row ``i`` filled in immediately after task
``i`` finishes. The upper triangle (``j > i``) is *zero-shot* performance on
tasks not yet trained, which is what forward transfer needs.

Both metric directions are supported via ``higher_is_better``:
exact-match accuracy is higher-is-better, bits-per-byte is lower-is-better.
Internally every quantity is computed on a higher-is-better "score", so the
sign conventions below hold for both:

* **forgetting**\\[j] = best score ever seen on task j (rows 0..k-1) minus the
  final score. Positive = the model got worse than it once was.
* **BWT** = mean over j<k-1 of (final score on j) - (score on j right after
  learning j). Negative = learning later tasks hurt earlier ones.
* **FWT** = mean over j>=1 of (zero-shot score on j before training it) -
  (score of a freshly initialised model on j). Positive = earlier tasks
  helped before task j was ever seen.
* **plasticity** = steps needed to reach a target score on a new task,
  relative to a fresh model's steps. >1 means the sequentially-trained model
  has *lost* plasticity.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import numpy as np


def _score(value: float, higher_is_better: bool) -> float:
    """Map a metric onto a higher-is-better axis."""
    return value if higher_is_better else -value


@dataclasses.dataclass
class CLResult:
    """The accuracy matrix plus everything derived from it."""

    task_names: list[str]
    metric_name: str
    higher_is_better: bool
    R: list[list[float]]  # k x k, R[i][j]; NaN where not measured
    random_init: list[float] | None = None  # fresh-model score per task, for FWT
    plasticity: dict | None = None
    extra: dict = dataclasses.field(default_factory=dict)

    # -- helpers -----------------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.task_names)

    def _S(self) -> np.ndarray:
        return np.array(
            [[_score(v, self.higher_is_better) for v in row] for row in self.R],
            dtype=float,
        )

    # -- metrics -----------------------------------------------------------
    def final_performance(self) -> dict:
        """Per-task and average performance after the whole sequence."""
        last = self.R[-1]
        finite = [v for v in last if not math.isnan(v)]
        return {
            "per_task": dict(zip(self.task_names, last)),
            "average": float(np.mean(finite)) if finite else float("nan"),
        }

    def forgetting(self) -> dict:
        """best past score - final score, per task (positive = forgot)."""
        S = self._S()
        out: dict[str, float] = {}
        for j, name in enumerate(self.task_names):
            col = S[:, j]
            seen = col[~np.isnan(col)]
            if len(seen) == 0 or math.isnan(S[-1, j]):
                out[name] = float("nan")
                continue
            out[name] = float(np.max(seen) - S[-1, j])
        vals = [v for v in out.values() if not math.isnan(v)]
        return {
            "per_task": out,
            "average": float(np.mean(vals)) if vals else float("nan"),
            # The last task cannot have been forgotten; report the average over
            # the tasks that could be, which is the number usually quoted.
            "average_excluding_last": float(
                np.mean([out[n] for n in self.task_names[:-1] if not math.isnan(out[n])])
            )
            if self.k > 1
            else float("nan"),
        }

    def bwt(self) -> float:
        """Backward transfer: mean_{j<k-1} R[k-1][j] - R[j][j] (in score space)."""
        if self.k < 2:
            return float("nan")
        S = self._S()
        terms = [
            S[-1, j] - S[j, j]
            for j in range(self.k - 1)
            if not (math.isnan(S[-1, j]) or math.isnan(S[j, j]))
        ]
        return float(np.mean(terms)) if terms else float("nan")

    def fwt(self) -> float:
        """Forward transfer vs a freshly initialised model.

        Uses the zero-shot entry R[j-1][j] -- task j measured after training
        through j-1, i.e. before task j is ever trained -- against
        ``random_init[j]``.
        """
        if self.k < 2 or self.random_init is None:
            return float("nan")
        S = self._S()
        b = [_score(v, self.higher_is_better) for v in self.random_init]
        terms = [
            S[j - 1, j] - b[j]
            for j in range(1, self.k)
            if not (math.isnan(S[j - 1, j]) or math.isnan(b[j]))
        ]
        return float(np.mean(terms)) if terms else float("nan")

    def summary(self) -> dict:
        f = self.forgetting()
        return {
            "task_names": self.task_names,
            "metric": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "R": self.R,
            "random_init": self.random_init,
            "final": self.final_performance(),
            "forgetting": f,
            "bwt": self.bwt(),
            "fwt": self.fwt(),
            "plasticity": self.plasticity,
            **self.extra,
        }


# --------------------------------------------------------------------------
# Plasticity
# --------------------------------------------------------------------------


def steps_to_target(
    curve: Sequence[tuple[int, float]],
    target: float,
    higher_is_better: bool,
) -> int | None:
    """First step in ``curve`` (list of (step, value)) that reaches ``target``.

    ``None`` means the target was never reached, which is itself a finding and
    must be reported rather than silently coerced to the max step.
    """
    for step, value in curve:
        if math.isnan(value):
            continue
        if (value >= target) if higher_is_better else (value <= target):
            return step
    return None


def plasticity(
    sequential_curve: Sequence[tuple[int, float]],
    fresh_curve: Sequence[tuple[int, float]],
    target: float,
    higher_is_better: bool,
) -> dict:
    """Steps-to-target for a sequentially-trained model vs a fresh model."""
    s = steps_to_target(sequential_curve, target, higher_is_better)
    f = steps_to_target(fresh_curve, target, higher_is_better)
    ratio = None
    if s is not None and f is not None and f > 0:
        ratio = s / f
    return {
        "target": target,
        "steps_sequential": s,
        "steps_fresh": f,
        "ratio": ratio,
        "note": (
            "None means the target was never reached within the step budget."
            if (s is None or f is None)
            else ""
        ),
    }


# --------------------------------------------------------------------------
# Aggregation across seeds
# --------------------------------------------------------------------------


def aggregate_seeds(results: list[dict]) -> dict:
    """mean/std over seeds of every numeric leaf in a list of summary dicts.

    Structure is taken from the first result; leaves that are numeric become
    ``{"mean": ..., "std": ..., "n": ..., "values": [...]}``. Non-numeric
    leaves must agree across seeds and are passed through.
    """
    if not results:
        return {}

    def agg(values: list):
        first = values[0]
        if isinstance(first, dict):
            return {k: agg([v[k] for v in values]) for k in first}
        if isinstance(first, list):
            if first and isinstance(first[0], (list, dict)):
                return [agg([v[i] for v in values]) for i in range(len(first))]
            if first and all(isinstance(x, (int, float)) for x in first):
                arr = np.array(values, dtype=float)
                return {
                    "mean": np.nanmean(arr, axis=0).tolist(),
                    "std": np.nanstd(arr, axis=0).tolist(),
                    "n": len(values),
                }
            return first
        if isinstance(first, bool) or first is None or isinstance(first, str):
            return first
        if isinstance(first, (int, float)):
            arr = np.array(values, dtype=float)
            return {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "n": len(values),
                "values": [None if (isinstance(v, float) and math.isnan(v)) else v for v in values],
            }
        return first

    return agg(results)
