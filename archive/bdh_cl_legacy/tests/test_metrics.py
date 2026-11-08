"""Metric definitions, checked against hand-computed cases.

These are the numbers the whole report rests on, so they are pinned against
matrices small enough to verify by hand, in both metric directions.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.cl.metrics import (  # noqa: E402
    CLResult,
    aggregate_seeds,
    plasticity,
    steps_to_target,
)


def test_forgetting_and_bwt_higher_is_better():
    # 3 tasks, accuracy. R[i][j]: perf on j after training through i.
    R = [
        [0.90, 0.10, 0.05],  # after T0
        [0.40, 0.95, 0.10],  # after T1  -> T0 dropped 0.90 -> 0.40
        [0.30, 0.50, 0.99],  # after T2  -> T0 0.30, T1 0.50
    ]
    r = CLResult(["t0", "t1", "t2"], "exact_match", True, R)

    assert r.final_performance()["average"] == pytest.approx((0.30 + 0.50 + 0.99) / 3)

    f = r.forgetting()["per_task"]
    assert f["t0"] == pytest.approx(0.90 - 0.30)
    assert f["t1"] == pytest.approx(0.95 - 0.50)
    assert f["t2"] == pytest.approx(0.0)  # last task, nothing to forget
    assert r.forgetting()["average_excluding_last"] == pytest.approx((0.60 + 0.45) / 2)

    # BWT = mean over j<2 of R[2][j] - R[j][j]
    assert r.bwt() == pytest.approx(((0.30 - 0.90) + (0.50 - 0.95)) / 2)


def test_forgetting_and_bwt_lower_is_better():
    """Same shape but bits-per-byte, where smaller is better. Forgetting must
    still come out positive when the model gets worse."""
    R = [
        [1.0, 4.0, 4.5],
        [2.5, 1.1, 4.0],
        [3.0, 2.0, 1.0],
    ]
    r = CLResult(["a", "b", "c"], "bits_per_byte", False, R)
    f = r.forgetting()["per_task"]
    # best (lowest) bpb on 'a' was 1.0, final is 3.0 -> forgot 2.0
    assert f["a"] == pytest.approx(2.0)
    assert f["b"] == pytest.approx(0.9)
    assert f["c"] == pytest.approx(0.0)
    # BWT negative = got worse
    assert r.bwt() == pytest.approx((-(3.0 - 1.0) + -(2.0 - 1.1)) / 2)
    assert r.bwt() < 0


def test_fwt_uses_zeroshot_vs_random_init():
    R = [
        [0.9, 0.30, 0.20],
        [0.4, 0.90, 0.35],
        [0.3, 0.50, 0.90],
    ]
    # A fresh model gets 0.10 on each task.
    r = CLResult(["t0", "t1", "t2"], "exact_match", True, R, random_init=[0.1, 0.1, 0.1])
    # FWT = mean over j>=1 of R[j-1][j] - b[j] = ((0.30-0.1) + (0.35-0.1))/2
    assert r.fwt() == pytest.approx((0.20 + 0.25) / 2)

    # With no random-init reference, FWT is undefined, not silently zero.
    assert math.isnan(CLResult(["a", "b"], "m", True, [[1, 0], [0, 1]]).fwt())


def test_nan_entries_are_tolerated():
    nan = float("nan")
    R = [[0.5, nan], [0.2, 0.8]]
    r = CLResult(["a", "b"], "exact_match", True, R)
    assert r.forgetting()["per_task"]["a"] == pytest.approx(0.3)
    assert r.final_performance()["average"] == pytest.approx(0.5)


def test_steps_to_target_and_plasticity():
    curve = [(0, 0.0), (10, 0.2), (20, 0.6), (30, 0.9)]
    assert steps_to_target(curve, 0.5, True) == 20
    assert steps_to_target(curve, 0.95, True) is None

    lower = [(0, 5.0), (10, 2.0), (20, 1.0)]
    assert steps_to_target(lower, 1.5, False) == 20

    p = plasticity(curve, [(0, 0.0), (10, 0.7)], 0.5, True)
    assert p["steps_sequential"] == 20 and p["steps_fresh"] == 10
    assert p["ratio"] == pytest.approx(2.0)  # 2x slower => lost plasticity

    # Never reaching the target must be reported, not coerced.
    p2 = plasticity(curve, [(0, 0.0)], 0.5, True)
    assert p2["steps_fresh"] is None and p2["ratio"] is None and p2["note"]


def test_aggregate_seeds_mean_std():
    res = [
        {"bwt": -0.1, "final": {"average": 0.5}, "R": [[1.0, 0.0]], "name": "x"},
        {"bwt": -0.3, "final": {"average": 0.7}, "R": [[1.0, 0.2]], "name": "x"},
    ]
    agg = aggregate_seeds(res)
    assert agg["bwt"]["mean"] == pytest.approx(-0.2)
    assert agg["bwt"]["std"] == pytest.approx(0.1)
    assert agg["bwt"]["n"] == 2
    assert agg["final"]["average"]["mean"] == pytest.approx(0.6)
    assert agg["R"][0]["mean"] == pytest.approx([1.0, 0.1])
    assert agg["name"] == "x"
