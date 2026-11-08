"""The interference probe's measurements mean what they claim.

Each metric is checked on a case where the right answer is known by
construction, so a sign error or a wrong axis shows up here rather than as a
plausible-looking number in the results.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.scripts.interference_probe import (  # noqa: E402
    change_locality, cosines, group_of, overlap, probe_pair,
)


def test_group_of_does_not_confuse_encoder_v_with_encoder():
    """encoder is a prefix of encoder_v; testing it first would misfile the
    value path as the query path and corrupt the per-group cosines."""
    assert group_of("core.encoder_v") == "enc_y"
    assert group_of("core.encoder") == "enc_x"
    assert group_of("neuron_unit_y.weight") == "enc_y"
    assert group_of("neuron_unit_x.A") == "enc_x"
    assert group_of("core.decoder") == "dec"


def test_cosines_of_identical_and_opposite_gradients():
    g = {"core.encoder": torch.randn(3, 4), "core.decoder": torch.randn(5)}
    same = cosines(g, g)
    assert same["all"] == pytest.approx(1.0, abs=1e-6)
    opp = cosines(g, {k: -v for k, v in g.items()})
    assert opp["all"] == pytest.approx(-1.0, abs=1e-6)
    assert opp["enc_x"] == pytest.approx(-1.0, abs=1e-6)


def test_locality_of_a_change_confined_to_one_unit():
    """Changing one unit of eight must report that 1/8 of units carry the
    change, and a change spread evenly must report nearly all of them."""
    before = torch.ones(1, 2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    one = before.clone()
    one[..., 0] += 1.0
    loc = change_locality(before, one, mask)
    assert loc["frac_units_90"] == pytest.approx(1 / 8)
    spread = before + 0.1
    loc2 = change_locality(before, spread, mask)
    assert loc2["frac_units_90"] > 0.8
    assert loc["rel_change"] > 0


def test_locality_ignores_padded_positions():
    before = torch.ones(1, 1, 2, 4)
    after = before.clone()
    after[0, 0, 1, :] += 10.0           # change only at a padded position
    mask = torch.tensor([[True, False]])
    assert change_locality(before, after, mask)["rel_change"] == 0.0


def test_overlap_is_one_for_identical_codes_and_zero_for_disjoint():
    U = 16
    a = torch.zeros(1, 4, 5, U)
    a[..., :8] = 1.0
    m = torch.ones(4, 5, dtype=torch.bool)
    same = overlap(a, m, a.clone(), m, n_pairs=200, seed=0)
    assert same["cross"] == pytest.approx(1.0, abs=1e-5)
    b = torch.zeros(1, 4, 5, U)
    b[..., 8:] = 1.0                     # disjoint support
    dis = overlap(a, m, b, m, n_pairs=200, seed=0)
    assert dis["cross"] == pytest.approx(0.0, abs=1e-6)
    assert dis["within_A"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["bdh", "dendritic", "deep", "gpt"])
def test_probe_runs_end_to_end(kind):
    from bdh_cl.cl.tasks import build_synthetic_sequence
    from bdh_cl.config import BDHSpec, GPTSpec, TrainSpec
    from bdh_cl.models.bdh_wrapper import BDHWrapper
    from bdh_cl.models.gpt_baseline import GPTBaseline

    base = dict(n_layer=1, n_embd=32, n_head=2, mlp_internal_dim_multiplier=2, dropout=0.0)
    build = {
        "bdh": lambda: BDHWrapper(BDHSpec(**base)),
        "dendritic": lambda: BDHWrapper(BDHSpec(**base, neuron_unit="dendritic",
                                                neuron_unit_kwargs={"n_branches": 4, "soma": "sum"})),
        "deep": lambda: BDHWrapper(BDHSpec(**base, neuron_unit="deep",
                                           neuron_unit_kwargs={"match_branches": 4})),
        "gpt": lambda: GPTBaseline(GPTSpec(n_layer=1, n_embd=32, n_head=2)),
    }[kind]
    ta, tb = build_synthetic_sequence(["copy", "reverse"], n_eval=8, batch_size=4)
    train = TrainSpec(block_size=64, batch_size=4, steps_per_task=3, warmup_steps=1,
                      eval_every=99, log_every=99, seed=0)
    r = probe_pair(build, ta, tb, train, torch.device("cpu"), seed=0,
                   b_steps=(1, 2), grad_batches=1, n_pairs=50)
    assert [x["steps"] for x in r["trajectory"]] == [1, 2]
    for x in r["trajectory"]:
        assert math.isfinite(x["ratio"]) and math.isfinite(x["dA"])
    assert -1.0 - 1e-6 <= r["grad_cos"]["all"] <= 1.0 + 1e-6
    assert 0.0 <= r["locality"]["frac_units_90"] <= 1.0
    assert math.isfinite(r["overlap"]["cross"])
