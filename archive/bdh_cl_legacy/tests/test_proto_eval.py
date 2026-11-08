"""proto_eval.load tolerates exactly one kind of old checkpoint: a sparse
dendritic unit saved before the ``coactivity`` buffer existed (exp18)."""

import pytest
import torch

from bdh_cl.models.neuron_units import SparseDendriticNeuronUnit
from bdh_cl.scripts.proto_eval import load


def _unit():
    return SparseDendriticNeuronUnit(2, 16, 8, n_branches=2, contacts=4, rewire_every=5,
                                     rewire_mode="coactivity")


def test_missing_coactivity_buffer_loads(tmp_path):
    src = _unit()
    sd = {k: v for k, v in src.state_dict().items() if not k.endswith("coactivity")}
    torch.save({"model": sd}, tmp_path / "old.pt")
    m = load({"u": _unit}, "u", tmp_path / "old.pt", "cpu")
    assert torch.equal(m.weight, src.weight)
    assert not m.training


def test_other_missing_key_still_fails(tmp_path):
    sd = {k: v for k, v in _unit().state_dict().items() if k != "weight"}
    torch.save({"model": sd}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="missing"):
        load({"u": _unit}, "u", tmp_path / "bad.pt", "cpu")
