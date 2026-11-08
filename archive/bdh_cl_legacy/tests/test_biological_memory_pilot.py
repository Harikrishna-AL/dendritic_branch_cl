"""Checkpoint correctness for the published-circuit assay (GPLv3+)."""
import copy
import numpy as np
import pytest

from bdh_cl.scripts.biological_memory_pilot import model, deliver, compact_clone, probe


@pytest.mark.parametrize("arm", ["full", "no_ltm", "no_plasticity", "unpaired"])
def test_checkpoint_continuation_and_probe_isolation(arm):
    m = model(2021, arm)
    deliver(m, [(0, True, True), (1, False, True)] * 6)
    c = compact_clone(m)
    full = copy.deepcopy(m)
    schedule = [(None, False, True), (1, True, True), (0, False, True)]
    deliver(c, schedule)
    deliver(full, schedule)
    np.testing.assert_array_equal(c._v[c._t], full._v[full._t])
    np.testing.assert_array_equal(c.w_k2m[c._t], full.w_k2m[full._t])
    before = copy.deepcopy(m)
    for reset in ["none", "activity", "weights"]:
        probe(m, reset)
    np.testing.assert_array_equal(m._v, before._v)
    np.testing.assert_array_equal(m.w_k2m, before.w_k2m)
    a, b = m.rng.get_state(), before.rng.get_state()
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
