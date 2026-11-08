"""Sparse dendritic unit with structural rewiring."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.config import BDHSpec, TrainSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.neuron_units import (  # noqa: E402
    DendriticNeuronUnit,
    SparseDendriticNeuronUnit,
    build_neuron_unit,
)

NH, D, N, K = 2, 16, 8, 4


def unit(**kw):
    torch.manual_seed(0)
    base = dict(n_head=NH, d=D, N=N, n_branches=K, soma="max", contacts=4)
    base.update(kw)
    return SparseDendriticNeuronUnit(**base)


def per_branch_counts(u):
    return u.mask.sum(dim=1)  # (nh, k*N)


def test_every_branch_has_exactly_the_contact_budget():
    u = unit()
    assert torch.all(per_branch_counts(u) == 4)
    assert u.active_synapses() == NH * K * N * 4


def test_dormant_synapses_do_not_affect_the_output():
    u = unit()
    x = torch.randn(2, 1, 5, D)
    before = u(x)
    with torch.no_grad():
        u.weight.add_(torch.where(u.mask, 0.0, 100.0))
    assert torch.equal(u(x), before)


def test_dormant_synapses_receive_no_gradient():
    u = unit()
    u(torch.randn(2, 1, 5, D)).sum().backward()
    assert torch.all(u.weight.grad[~u.mask] == 0)
    assert u.weight.grad[u.mask].abs().sum() > 0


def test_full_contacts_single_branch_equals_dense_unit():
    torch.manual_seed(0)
    dense = DendriticNeuronUnit(NH, D, N, n_branches=1, soma="max")
    torch.manual_seed(0)
    sparse = SparseDendriticNeuronUnit(NH, D, N, n_branches=1, soma="max", contacts=D)
    x = torch.randn(2, 1, 5, D)
    assert torch.equal(dense(x), sparse(x))


def test_masks_differ_across_run_seeds_and_paths():
    torch.manual_seed(0)
    a = SparseDendriticNeuronUnit(NH, D, N, n_branches=K, contacts=4)
    torch.manual_seed(1)
    b = SparseDendriticNeuronUnit(NH, D, N, n_branches=K, contacts=4)
    torch.manual_seed(0)
    c = SparseDendriticNeuronUnit(NH, D, N, n_branches=K, contacts=4, mask_seed=1)
    assert not torch.equal(a.mask, b.mask)
    assert not torch.equal(a.mask, c.mask)


def test_rewire_preserves_budget_prunes_weakest_and_regrows_dormant_at_zero():
    u = unit(rewire_every=1, rewire_frac=0.25)  # 1 synapse per branch
    with torch.no_grad():
        u.weight.copy_(torch.randn_like(u.weight) * u.mask)
    old_mask = u.mask.clone()
    old_w = u.weight.detach().clone()
    moved = u.rewire()
    assert moved == NH * K * N
    assert torch.all(per_branch_counts(u) == 4)
    pruned = old_mask & ~u.mask
    grown = u.mask & ~old_mask
    assert torch.all(pruned.sum(1) == 1) and torch.all(grown.sum(1) == 1)
    # The pruned synapse was the weakest active one of its branch.
    weakest = torch.where(old_mask, old_w.abs(), torch.inf).amin(dim=1)
    assert torch.allclose((old_w.abs() * pruned).sum(dim=1), weakest)
    assert torch.all(u.weight[grown] == 0)
    # Untouched synapses keep their weights.
    kept = old_mask & u.mask
    assert torch.equal(u.weight[kept], old_w[kept])


def test_rewire_zeroes_adam_state_of_touched_entries():
    u = unit(rewire_every=1, rewire_frac=0.25)
    opt = torch.optim.AdamW(u.parameters(), lr=1e-2)
    u(torch.randn(2, 1, 5, D)).sum().backward()
    opt.step()
    old_mask = u.mask.clone()
    u.rewire(opt)
    touched = old_mask ^ u.mask
    st = opt.state[u.weight]
    assert torch.all(st["exp_avg"][touched] == 0)
    assert torch.all(st["exp_avg_sq"][touched] == 0)


def test_post_step_rewires_on_schedule_only():
    u = unit(rewire_every=3)
    m0 = u.mask.clone()
    u.post_step(); u.post_step()
    assert torch.equal(u.mask, m0)
    u.post_step()
    assert not torch.equal(u.mask, m0)


def test_no_rewiring_by_default():
    u = unit()
    m0 = u.mask.clone()
    for _ in range(10):
        u.post_step()
    assert torch.equal(u.mask, m0) and u.rewired_total == 0


def test_rewiring_requires_dormant_synapses():
    with pytest.raises(ValueError):
        unit(contacts=D, rewire_every=5)


def test_runner_calls_post_step_hook():
    """train_one_task must drive rewiring; otherwise a rewire arm silently
    equals the fixed-mask arm."""
    from bdh_cl.cl.runner import make_optimizer, train_one_task
    from bdh_cl.cl.tasks import build_synthetic_sequence

    spec = BDHSpec(n_layer=1, n_embd=D, n_head=NH, mlp_internal_dim_multiplier=N * NH // D,
                   neuron_unit="sparse_dendritic",
                   neuron_unit_kwargs={"n_branches": K, "contacts": 4, "rewire_every": 2})
    torch.manual_seed(0)
    model = BDHWrapper(spec)
    task = build_synthetic_sequence(["copy"], n_eval=4, eval_seed=0, batch_size=4)[0]
    train = TrainSpec(block_size=64, batch_size=4, steps_per_task=4, warmup_steps=1,
                      eval_every=100, log_every=100)
    opt = make_optimizer(model, train)
    train_one_task(model, task, opt, train, torch.device("cpu"),
                   np.random.default_rng(0), verbose=False)
    assert model.neuron_unit_x.rewired_total > 0
    assert model.neuron_unit_y.rewired_total > 0
    for u in (model.neuron_unit_x, model.neuron_unit_y):
        assert torch.all(u.mask.sum(1) == 4)


def test_selectable_by_name():
    u = build_neuron_unit("sparse_dendritic", n_head=NH, d=D, N=N, contacts=4)
    assert isinstance(u, SparseDendriticNeuronUnit)


def test_unknown_rewire_mode_rejected():
    with pytest.raises(ValueError):
        unit(rewire_mode="bogus")


def test_coactivity_defaults_to_zero_and_only_tracked_in_that_mode():
    u = unit(rewire_every=1, rewire_frac=0.25)  # rewire_mode="random"
    assert torch.all(u.coactivity == 0)
    u.train()
    u(torch.randn(2, 1, 5, D))
    assert torch.all(u.coactivity == 0)  # not accumulated outside coactivity mode


def test_coactivity_accumulates_during_training_forward_only():
    u = unit(rewire_mode="coactivity")
    u.train()
    u(torch.randn(2, 1, 5, D))
    assert torch.any(u.coactivity != 0)
    before = u.coactivity.clone()
    u.eval()
    u(torch.randn(2, 1, 5, D))
    assert torch.equal(u.coactivity, before)  # eval-mode forward does not update it


def test_coactivity_rewire_regrows_highest_trace_dormant_synapse():
    u = unit(rewire_every=1, rewire_frac=0.25, rewire_mode="coactivity")
    with torch.no_grad():
        u.weight.copy_(torch.randn_like(u.weight) * u.mask)
        # Make one dormant synapse per branch column clearly the most
        # attractive regrow target.
        u.coactivity.copy_(torch.rand_like(u.coactivity) * ~u.mask)
        best = torch.where(~u.mask, u.coactivity, torch.full_like(u.coactivity, -1.0))
        want = best.argmax(dim=1)
    old_mask = u.mask.clone()
    moved = u.rewire()
    assert moved == NH * K * N
    grown = u.mask & ~old_mask
    got = grown.float().argmax(dim=1)
    assert torch.equal(got, want)


def test_coactivity_mode_is_selectable_by_name():
    u = build_neuron_unit(
        "sparse_dendritic", n_head=NH, d=D, N=N, contacts=4, rewire_mode="coactivity"
    )
    assert isinstance(u, SparseDendriticNeuronUnit)
    assert u.rewire_mode == "coactivity"
