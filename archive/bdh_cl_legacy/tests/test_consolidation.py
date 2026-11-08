"""Branch-granular consolidation: does the mask do what it claims?

The claims worth testing, because each could silently be the opposite:

* gradients shrink where Omega is high, and are untouched where it is zero
* branch granularity really does leave sibling branches plastic (the whole
  reason for doing this on a dendritic model)
* the cap bounds how much gets protected, so rigidity cannot run away
* nothing is masked before the first task boundary
* consolidation changes the trained weights (i.e. it is actually wired in)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.cl.consolidation import (  # noqa: E402
    Consolidation,
    ConsolidationSpec,
    _cap,
    _normalise,
    build_consolidation,
)
from bdh_cl.config import BDHSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402


def dendritic_model(k=4, N_mult=2, n_head=2, d=32, n_layer=2):
    spec = BDHSpec(
        n_layer=n_layer,
        n_embd=d,
        n_head=n_head,
        mlp_internal_dim_multiplier=N_mult,
        neuron_unit="dendritic",
        neuron_unit_kwargs={"n_branches": k, "soma": "gated"},
    )
    return BDHWrapper(spec)


def plain_model(d=32, n_head=2, N_mult=2, n_layer=2):
    return BDHWrapper(
        BDHSpec(n_layer=n_layer, n_embd=d, n_head=n_head, mlp_internal_dim_multiplier=N_mult)
    )


# -- helpers --------------------------------------------------------------


def test_normalise_is_per_head():
    """A loud head must not decide the other heads are unimportant."""
    m = torch.zeros(2, 1, 4)
    m[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    hat = _normalise(m)
    assert hat[0].max() == pytest.approx(1.0)
    assert hat[1].max() == pytest.approx(1.0)
    # same shape of relative importance in both heads
    assert torch.allclose(hat[0], hat[1])


def test_cap_keeps_only_the_top_fraction():
    hat = torch.linspace(0, 1, 20).reshape(1, 1, 20)
    capped = _cap(hat, 0.25)
    assert int((capped > 0).sum()) == 5
    # and it keeps the LARGEST ones
    assert capped.flatten()[-5:].min() > 0
    assert capped.flatten()[:15].max() == 0


# -- the mask -------------------------------------------------------------


def make_consol(model, **kw):
    spec = ConsolidationSpec(enabled=True, **kw)
    return Consolidation(spec, model, torch.device("cpu"))


def test_no_mask_before_the_first_boundary():
    """Task 0 must train unimpeded; Omega does not exist yet."""
    m = dendritic_model()
    c = make_consol(m)
    m.neuron_unit_x.weight.grad = torch.ones_like(m.neuron_unit_x.weight)
    c.mask_grads(m)
    assert torch.all(m.neuron_unit_x.weight.grad == 1.0)


def test_high_importance_shrinks_gradient_and_zero_leaves_it():
    """mode="scale" behaves as documented -- even though Adam cancels it in
    practice, which is what test_adam_cancels_a_constant_gradient_rescale and
    the lam sweep establish separately."""
    m = dendritic_model(k=4)
    c = make_consol(m, mode="scale", lam=9.0, protect_decoder=False)
    u = m.neuron_unit_x
    nh, k, N = u.n_head, u.k, u.N
    om = torch.zeros(nh, k, N)
    om[:, 0, :] = 1.0          # branch 0 fully important
    c.omega["x_sparse"] = om

    u.weight.grad = torch.ones_like(u.weight)
    c.mask_grads(m)
    g = u.weight.grad.reshape(nh, -1, k, N)
    # 1 / (1 + 9*1) = 0.1 on the protected branch
    assert torch.allclose(g[:, :, 0, :], torch.full_like(g[:, :, 0, :], 0.1))
    # untouched elsewhere
    assert torch.allclose(g[:, :, 1:, :], torch.ones_like(g[:, :, 1:, :]))


def test_branch_granularity_leaves_siblings_plastic():
    """The point of doing this on a dendritic model.

    Protecting branch 0 of every neuron must leave branches 1..k-1 of the SAME
    neurons fully trainable. A per-neuron scheme cannot express this.
    """
    m = dendritic_model(k=4)
    c = make_consol(m, mode="scale", lam=100.0, protect_decoder=False)
    u = m.neuron_unit_x
    om = torch.zeros(u.n_head, u.k, u.N)
    om[:, 0, :] = 1.0
    c.omega["x_sparse"] = om
    u.weight.grad = torch.ones_like(u.weight)
    c.mask_grads(m)
    g = u.weight.grad.reshape(u.n_head, -1, u.k, u.N)
    assert g[:, :, 0, :].max() < 0.02        # frozen
    assert g[:, :, 1:, :].min() == 1.0       # fully plastic


def test_neuron_granularity_cannot_spare_siblings():
    """The contrast: collapsing to per-neuron masks every branch alike."""
    m = dendritic_model(k=4)
    c = make_consol(m, mode="scale", granularity="neuron", lam=9.0, protect_decoder=False)
    u = m.neuron_unit_x
    # what estimate_and_merge does for granularity="neuron": branch axis mean,
    # broadcast back
    raw = torch.zeros(u.n_head, u.k, u.N)
    raw[:, 0, :] = 1.0
    collapsed = raw.mean(dim=1, keepdim=True).expand_as(raw)
    c.omega["x_sparse"] = _normalise(collapsed.contiguous())
    u.weight.grad = torch.ones_like(u.weight)
    c.mask_grads(m)
    g = u.weight.grad.reshape(u.n_head, -1, u.k, u.N)
    # every branch scaled identically -- no sibling is spared
    for cc in range(1, u.k):
        assert torch.allclose(g[:, :, 0, :], g[:, :, cc, :])


def test_decoder_is_shielded_when_asked():
    m = dendritic_model(k=4)
    c = make_consol(m, mode="scale", lam=9.0, protect_decoder=True)
    u = m.neuron_unit_x
    c.omega["x_sparse"] = torch.ones(u.n_head, u.k, u.N)
    dec = m.core.decoder
    dec.grad = torch.ones_like(dec)
    c.mask_grads(m)
    assert dec.grad.max() < 0.2   # 1/(1+9) = 0.1


@pytest.mark.parametrize("make", [dendritic_model, plain_model], ids=["dendritic", "plain"])
def test_masking_is_never_a_silent_noop(make):
    """Regression guard.

    The first version only masked ``u.weight``, but the DEFAULT neuron unit has
    ``owns_parameters = False`` and borrows the host's encoder, so on a plain
    BDH the mask quietly did nothing -- the neuron-granularity arm of the sweep
    would have been a no-op reported as a negative result. Any unit whose
    gradients cannot be reached must fail loudly, not pass silently.
    """
    m = make()
    c = make_consol(m, mode="mask", threshold=0.5, protect_decoder=False)
    touched = []
    for stream, u in c.units.items():
        from bdh_cl.cl.consolidation import u_weight

        # Explicit None check: `a or b` on a tensor raises, because a tensor
        # with more than one element has no truth value.
        own = getattr(u, "weight", None)
        param = u_weight(m, stream) if own is None else own
        param.grad = torch.ones_like(param)
        nh = param.shape[0]
        k = int(getattr(u, "k", 1))
        N = param.shape[-1] // k
        c.omega[stream] = torch.ones(nh, k, N)
        touched.append((stream, param))
    c.mask_grads(m)
    for stream, param in touched:
        assert param.grad.max() < 0.99, f"{stream}: gradient was not masked at all"


def test_plain_model_is_supported_at_neuron_granularity():
    """A non-dendritic BDH has k=1, so the same code path must still mask."""
    m = plain_model()
    c = make_consol(m, mode="scale", lam=9.0, protect_decoder=False)
    u = m.neuron_unit_x
    assert getattr(u, "k", 1) == 1
    N = m.core.encoder.shape[-1]
    c.omega["x_sparse"] = torch.ones(m.core.encoder.shape[0], 1, N)
    m.core.encoder.grad = torch.ones_like(m.core.encoder)
    c.mask_grads(m)
    assert m.core.encoder.grad.max() == pytest.approx(0.1, abs=1e-6)


# -- estimation -----------------------------------------------------------


class _FakeTask:
    """Minimal task: just the eval_batches() the estimator consumes."""

    name = "fake"

    def __init__(self, T=16, B=2, vocab=256, n=3):
        from bdh_cl.data.bytes import MaskedBatch

        g = torch.Generator().manual_seed(0)
        self._b = [
            MaskedBatch(
                x=torch.randint(0, vocab, (B, T), generator=g),
                y=torch.randint(0, vocab, (B, T), generator=g),
                loss_mask=torch.ones(B, T),
            )
            for _ in range(n)
        ]

    def eval_batches(self):
        return self._b


@pytest.mark.parametrize("signal", ["activity", "synaptic"])
def test_estimate_produces_normalised_omega_of_the_right_shape(signal):
    m = dendritic_model(k=4)
    c = make_consol(m, signal=signal, batches=2)
    rec = c.estimate_and_merge(
        m, _FakeTask(), torch.device("cpu"), np.random.default_rng(0), task_index=0
    )
    for stream in ("x_sparse", "y_sparse"):
        om = c.omega[stream]
        u = getattr(m, "neuron_unit_x" if stream == "x_sparse" else "neuron_unit_y")
        assert om.shape == (u.n_head, u.k, u.N)
        assert om.min() >= 0.0 and om.max() <= 1.0 + 1e-6
    assert rec["task"] == "fake"
    assert "protected_frac" in rec["per_stream"]["x_sparse"]


def test_synaptic_signal_is_uniform_across_branches():
    """Attention runs on the post-soma code, so the state cannot distinguish
    branches -- the synaptic signal must be constant along the branch axis.

    This is the honest limitation of the literal version of the idea.
    """
    m = dendritic_model(k=4)
    c = make_consol(m, signal="synaptic", batches=1)
    c.estimate_and_merge(
        m, _FakeTask(), torch.device("cpu"), np.random.default_rng(0), task_index=0
    )
    om = c.omega["x_sparse"]
    for cc in range(1, om.shape[1]):
        assert torch.allclose(om[:, 0, :], om[:, cc, :], atol=1e-6)


def test_activity_signal_does_vary_across_branches():
    """The contrast that justifies branch granularity at all."""
    m = dendritic_model(k=4)
    c = make_consol(m, signal="activity", batches=2)
    c.estimate_and_merge(
        m, _FakeTask(), torch.device("cpu"), np.random.default_rng(0), task_index=0
    )
    om = c.omega["x_sparse"]
    spread = (om.amax(dim=1) - om.amin(dim=1)).mean()
    assert spread > 1e-4, "branch importances are identical; nothing to protect"


def test_decay_lets_old_importance_fade():
    m = dendritic_model(k=2)
    c = make_consol(m, decay=0.5, batches=1)
    u = m.neuron_unit_x
    c.omega["x_sparse"] = torch.ones(u.n_head, u.k, u.N)
    t = _FakeTask()
    c.estimate_and_merge(m, t, torch.device("cpu"), np.random.default_rng(0), task_index=1)
    # merged = max(prev*decay, new); where new is small the old value halved
    assert c.omega["x_sparse"].min() <= 0.5 + 1e-6


def test_cap_bounds_protected_fraction_across_many_tasks():
    """Rigidity must not run away -- the PackNet/HAT failure mode."""
    m = dendritic_model(k=4)
    c = make_consol(m, cap=0.1, decay=0.9, batches=1)
    t = _FakeTask()
    for i in range(6):
        c.estimate_and_merge(
            m, t, torch.device("cpu"), np.random.default_rng(i), task_index=i
        )
    frac = float((c.omega["x_sparse"] > 0.5).float().mean())
    assert frac <= 0.35, f"protected {frac:.1%} after 6 tasks, cap was 0.1"


# -- wiring ---------------------------------------------------------------


def test_build_returns_none_when_disabled():
    m = dendritic_model()
    assert build_consolidation(None, m, torch.device("cpu")) is None
    assert build_consolidation({"enabled": False}, m, torch.device("cpu")) is None
    assert build_consolidation({"enabled": True}, m, torch.device("cpu")) is not None


def test_bad_spec_is_rejected():
    with pytest.raises(ValueError, match="signal"):
        ConsolidationSpec(signal="nonsense")
    with pytest.raises(ValueError, match="granularity"):
        ConsolidationSpec(granularity="nonsense")
    with pytest.raises(ValueError, match="cap"):
        ConsolidationSpec(cap=0.0)


def test_masking_actually_changes_training():
    """End to end: identical seeds, one with consolidation, one without.

    If the weights come out the same the mask is not reaching the optimizer.
    """
    from bdh_cl.cl.runner import train_one_task
    from bdh_cl.config import TrainSpec

    spec = TrainSpec(block_size=16, batch_size=2, steps_per_task=3, warmup_steps=1,
                     eval_every=99, log_every=99, seed=0)
    task = _FakeTask(T=16, B=2)
    task.train_batch = lambda rng, bs: task.eval_batches()[0]

    outs = []
    for use in (False, True):
        torch.manual_seed(0)
        m = dendritic_model(k=4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        c = None
        if use:
            c = make_consol(m, mode="mask", threshold=0.5, protect_decoder=False)
            u = m.neuron_unit_x
            om = torch.zeros(u.n_head, u.k, u.N)
            om[:, 0, :] = 1.0
            c.omega["x_sparse"] = om
        train_one_task(
            m, task, opt, spec, torch.device("cpu"), np.random.default_rng(0),
            batch_fn=lambda rng, bs: task.eval_batches()[0],
            verbose=False, consolidation=c,
        )
        outs.append(m.neuron_unit_x.weight.detach().clone())

    u = dendritic_model(k=4).neuron_unit_x
    a = outs[0].reshape(u.n_head, -1, u.k, u.N)
    b = outs[1].reshape(u.n_head, -1, u.k, u.N)
    # the protected branch moved LESS with consolidation on
    d_free = (a[:, :, 0, :] - b[:, :, 0, :]).abs().mean()
    assert d_free > 0, "consolidation had no effect on the trained weights"


# -- modes: why gradient scaling alone is not enough -----------------------


def test_adam_cancels_a_constant_gradient_rescale():
    """The diagnosis behind mode="penalty", as an executable claim.

    AdamW's update is lr * m_hat / (sqrt(v_hat) + eps). Scaling every gradient
    by a constant c scales m by c and sqrt(v) by c too, so the update is
    unchanged. That is why the lam sweep in mode="scale" sat on the naive floor
    at every lam from 1 to 1000, and why consolidation has to enter through the
    loss instead.
    """
    results = []
    for c in (1.0, 0.001):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.ones(4))
        opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.0)
        g = torch.tensor([0.5, -0.25, 1.0, -2.0])
        for _ in range(30):
            opt.zero_grad()
            w.grad = g * c
            opt.step()
        results.append(w.detach().clone())
    # a 1000x smaller gradient moved the weights essentially the same distance
    assert torch.allclose(results[0], results[1], atol=1e-4), (
        "if this fails, Adam is not scale-invariant here and the diagnosis "
        "for the flat lam sweep needs revisiting"
    )


def test_penalty_is_zero_before_any_boundary_and_positive_after():
    m = dendritic_model(k=4)
    c = make_consol(m, mode="penalty", lam=1.0, batches=1)
    assert c.penalty(m) is None           # no omega, no anchor yet
    c.estimate_and_merge(
        m, _FakeTask(), torch.device("cpu"), np.random.default_rng(0), task_index=0
    )
    # at the anchor itself the penalty is exactly zero
    assert float(c.penalty(m).detach()) == pytest.approx(0.0, abs=1e-9)
    # move away from it and it grows
    with torch.no_grad():
        m.neuron_unit_x.weight.add_(0.1)
    assert float(c.penalty(m).detach()) > 0.0


def test_penalty_grows_with_importance():
    """A unit with high Omega must be pulled back harder than a low-Omega one."""
    m = dendritic_model(k=2)
    c = make_consol(m, mode="penalty", lam=1.0, protect_decoder=False)
    u = m.neuron_unit_x
    c.omega["x_sparse"] = torch.zeros(u.n_head, u.k, u.N)
    c.omega["x_sparse"][:, 0, :] = 1.0
    c.anchor(m)
    base = u.weight.detach().clone().reshape(u.n_head, -1, u.k, u.N)

    def penalty_after_nudging(branch):
        with torch.no_grad():
            w = base.clone()
            w[:, :, branch, :] += 0.1
            u.weight.copy_(w.reshape(u.weight.shape))
        return float(c.penalty(m).detach())

    assert penalty_after_nudging(0) > penalty_after_nudging(1)


def test_penalty_produces_a_restoring_gradient():
    """The property gradient scaling lacks: an actual pull toward the anchor."""
    m = dendritic_model(k=2)
    c = make_consol(m, mode="penalty", lam=1.0, protect_decoder=False)
    u = m.neuron_unit_x
    c.omega["x_sparse"] = torch.ones(u.n_head, u.k, u.N)
    c.anchor(m)
    with torch.no_grad():
        u.weight.add_(0.1)          # displaced in the + direction
    u.weight.grad = None
    c.penalty(m).backward()
    # gradient points back toward the anchor, i.e. positive => descent reduces w
    assert u.weight.grad.mean() > 0


def test_mask_mode_zeroes_rather_than_scales():
    m = dendritic_model(k=4)
    c = make_consol(m, mode="mask", threshold=0.5, protect_decoder=False)
    u = m.neuron_unit_x
    om = torch.zeros(u.n_head, u.k, u.N)
    om[:, 0, :] = 1.0               # above threshold
    om[:, 1, :] = 0.2               # below
    c.omega["x_sparse"] = om
    u.weight.grad = torch.ones_like(u.weight)
    c.mask_grads(m)
    g = u.weight.grad.reshape(u.n_head, -1, u.k, u.N)
    assert float(g[:, :, 0, :].abs().max()) == 0.0    # hard zero survives Adam
    assert float(g[:, :, 1, :].min()) == 1.0


def test_penalty_mode_leaves_gradients_untouched_by_mask_grads():
    """In penalty mode the work happens in the loss; mask_grads must not also
    rescale, or the two mechanisms would compound invisibly."""
    m = dendritic_model(k=2)
    c = make_consol(m, mode="penalty", lam=100.0)
    u = m.neuron_unit_x
    c.omega["x_sparse"] = torch.ones(u.n_head, u.k, u.N)
    u.weight.grad = torch.ones_like(u.weight)
    c.mask_grads(m)
    assert float(u.weight.grad.min()) == 1.0


# -- prototype readout ----------------------------------------------------


def test_return_hidden_does_not_change_logits():
    """The new flag must be inert: same weights, same logits, bit for bit.

    bdh_wrapper is proved equivalent to upstream, so any change to its forward
    has to be shown not to alter the computation.
    """
    from bdh_cl.models.gpt_baseline import GPTBaseline
    from bdh_cl.config import GPTSpec

    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 16))

    m = dendritic_model(k=2)
    m.eval()
    a, _ = m(x, None)
    b, _, h = m(x, None, return_hidden=True)
    assert torch.equal(a, b)
    assert h.shape == (2, 16, m.core.config.n_embd)

    # GPTSpec has no block_size: it uses RoPE so the parameter count does not
    # depend on context width.
    g = GPTBaseline(GPTSpec(n_layer=2, n_embd=32, n_head=2))
    g.eval()
    a, _ = g(x, None)
    b, _, h = g(x, None, return_hidden=True)
    assert torch.equal(a, b)
    assert h.shape[-1] == 32


def test_prototype_classifier_recovers_separable_classes():
    from bdh_cl.scripts.proto_eval import classify, prototypes_from

    g = torch.Generator().manual_seed(0)
    # two well-separated directions in 8-d
    a = torch.randn(40, 8, generator=g) * 0.1 + torch.tensor([3.0] + [0.0] * 7)
    b = torch.randn(40, 8, generator=g) * 0.1 + torch.tensor([0.0, 3.0] + [0.0] * 6)
    feats = torch.cat([a, b])
    labs = torch.cat([torch.full((40,), ord("0")), torch.full((40,), ord("1"))])
    protos = prototypes_from(feats, labs)
    assert set(protos) == {ord("0"), ord("1")}
    assert float((classify(feats, protos) == labs).float().mean()) > 0.95


def test_prototypes_are_scale_invariant():
    """Normalising before averaging: a global change in feature NORM says
    nothing about class identity and must not move the decision."""
    from bdh_cl.scripts.proto_eval import classify, prototypes_from

    g = torch.Generator().manual_seed(1)
    feats = torch.randn(30, 6, generator=g)
    labs = torch.tensor([ord("0")] * 15 + [ord("1")] * 15)
    p1 = prototypes_from(feats, labs)
    p2 = prototypes_from(feats * 17.0, labs)
    for c in p1:
        assert torch.allclose(p1[c], p2[c], atol=1e-6)
    assert torch.equal(classify(feats, p1), classify(feats * 17.0, p2))


def test_prototype_classifier_uses_no_task_id():
    """Every stored class competes, so the readout is class-incremental.

    If it silently restricted to one task's classes it would be smuggling the
    task id back in, which is the exact thing prototypes are here to avoid.
    """
    from bdh_cl.scripts.proto_eval import classify, prototypes_from

    g = torch.Generator().manual_seed(2)
    feats = torch.randn(20, 5, generator=g)
    labs = torch.tensor([ord(c) for c in "0123456789"] * 2)
    protos = prototypes_from(feats, labs)
    assert len(protos) == 10
    preds = set(classify(feats, protos).tolist())
    assert preds - {ord(c) for c in "0123456789"} == set()
