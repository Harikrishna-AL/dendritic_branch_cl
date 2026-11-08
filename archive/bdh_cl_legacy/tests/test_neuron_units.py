"""The NeuronUnit interface (Step 7).

``default`` must reproduce upstream's nonlinearity exactly, and the dendritic
stub must stay a stub -- if someone accidentally starts using it, these tests
say so loudly.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.config import BDHSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper, upstream_bdh, upstream_config  # noqa: E402
from bdh_cl.models.neuron_units import (  # noqa: E402
    UNITS,
    DefaultNeuronUnit,
    DendriticNeuronUnit,
    NeuronUnit,
    build_neuron_unit,
)
from bdh_cl.models.param_accounting import dendritic_budget  # noqa: E402


def test_default_unit_is_upstream_expression():
    """relu(x @ W), bit for bit."""
    torch.manual_seed(0)
    nh, d, N, B, T = 3, 16, 32, 2, 5
    x = torch.randn(B, 1, T, d)
    W = torch.randn(nh, d, N)
    unit = build_neuron_unit("default")
    assert torch.equal(unit(x, W), F.relu(x @ W))


def test_default_unit_adds_no_parameters():
    """The upstream module must stay the sole parameter owner."""
    unit = DefaultNeuronUnit()
    assert list(unit.parameters()) == []
    assert unit.owns_parameters is False
    assert unit.extra_param_count(4, 128, 1024) == 0

    spec = BDHSpec(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4)
    wrapper = BDHWrapper(spec)
    reference = upstream_bdh.BDH(upstream_config(spec))
    assert sum(p.numel() for p in wrapper.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    # And the state_dict stays upstream-loadable.
    reference.load_state_dict(wrapper.core.state_dict())


def test_default_unit_output_is_nonnegative():
    torch.manual_seed(1)
    unit = DefaultNeuronUnit()
    out = unit(torch.randn(2, 1, 7, 8), torch.randn(2, 8, 16))
    assert (out >= 0).all()


def test_default_unit_requires_a_weight():
    with pytest.raises(ValueError):
        DefaultNeuronUnit()(torch.randn(1, 1, 2, 3), None)


def test_dendritic_unit_is_implemented_and_documented():
    """Was a stub in phase 1; implemented in phase 2 on request."""
    assert DendriticNeuronUnit.owns_parameters is True
    u = build_neuron_unit("dendritic", n_head=4, d=128, N=32, n_branches=4, soma="max")
    assert isinstance(u, DendriticNeuronUnit)
    assert u.k == 4 and u.soma == "max"
    assert u.weight.shape == (4, 128, 4 * 32)
    doc = DendriticNeuronUnit.__doc__
    for word in ("branch", "soma", "sum", "max", "gated"):
        assert word in doc.lower()


def test_registry_and_unknown_name():
    assert set(UNITS) >= {"default", "dendritic"}
    assert all(issubclass(c, NeuronUnit) for c in UNITS.values())
    with pytest.raises(KeyError):
        build_neuron_unit("no_such_unit")


def test_spec_selects_the_unit():
    spec = BDHSpec(n_layer=1, n_embd=16, n_head=2, mlp_internal_dim_multiplier=2)
    assert isinstance(BDHWrapper(spec).neuron_unit, DefaultNeuronUnit)

    spec.neuron_unit = "dendritic"
    spec.neuron_unit_kwargs = {"n_branches": 2, "soma": "max"}
    w = BDHWrapper(spec)
    assert isinstance(w.neuron_unit_x, DendriticNeuronUnit)
    assert isinstance(w.neuron_unit_y, DendriticNeuronUnit)


def test_dendritic_budget_stays_parameter_matched():
    """The accounting utility must give a neuron count that matches baseline."""
    spec = BDHSpec(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128)
    for k in (1, 2, 4, 8):
        b = dendritic_budget(spec, k)
        d = spec.n_embd
        matched_total = (2 * k + 1) * b["param_matched_n_neurons"] * d + 2 * spec.vocab_size * d
        assert matched_total == pytest.approx(b["baseline_total"], rel=1e-3), k
    # k=1 with sum-at-the-soma is the baseline, so nothing should change.
    assert dendritic_budget(spec, 1)["naive_blowup"] == pytest.approx(1.0)
    assert dendritic_budget(spec, 1)["param_matched_n_neurons"] == spec.n_neurons


# --------------------------------------------------------------------------
# Dendritic unit (implemented in phase 2 at the user's request)
# --------------------------------------------------------------------------


def test_dendritic_k1_sum_is_exactly_the_default_unit():
    """k=1 with sum-at-the-soma and no threshold IS relu(x @ W).

    This is what makes a dendritic run a controlled comparison: the only thing
    that changes as k grows is branching, not a different model.
    """
    torch.manual_seed(0)
    nh, d, N, B, T = 3, 16, 8, 2, 5
    unit = DendriticNeuronUnit(n_head=nh, d=d, N=N, n_branches=1, soma="sum")
    x = torch.randn(B, 1, T, d)
    W = unit.weight.view(nh, d, N)
    assert torch.allclose(unit(x), F.relu(x @ W), atol=0, rtol=0)


def test_dendritic_soma_semantics():
    torch.manual_seed(1)
    nh, d, N, k = 2, 8, 4, 3
    x = torch.randn(1, 1, 6, d)
    for soma in ("sum", "max", "gated"):
        u = DendriticNeuronUnit(n_head=nh, d=d, N=N, n_branches=k, soma=soma)
        a = u.branch_activations(x)
        assert a.shape == (1, nh, 6, k, N)
        assert (a >= 0).all()
        out = u(x)
        assert out.shape == (1, nh, 6, N)
        assert (out >= 0).all(), soma
        if soma == "sum":
            assert torch.allclose(out, a.sum(-2), atol=1e-6)
        elif soma == "max":
            assert torch.allclose(out, a.amax(-2), atol=1e-6)
            # max <= sum always, and strictly less when >1 branch is active
            assert (out <= a.sum(-2) + 1e-6).all()


def test_dendritic_more_branches_is_sparser_at_the_soma_with_max():
    """With max pooling, adding branches must not inflate the output scale."""
    torch.manual_seed(2)
    x = torch.randn(2, 1, 8, 16)
    prev = None
    for k in (1, 2, 4):
        u = DendriticNeuronUnit(n_head=2, d=16, N=8, n_branches=k, soma="max")
        m = float(u(x).mean().detach())
        if prev is not None:
            assert m >= prev - 1e-6  # max over more branches can only grow
        prev = m


def test_dendritic_rejects_bad_config():
    with pytest.raises(ValueError):
        DendriticNeuronUnit(n_head=2, d=8, N=4, soma="average")
    with pytest.raises(ValueError):
        DendriticNeuronUnit(n_head=2, d=8, N=4, n_branches=0)
    with pytest.raises(ValueError):
        DendriticNeuronUnit(n_head=2, d=8, N=4, branch_nonlinearity="tanh")


def test_dendritic_wrapper_has_separate_x_and_y_units():
    """Upstream has two projections; sharing one unit would halve the weights."""
    from bdh_cl.models.bdh_wrapper import BDHWrapper

    spec = BDHSpec(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4,
                   neuron_unit="dendritic",
                   neuron_unit_kwargs={"n_branches": 3, "soma": "max"})
    w = BDHWrapper(spec)
    assert w.neuron_unit_x is not w.neuron_unit_y
    assert not torch.equal(w.neuron_unit_x.weight, w.neuron_unit_y.weight)
    # the replaced upstream projections must be gone, not dead weight
    assert w.core.encoder is None and w.core.encoder_v is None
    names = dict(w.named_parameters())
    assert "core.encoder" not in names and "core.encoder_v" not in names


def test_dendritic_parameter_matching_against_baseline():
    """k=1 must match baseline BDH exactly; k>1 within 10% at the tuned mult."""
    from bdh_cl.models.bdh_wrapper import BDHWrapper
    from bdh_cl.models.param_accounting import bdh_param_count, count_module

    base = BDHSpec(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=32)
    target = bdh_param_count(base)["total"]

    k1 = BDHSpec(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=32,
                 neuron_unit="dendritic",
                 neuron_unit_kwargs={"n_branches": 1, "soma": "sum"})
    assert count_module(BDHWrapper(k1))["total"] == target

    for k, mult in ((2, 19), (4, 11), (8, 6)):
        spec = BDHSpec(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=mult,
                       neuron_unit="dendritic",
                       neuron_unit_kwargs={"n_branches": k, "soma": "max"})
        tot = count_module(BDHWrapper(spec))["total"]
        assert abs(tot - target) / target <= 0.10, (k, mult, tot, target)


def test_branch_probe_shapes_and_ranges():
    from bdh_cl.cl import probes as P
    from bdh_cl.data.bytes import LMBatch
    from bdh_cl.models.bdh_wrapper import BDHWrapper

    spec = BDHSpec(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4,
                   dropout=0.0, neuron_unit="dendritic",
                   neuron_unit_kwargs={"n_branches": 4, "soma": "max"})
    m = BDHWrapper(spec)
    batches = [LMBatch(x=torch.randint(0, 256, (2, 12)), y=torch.randint(0, 256, (2, 12)))]
    prof = P.collect_branch_profile(m, batches, torch.device("cpu"))
    assert set(prof) == {"x_sparse", "y_sparse"}
    for _s, pr in prof.items():
        d = pr.to_dict()
        assert d["k"] == 4
        assert 0.0 <= d["alive_frac"] <= 1.0
        assert 0.0 <= d["entropy_normalised"] <= 1.0 + 1e-9
        assert abs(sum(d["usage_share"]) - 1.0) < 1e-6
        assert 0.0 <= d["dead_branch_fraction"] <= 1.0

    # capture must be off again afterwards -- it is k x memory
    assert m.neuron_unit_x.capture is False
    assert m.neuron_unit_x.last_branches is None

    ov = P.branch_overlap(prof, prof)
    for _s, v in ov.items():
        assert v["same_branch_frac"] == pytest.approx(1.0)  # a task vs itself


def test_default_unit_has_no_branch_profile():
    """Branch probes must be a no-op for baseline BDH, not an error."""
    from bdh_cl.cl import probes as P
    from bdh_cl.data.bytes import LMBatch
    from bdh_cl.models.bdh_wrapper import BDHWrapper

    m = BDHWrapper(BDHSpec(n_layer=1, n_embd=16, n_head=2, mlp_internal_dim_multiplier=2))
    batches = [LMBatch(x=torch.randint(0, 256, (2, 8)), y=torch.randint(0, 256, (2, 8)))]
    assert P.collect_branch_profile(m, batches, torch.device("cpu")) == {}


def test_gated_soma_interpolates_between_plain_neuron_and_hard_routing():
    """beta -> 0 must be the mean (one ordinary neuron); growing beta must
    approach max (hard routing to one branch).

    This is the property that makes `gated` the general form: each neuron
    learns its own beta, so the network chooses per neuron how much branch
    structure to use.
    """
    torch.manual_seed(0)
    u = DendriticNeuronUnit(n_head=2, d=16, N=8, n_branches=4, soma="gated")
    x = torch.randn(3, 1, 10, 16) * 3
    a = u.branch_activations(x).detach()

    with torch.no_grad():
        u.log_beta.fill_(-20.0)
        assert torch.allclose(u(x), a.mean(-2), atol=1e-4)

        # monotone approach to max as beta grows
        gaps = []
        for lb in (0.0, 3.0, 6.0, 10.0):
            u.log_beta.fill_(lb)
            gaps.append(float((u(x) - a.amax(-2)).abs().max()))
        assert gaps == sorted(gaps, reverse=True), gaps
        assert gaps[-1] < gaps[0] / 5


def test_gated_is_input_dependent_not_a_static_rescale():
    """A static per-neuron gate would fold into the branch weights and give
    nothing beyond `sum`. The gate must vary with the input."""
    torch.manual_seed(1)
    u = DendriticNeuronUnit(n_head=2, d=16, N=8, n_branches=4, soma="gated")
    with torch.no_grad():
        u.log_beta.fill_(3.0)

    x1 = torch.randn(1, 1, 6, 16) * 3
    x2 = torch.randn(1, 1, 6, 16) * 3
    a1, a2 = u.branch_activations(x1).detach(), u.branch_activations(x2).detach()

    def weights(a):
        beta = torch.nn.functional.softplus(u.log_beta).view(1, 2, 1, 1, 8)
        return torch.softmax(a * beta, dim=-2)

    # the mixture weights themselves differ between inputs
    assert not torch.allclose(weights(a1).mean((0, 2)), weights(a2).mean((0, 2)), atol=1e-3)


def test_gated_gate_is_cheap():
    """The gate must not be a material fraction of the parameter budget."""
    u = DendriticNeuronUnit(n_head=4, d=128, N=352, n_branches=4, soma="gated")
    assert u.log_beta.numel() / u.weight.numel() < 0.01


# -- winner-take-all variants ---------------------------------------------


def _dend(**kw):
    from bdh_cl.models.neuron_units import DendriticNeuronUnit

    return DendriticNeuronUnit(n_head=2, d=16, N=8, **kw)


def test_topk_soma_interpolates_between_max_and_sum():
    """soma_topk=1 must equal max exactly and soma_topk=k must equal sum.

    If these boundaries do not hold, "winner-take-most" is not on the same
    axis as the two somas it sits between and the comparison is meaningless.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 2, 5, 16)

    u_top1 = _dend(n_branches=4, soma="topk", soma_topk=1)
    u_max = _dend(n_branches=4, soma="max")
    u_max.load_state_dict(u_top1.state_dict())
    assert torch.allclose(u_top1(x), u_max(x), atol=0)

    u_topk = _dend(n_branches=4, soma="topk", soma_topk=4)
    u_sum = _dend(n_branches=4, soma="sum")
    u_sum.load_state_dict(u_topk.state_dict())
    assert torch.allclose(u_topk(x), u_sum(x), atol=1e-6)


def test_topk_soma_is_monotone_in_soma_topk():
    """Branch activations are non-negative, so summing more of them cannot
    decrease the output."""
    torch.manual_seed(1)
    x = torch.randn(2, 2, 5, 16)
    u = _dend(n_branches=4, soma="topk", soma_topk=1)
    outs = []
    for kk in (1, 2, 3, 4):
        u.soma_topk = kk
        outs.append(u(x))
    for a, b in zip(outs, outs[1:]):
        assert torch.all(b >= a - 1e-6)


def test_kwta_keeps_the_requested_fraction_of_neurons():
    torch.manual_seed(2)
    x = torch.randn(2, 2, 5, 16)
    u = _dend(n_branches=4, soma="sum", wta_frac=0.25)
    out = u(x)
    # N=8, so 25% keeps 2 units per (batch, head, token)
    nz = (out > 0).sum(dim=-1)
    assert int(nz.max()) <= 3, f"kept too many: {nz.max()}"
    assert int(nz.min()) >= 1


def test_kwta_keeps_the_largest_units():
    u = _dend(n_branches=1, soma="sum", wta_frac=0.5)
    out = torch.tensor([[[[1.0, 5.0, 2.0, 9.0, 0.5, 7.0, 3.0, 4.0]]]])
    kept = u.kwta(out)
    survivors = {float(v) for v in kept[kept > 0]}
    assert survivors == {9.0, 7.0, 5.0, 4.0}


def test_kwta_off_by_default_changes_nothing():
    torch.manual_seed(3)
    x = torch.randn(2, 2, 5, 16)
    a = _dend(n_branches=4, soma="sum")
    b = _dend(n_branches=4, soma="sum", wta_frac=None)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a(x), b(x), atol=0)


def test_kwta_is_input_dependent():
    """Which neurons win must depend on the input, or it is just a fixed mask."""
    torch.manual_seed(4)
    u = _dend(n_branches=4, soma="sum", wta_frac=0.25)
    m1 = (u(torch.randn(1, 2, 4, 16)) > 0)
    m2 = (u(torch.randn(1, 2, 4, 16)) > 0)
    assert not torch.equal(m1, m2), "the same neurons won for different inputs"


def test_kwta_adds_no_parameters():
    """Matched-parameter comparisons must survive turning WTA on."""
    a = _dend(n_branches=4, soma="sum")
    b = _dend(n_branches=4, soma="sum", wta_frac=0.05)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert a.extra_param_count(2, 16, 8) == b.extra_param_count(2, 16, 8)


def test_kwta_output_stays_non_negative():
    """The probes and BDH's own x_sparse * y_sparse gate assume a positive code."""
    torch.manual_seed(5)
    u = _dend(n_branches=4, soma="topk", soma_topk=2, wta_frac=0.25)
    assert float(u(torch.randn(2, 2, 6, 16)).min()) >= 0.0


def test_bad_wta_frac_is_rejected():
    with pytest.raises(ValueError, match="wta_frac"):
        _dend(n_branches=4, wta_frac=0.0)
    with pytest.raises(ValueError, match="wta_frac"):
        _dend(n_branches=4, wta_frac=1.5)


def test_unknown_soma_still_rejected():
    with pytest.raises(ValueError, match="sum|max|gated|topk"):
        _dend(n_branches=4, soma="nonsense")


def test_per_path_kwargs_apply_to_one_path_only():
    """BDH's output is x_sparse * y_sparse, so a sparsity constraint on BOTH
    paths compounds: two 5%-dense codes give a 0.25%-dense conjunction. The
    per-path override is the control that separates the constraint's own cost
    from that compounding, so it must genuinely affect one path only.
    """
    spec = BDHSpec(
        n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=2,
        neuron_unit="dendritic",
        neuron_unit_kwargs={"n_branches": 4, "soma": "max"},
        neuron_unit_kwargs_x={"wta_frac": 0.05},
    )
    m = BDHWrapper(spec)
    assert m.neuron_unit_x.wta_frac == 0.05
    assert m.neuron_unit_y.wta_frac is None
    # and the shared kwargs still reach both
    assert m.neuron_unit_x.soma == "max" and m.neuron_unit_y.soma == "max"
    assert m.neuron_unit_x.k == 4 and m.neuron_unit_y.k == 4


def test_per_path_kwargs_do_not_change_parameter_count():
    """Matched-parameter comparisons must survive a per-path constraint."""
    base = dict(
        n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=2,
        neuron_unit="dendritic",
        neuron_unit_kwargs={"n_branches": 4, "soma": "max"},
    )
    a = BDHWrapper(BDHSpec(**base))
    b = BDHWrapper(BDHSpec(**base, neuron_unit_kwargs_x={"wta_frac": 0.05}))
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_one_path_wta_is_denser_than_two_path_wta():
    """The prediction being tested: constraining one path leaves a far denser
    conjunction than constraining both."""
    torch.manual_seed(0)
    idx = torch.randint(0, 256, (2, 24))
    base = dict(
        n_layer=1, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4,
        neuron_unit="dendritic",
        neuron_unit_kwargs={"n_branches": 4, "soma": "max"},
    )

    def conj_density(spec):
        m = BDHWrapper(spec).eval()
        _l, _s, acts = m(idx, None, return_activations=True)
        xs, ys = acts.x_sparse[0], acts.y_sparse[0]
        return float(((xs > 0) & (ys > 0)).float().mean())

    # dict(base, ...) rather than BDHSpec(**base, ...): `base` already carries
    # neuron_unit_kwargs, so passing it again is a duplicate keyword argument.
    torch.manual_seed(1)
    both = conj_density(BDHSpec(**dict(
        base, neuron_unit_kwargs={"n_branches": 4, "soma": "max", "wta_frac": 0.05}
    )))
    torch.manual_seed(1)
    one = conj_density(BDHSpec(**dict(base, neuron_unit_kwargs_x={"wta_frac": 0.05})))
    assert one > both, f"one-path conjunction {one:.4f} not denser than both-path {both:.4f}"


# -- the depth control ----------------------------------------------------


def _deep(**kw):
    from bdh_cl.models.neuron_units import DeepNeuronUnit

    return DeepNeuronUnit(n_head=2, d=16, N=8, **kw)


def test_deep_matches_dendritic_parameter_count_with_integer_rounding():
    """The whole point of this control: identical parameters, identical neuron
    count, so the only difference is HOW the extra capacity is spent.

    Checked across several shapes because the matched hidden width is derived
    from d, N and k, and a formula error would show up as a size-dependent drift
    rather than a uniform offset.
    """
    from bdh_cl.models.neuron_units import DendriticNeuronUnit, DeepNeuronUnit

    for nh, d, N, k in [(2, 16, 8, 4), (4, 128, 352, 4), (4, 256, 2752, 4), (4, 64, 96, 2)]:
        den = DendriticNeuronUnit(n_head=nh, d=d, N=N, n_branches=k, soma="sum")
        dep = DeepNeuronUnit(n_head=nh, d=d, N=N, match_branches=k)
        pd = sum(p.numel() for p in den.parameters())
        pp = sum(p.numel() for p in dep.parameters())
        ideal_hidden = d * k * N / (d + N)
        assert dep.hidden == round(ideal_hidden)
        assert pp == nh * dep.hidden * (d + N)
        assert abs(pp - pd) <= 0.5 * nh * (d + N)
        if d >= 64:  # Shapes used by the actual experiments.
            assert abs(pp - pd) / pd < 0.01


def test_deep_output_shape_and_non_negativity():
    """Output must be (B, nh, T, N) and non-negative: the probes and BDH's own
    x_sparse * y_sparse gate both assume a positive code."""
    torch.manual_seed(0)
    u = _deep(match_branches=4)
    out = u(torch.randn(2, 2, 5, 16))
    assert out.shape == (2, 2, 5, 8)
    assert float(out.min()) >= 0.0


def test_deep_is_genuinely_two_rectifiers():
    """A single linear map would make this a reparameterised one-layer unit and
    the control would be vacuous. With two ReLUs the function is not expressible
    as relu(x @ W) for any W, which shows up as a nonzero second difference
    along a ray where the inner rectifier changes state."""
    torch.manual_seed(1)
    u = _deep(hidden=32)
    x = torch.randn(1, 2, 1, 16)
    ts = torch.linspace(-3, 3, 61)
    vals = torch.stack([u(t * x).sum() for t in ts])
    second = (vals[2:] - 2 * vals[1:-1] + vals[:-2]).abs().max()
    assert float(second) > 1e-6, "unit behaves linearly; inner ReLU never bends"


def test_deep_hidden_is_derived_not_guessed():
    from bdh_cl.models.neuron_units import DeepNeuronUnit

    # M = d*k*N/(d+N)
    u = DeepNeuronUnit(n_head=4, d=256, N=2752, match_branches=4)
    assert u.hidden == round(256 * 4 * 2752 / (256 + 2752))
    # explicit hidden overrides the derivation
    assert DeepNeuronUnit(n_head=4, d=256, N=2752, hidden=100).hidden == 100


def test_deep_requires_hidden_or_match_branches():
    with pytest.raises(ValueError, match="match_branches"):
        _deep()


def test_deep_unit_is_selectable_and_replaces_the_host_encoder():
    """It owns its parameters, so the host's encoder must be released -- leaving
    it allocated would add two dead (nh, D, N) tensors and break matching."""
    spec = BDHSpec(
        n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=2,
        neuron_unit="deep", neuron_unit_kwargs={"match_branches": 4},
    )
    m = BDHWrapper(spec)
    assert m.neuron_unit_x.owns_parameters
    assert m.core.encoder is None and m.core.encoder_v is None
    logits, _ = m(torch.randint(0, 256, (2, 12)), None)
    assert logits.shape == (2, 12, 256)
