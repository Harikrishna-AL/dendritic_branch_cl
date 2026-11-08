"""The wrapper must reproduce upstream BDH exactly.

``BDHWrapper.forward`` re-walks upstream's layer loop so it can expose
per-layer activations and swap the neuron nonlinearity. These tests are the
proof that, with ``neuron_unit="default"``, it changes nothing.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.config import BDHSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import (  # noqa: E402
    BDHWrapper,
    masked_cross_entropy,
    upstream_bdh,
    upstream_config,
)
from bdh_cl.models.param_accounting import bdh_param_count, count_module  # noqa: E402

SPECS = [
    BDHSpec(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4, dropout=0.0),
    BDHSpec(n_layer=4, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0),
    # A dropout-carrying config, exercised in eval() mode where drop is identity.
    BDHSpec(n_layer=3, n_embd=32, n_head=2, mlp_internal_dim_multiplier=8, dropout=0.1),
]


def _batch(seed, B=2, T=16, vocab=256):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (B, T), generator=g)
    tgt = torch.randint(0, vocab, (B, T), generator=g)
    return idx, tgt


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"d{s.n_embd}L{s.n_layer}")
def test_forward_matches_upstream_eval(spec):
    """Same parameters in, same logits and loss out."""
    torch.manual_seed(1337)
    wrapper = BDHWrapper(spec).eval()

    torch.manual_seed(1337)
    reference = upstream_bdh.BDH(upstream_config(spec)).eval()

    # Identical init by construction (same seed, same allocation order), but
    # assert it rather than trust it.
    wsd, rsd = wrapper.core.state_dict(), reference.state_dict()
    assert wsd.keys() == rsd.keys()
    for k in rsd:
        assert torch.equal(wsd[k], rsd[k]), f"init mismatch at {k}"

    idx, tgt = _batch(0)
    with torch.no_grad():
        ref_logits, ref_loss = reference(idx, tgt)
        got_logits, got_loss = wrapper(idx, tgt)

    assert torch.allclose(got_logits, ref_logits, atol=0, rtol=0), "logits differ"
    assert torch.allclose(got_loss, ref_loss, atol=0, rtol=0), "loss differs"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"d{s.n_embd}L{s.n_layer}")
def test_forward_matches_upstream_train_mode(spec):
    """Train mode too. Dropout is the only stochastic op, so we seed it and
    check the two modules consume the RNG identically."""
    torch.manual_seed(1337)
    wrapper = BDHWrapper(spec).train()
    torch.manual_seed(1337)
    reference = upstream_bdh.BDH(upstream_config(spec)).train()
    reference.load_state_dict(wrapper.core.state_dict())

    idx, tgt = _batch(1)

    torch.manual_seed(99)
    ref_logits, ref_loss = reference(idx, tgt)
    torch.manual_seed(99)
    got_logits, got_loss = wrapper(idx, tgt)

    assert torch.allclose(got_logits, ref_logits, atol=0, rtol=0)
    assert torch.allclose(got_loss, ref_loss, atol=0, rtol=0)


def test_return_activations_does_not_change_output():
    """Asking for probes must not perturb the computation."""
    spec = SPECS[1]
    torch.manual_seed(7)
    w = BDHWrapper(spec).eval()
    idx, tgt = _batch(2)
    with torch.no_grad():
        a_logits, a_loss = w(idx, tgt)
        b_logits, b_loss, acts = w(idx, tgt, return_activations=True)
    assert torch.allclose(a_logits, b_logits, atol=0, rtol=0)
    assert torch.allclose(a_loss, b_loss, atol=0, rtol=0)

    # Shapes and non-negativity, which every probe relies on.
    assert len(acts.x_sparse) == spec.n_layer
    assert len(acts.y_sparse) == spec.n_layer
    B, T = idx.shape
    for t in (*acts.x_sparse, *acts.y_sparse):
        assert t.shape == (B, spec.n_head, T, spec.N)
        assert (t >= 0).all(), "NeuronUnit output must be non-negative"


def test_gradients_match_upstream():
    """Identical backward, not just identical forward."""
    spec = SPECS[0]
    torch.manual_seed(5)
    w = BDHWrapper(spec).eval()
    ref = upstream_bdh.BDH(upstream_config(spec)).eval()
    ref.load_state_dict(w.core.state_dict())

    idx, tgt = _batch(3)
    w(idx, tgt)[1].backward()
    ref(idx, tgt)[1].backward()

    got = dict(w.core.named_parameters())
    exp = dict(ref.named_parameters())
    for name, p in exp.items():
        assert torch.allclose(got[name].grad, p.grad, atol=1e-7, rtol=1e-5), name


def test_masked_cross_entropy_reduces_to_upstream():
    """loss_mask=None must be bit-identical to plain cross-entropy."""
    torch.manual_seed(11)
    logits = torch.randn(3, 9, 256)
    targets = torch.randint(0, 256, (3, 9))
    plain = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    assert torch.allclose(masked_cross_entropy(logits, targets, None), plain, atol=0, rtol=0)

    # An all-ones mask is the same loss.
    ones = torch.ones(3, 9)
    assert torch.allclose(masked_cross_entropy(logits, targets, ones), plain, atol=1e-6)

    # A mask must equal the cross-entropy restricted to the kept positions.
    mask = torch.zeros(3, 9)
    mask[:, 4:] = 1
    restricted = torch.nn.functional.cross_entropy(
        logits[:, 4:].reshape(-1, 256), targets[:, 4:].reshape(-1)
    )
    assert torch.allclose(masked_cross_entropy(logits, targets, mask), restricted, atol=1e-6)


def test_analytic_param_count_matches_module():
    """The Step 7 accounting utility must agree with a real module, and BDH's
    parameter count must not depend on n_layer (weights are shared)."""
    for spec in SPECS:
        w = BDHWrapper(spec)
        analytic = bdh_param_count(spec)["total"]
        empirical = count_module(w)["total"]
        assert analytic == empirical, (spec, analytic, empirical)

    a = BDHSpec(n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8)
    b = BDHSpec(n_layer=16, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8)
    assert bdh_param_count(a)["total"] == bdh_param_count(b)["total"]
    assert count_module(BDHWrapper(a))["total"] == count_module(BDHWrapper(b))["total"]
