import math

import numpy as np
import pytest
import torch

pytest.importorskip("minatar")
from bdh_cl.scripts import pop_cl_rl as R  # noqa: E402


def test_obs_padded_to_ten_channels():
    for g in R.GAMES:
        e = R.make_env(g, 0)
        e.reset()
        o = R.obs(e)
        assert o.shape == (10, 10, 10) and o.dtype == np.uint8


@pytest.mark.parametrize("arm", R.ARMS)
def test_qnet_shapes(arm):
    net = R.QNet(arm, width=64, k=8)
    q = net(torch.zeros(3, 10, 10, 10))
    assert q.shape == (3, R.N_ACTIONS)
    if net.is_pop:
        assert net.hidden.tired == ("tired" in arm)


def _adam_step(net):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.rand(4, 10, 10, 10)).sum().backward()
    opt.step()
    return opt


def test_reset_mlp_redraws_dead_units_only():
    net = R.QNet("dqn", width=16)
    opt = _adam_step(net)
    dead = torch.zeros(16, dtype=torch.bool)
    dead[[2, 5]] = True
    w0, h0 = net.hidden[0].weight.clone(), net.head.weight.clone()
    R.reset_mlp(net, opt, dead)
    w1 = net.hidden[0].weight
    assert not torch.equal(w1[dead], w0[dead]) and torch.equal(w1[~dead], w0[~dead])
    assert (net.head.weight[:, dead] == 0).all() and torch.equal(net.head.weight[:, ~dead], h0[:, ~dead])
    assert (opt.state[net.hidden[0].weight]["exp_avg"][dead] == 0).all()


def test_dormant_mlp_flags_silent_units():
    net = R.QNet("dqn", width=16)
    with torch.no_grad():
        net.hidden[0].bias[:4] = -1e4
    dead = R.dormant_mlp(net, torch.rand(32, 10, 10, 10), 0.1)
    assert dead[:4].all() and not dead[4:].all()


def test_reset_pop_restores_plasticity():
    net = R.QNet("pop_tired_reset", width=16, k=4)
    opt = _adam_step(net)
    net.hidden.fired_prev.fill_(100.0)
    net.hidden.end_task(1.0)
    assert (net.hidden.grad_scale < 0.02).all()
    dead = torch.zeros(16, dtype=torch.bool)
    dead[0] = True
    w0 = net.hidden.w.clone()
    R.reset_pop(net, opt, dead)
    assert net.hidden.grad_scale[0] == 1 and net.hidden.fired_prev[0] == 0
    assert net.hidden.grad_scale[1] < 0.02
    assert not torch.equal(net.hidden.w[0], w0[0]) and torch.equal(net.hidden.w[1:], w0[1:])
    assert (net.head.weight[:, 0] == 0).all()


TINY = ["--games", "breakout,freeway", "--cycles", "2", "--steps", "300", "--learn-start", "50",
        "--buffer", "300", "--eps-steps", "100", "--width", "32", "--k", "4", "--redo-every", "50",
        "--eval-episodes", "1", "--curve-bin", "100", "--device", "cpu", "--no-wandb"]


@pytest.mark.parametrize("arm", R.ARMS)
def test_end_to_end(arm, tmp_path):
    out = R.run(R.parse(["--arm", arm, "--out", str(tmp_path / "o.json")] + TINY))
    assert len(out["log"]) == 4
    assert [r["game"] for r in out["log"]] == ["breakout", "freeway"] * 2
    assert len(out["summary"]["breakout"]["train_by_cycle"]) == 2
    assert set(out["log"][-1]["eval"]) == {"breakout", "freeway"}
    assert not math.isnan(out["log"][0]["dormant_last"])
    if arm in ("dqn", "pop", "pop_tired"):
        assert all(r["resets"] == 0 for r in out["log"])
    if arm == "pop_tired":
        assert out["log"][-1]["tired_mean"] > 0
    if arm == "pop":
        assert out["log"][-1]["tired_mean"] > 0          # stats tracked, but gradient untouched
        assert not R.QNet("pop").hidden.tired


def test_homeo_boosts_silent_neuron_and_keeps_k_active():
    torch.manual_seed(0)
    layer = R.HomeoPop(8, 16, 2, 4, 0.5, eta=0.5, alpha=0.1)
    with torch.no_grad():
        layer.b[0] = -1e3                       # neuron 0 can never win on its own
    x = torch.randn(32, 8)
    fired0 = []
    for _ in range(300):
        a = layer(x)
        assert torch.allclose(a.detach().sum(1), torch.full((32,), 4.0))
        fired0.append(float((a.detach()[:, 0] > 0.5).sum()))
    assert fired0[0] == 0 and layer.e[0] > 0          # silent at first, so it was boosted
    assert sum(fired0[-50:]) > 0                      # ... until it joined the active set
    assert layer.r.sum().item() == pytest.approx(4.0, rel=0.05)   # population rate stays k


def test_homeo_up_only_never_suppresses():
    layer = R.HomeoPop(8, 16, 2, 4, 0.5, eta=0.5, alpha=0.1, up_only=True)
    for _ in range(50):
        layer(torch.randn(32, 8))
    assert (layer.e >= 0).all()
    both = R.HomeoPop(8, 16, 2, 4, 0.5, eta=0.5, alpha=0.1)
    for _ in range(50):
        both(torch.randn(32, 8))
    assert (both.e < 0).any()


def test_homeo_eval_does_not_update():
    layer = R.HomeoPop(8, 16, 2, 4, 0.5).eval()
    e0 = layer.e.clone()
    layer(torch.randn(5, 8))
    assert torch.equal(layer.e, e0)


def test_homeo_spreads_firing_in_qnet():
    torch.manual_seed(0)
    x = torch.rand(64, 10, 10, 10)
    plain, homeo = R.QNet("pop", width=64, k=4), R.QNet("pop_homeo", width=64, k=4, homeo_eta=0.2, homeo_alpha=0.05)
    homeo.load_state_dict(plain.state_dict(), strict=False)
    for _ in range(200):
        homeo(x)
    with torch.no_grad():
        plain.eval(); homeo.eval()
        used = lambda n: int((n.hidden(n.features(x)).sum(0) > 0).sum())  # noqa: E731
        assert used(homeo) > used(plain)


def test_graded_pop_same_winners_positive_magnitudes():
    torch.manual_seed(0)
    binary, graded = R.PopLayer(8, 16, 2, 4, 0.5), R.GradedPop(8, 16, 2, 4, 0.5)
    graded.load_state_dict(binary.state_dict())
    x = torch.randn(10, 8)
    b, g = binary(x).detach() > 0.5, graded(x).detach()
    assert torch.equal(g > 0, b) and (b.sum(1) == 4).all()
    assert g[b].std() > 0                                  # not all the same value


def test_graded_gradient_paths():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    true = R.GradedPop(8, 16, 2, 4, 0.5)
    out = true(x)
    out.sum().backward()
    never = (out.detach() > 0).sum(0) == 0                 # neurons that never won
    assert never.any() and (true.w.grad[never] == 0).all()  # true gradient: winners only
    st = R.GradedPop(8, 16, 2, 4, 0.5, surrogate=True)
    st.load_state_dict(true.state_dict())
    st(x).sum().backward()
    assert (st.w.grad[never].abs().sum() > 0)              # surrogate reaches losers too
    assert torch.allclose(st(x).detach(), out.detach(), atol=1e-6)   # same forward values
