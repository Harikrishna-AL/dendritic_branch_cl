import numpy as np
import pytest
import torch

from bdh_cl.data import cifar100
from bdh_cl.scripts import pop_cl_cifar as M


def test_scale_grad_identity_forward_scaled_backward():
    x = torch.ones(3, requires_grad=True)
    g = torch.tensor([1.0, 0.5, 0.0])
    y = M.scale_grad(x, g)
    assert torch.equal(y, x.detach())
    y.sum().backward()
    assert torch.allclose(x.grad, g)


def test_pop_layer_binary_topk_with_gradient():
    layer = M.PopLayer(d=8, n=30, branches=3, k=5, temp=0.5)
    h = torch.randn(4, 8, requires_grad=True)
    a = layer(h)
    assert torch.allclose(a.sum(1), torch.full((4,), 5.0))
    assert torch.allclose(a.detach(), a.detach().round())
    (a * torch.randn_like(a)).sum().backward()
    assert h.grad.abs().sum() > 0 and layer.w.grad.abs().sum() > 0
    assert layer.fired_task.sum() == 20


def test_tiredness_blocks_gradient_of_saturated_neurons():
    layer = M.PopLayer(d=8, n=10, branches=2, k=3, temp=0.5)
    layer.tired = True
    layer.fired_prev[:5] = 1e9                     # neurons 0-4 exhausted
    t = layer.end_task(tau=1.0)
    assert (t[:5] > 0.999).all() and (t[5:] == 0).all()
    h = torch.randn(64, 8)
    layer(h).sum().backward()
    assert layer.w.grad[:5].abs().max() < 1e-6 * layer.w.grad[5:].abs().max() + 1e-12


def test_summarize_forgetting():
    m = [[0.9], [0.5, 0.8], [0.4, 0.6, 0.7]]
    s = M.summarize(m)
    assert s["final_avg_acc"] == pytest.approx(np.mean([0.4, 0.6, 0.7]))
    # task0 best 0.9 -> 0.4 ; task1 best 0.8 -> 0.6
    assert s["forgetting"] == pytest.approx((0.5 + 0.2) / 2)


@pytest.fixture
def tiny_cifar(monkeypatch):
    rng = np.random.default_rng(0)

    def fake(root):
        def make(n):
            y = np.repeat(np.arange(100), n)
            return rng.integers(0, 256, (len(y), 3, 32, 32), dtype=np.uint8), y
        return make(3), make(2), make(2), list(range(100))

    monkeypatch.setattr(cifar100, "load", fake)


# CUDA when present: it is the path the experiment uses, and CPU conv backward
# aborted natively on a Capella GPU node (job 4386551) while passing elsewhere.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("arm", M.ARMS)
def test_every_arm_runs_end_to_end(tiny_cifar, arm):
    a = M.parser().parse_args(["--data", "x", "--arm", arm, "--epochs", "1", "--n-tasks", "2",
                               "--nf", "4", "--pop-n", "40", "--pop-k", "4", "--device", DEVICE,
                               "--no-wandb", "--batch", "16"])
    r = M.run(a)
    for k, s in r["summary"].items():
        assert 0.0 <= s["final_avg_acc"] <= 1.0
    if arm not in M.JOINT:
        assert len(r["matrices"]["ncm"]) == 2 and len(r["matrices"]["head"][1]) == 2
    assert 0.0 <= r["ncm_oracle_final"] <= 1.0


@pytest.mark.parametrize("arm,backbone,layers", [
    ("pop_tired", "none", 2), ("joint_pop", "none", 1), ("finetune_local", "mlp", 1), ("lwf", "mlp", 1)])
def test_no_conv_variants_run(tiny_cifar, arm, backbone, layers):
    a = M.parser().parse_args(["--data", "x", "--arm", arm, "--epochs", "1", "--n-tasks", "2",
                               "--backbone", backbone, "--pop-layers", str(layers),
                               "--pop-n", "40", "--pop-k", "4", "--device", DEVICE,
                               "--no-wandb", "--batch", "16"])
    r = M.run(a)
    assert 0.0 <= r["summary"]["head"]["final_avg_acc"] <= 1.0
    if arm == "pop_tired":
        assert "mean_L2" in r["log"][-1]["tired"]


def test_no_conv_model_has_no_conv():
    net = M.Net("pop", backbone="none", pop_n=40, pop_k=4, pop_layers=2)
    assert not any(isinstance(m, torch.nn.Conv2d) for m in net.modules())
    assert len(net.pops) == 2


@pytest.fixture
def tiny_feats(tmp_path):
    g = torch.Generator().manual_seed(0)
    def split(n):
        y = torch.arange(100).repeat_interleave(n)
        return torch.randn(len(y), 32, generator=g) + torch.nn.functional.one_hot(y % 32, 32) * 3, y
    path = tmp_path / "f.pt"
    torch.save({"train": split(4), "valid": split(2), "test": split(2)}, path)
    return str(path)


@pytest.mark.parametrize("arm,backbone,epochs", [
    ("pop_tired", "none", "1"), ("finetune_local", "none", "0"), ("joint", "none", "1"), ("lwf", "mlp", "1")])
def test_feature_input_runs(tiny_feats, arm, backbone, epochs):
    a = M.parser().parse_args(["--data", "x", "--feats", tiny_feats, "--arm", arm, "--epochs", epochs,
                               "--n-tasks", "2", "--backbone", backbone, "--pop-n", "40", "--pop-k", "4",
                               "--device", DEVICE, "--no-wandb", "--batch", "16", "--clip", "1.0"])
    r = M.run(a)
    assert 0.0 <= r["summary"]["head"]["final_avg_acc"] <= 1.0


def test_images_passes_features_through():
    f = torch.randn(3, 512)
    assert M.images(f, augment=True) is f


def test_feats_rejects_conv_backbone(tiny_feats):
    a = M.parser().parse_args(["--data", "x", "--feats", tiny_feats, "--arm", "pop", "--no-wandb", "--device", "cpu"])
    with pytest.raises(SystemExit):
        M.run(a)
