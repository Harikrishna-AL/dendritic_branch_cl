"""Fast checks of the model interface (no datasets needed): python -m pytest tests"""
import torch

from branch_routing.continual import ContinualTrainer, TrainConfig
from branch_routing.models import BranchCNN, BranchMLP, build_model


def _check(model, x, y):
    for mode in ['soft', 'trace']: assert model(x, mode).shape == (len(x), 10)
    routes = [r.expand(len(x), -1) for r in model.current_route()]
    assert model(x, 'route', routes).shape == (len(x), 10)
    loss, feedback = model.substitution_loss(x, y); loss.backward()
    assert [f.shape for f in feedback] == [layer.error.shape for layer in model.branch_layers()]


def test_branch_mlp():
    for depth in [1, 2]: _check(BranchMLP(depth, 0), torch.randn(8, 784), torch.randint(0, 10, (8,)))


def test_branch_cnn():
    for arm in ['conv_b4_sparse', 'conv_b1_wide']: _check(BranchCNN(arm, 0), torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8), torch.randint(0, 10, (4,)))


def test_plain_matches_branch_size():
    b, p = build_model('mnist', 'branch', 0), build_model('mnist', 'plain', 0)
    nb, np_ = (sum(q.numel() for q in m.parameters()) for m in (b, p)); assert abs(nb - np_) / nb < .01


def test_trace_update_and_route_storage():
    m = BranchMLP(1, 0); tr = ContinualTrainer(m, TrainConfig(), 0); before = m.layers[0].error.clone()
    tr.train_step(torch.randn(16, 784), torch.randint(0, 2, (16,)))
    assert not torch.equal(before, m.layers[0].error) and tr.routes[0].shape == (5, 168)


def test_hat_masks_and_protection():
    from branch_routing.continual import HATTrainer, TrainConfig as TC
    from branch_routing.models import build_hat
    m = build_hat('mnist', 0); tr = HATTrainer(m, TC(), 0)
    x, y = torch.randn(8, 784), torch.randint(0, 2, (8,))
    opt = torch.optim.SGD(m.parameters(), lr=.1); tr._step(opt, x, y, 0, 1.)
    tr.used = [torch.ones(m.width), torch.ones(m.width)]  # everything used -> hidden weights frozen
    before = m.fc1.weight.clone(); tr._step(opt, x, y, 1, 1.); assert torch.equal(before, m.fc1.weight)


def test_gate_rules_override_selection():
    from branch_routing.continual import GateRule
    m = BranchMLP(1, 0)
    for rule in ['fixed_random', 'random_per_batch', 'most_active', 'task_assigned']:
        tr = ContinualTrainer(m, TrainConfig(), 0, gate_rule=GateRule(rule, 0)); tr.gate_rule.task_start(m, 0)
        tr.train_step(torch.randn(16, 784), torch.randint(0, 2, (16,))); assert m.layers[0].override is not None
        GateRule.clear(m)
    assert m.layers[0].override is None
