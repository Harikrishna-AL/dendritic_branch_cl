"""Credit routing, teacher isolation and consolidation in the local pilot."""
import copy

import torch
import torch.nn.functional as F

from bdh_cl.scripts.dendritic_cl_pilot import Consolidation, Student, distill


def test_conditions_share_parameter_budget_and_initialization():
    original = Student(inputs=4, neurons=3, branches=4)
    summed = copy.deepcopy(original)
    summed.pooling = "sum"
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in summed.parameters())
    for a, b in zip(original.parameters(), summed.parameters()):
        torch.testing.assert_close(a, b)


def test_max_routes_gradient_only_to_active_winning_branch():
    model = Student(inputs=2, neurons=1, branches=3)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[1., 0.], [2., 0.], [-1., 0.]]]))
        model.readout.weight.fill_(1.)
    model(torch.tensor([[1., 0.]])).sum().backward()
    assert model.weight.grad[0, 1, 0] != 0
    assert not model.weight.grad[0, 0].any()
    assert not model.weight.grad[0, 2].any()


def test_distillation_target_is_detached_and_identical_logits_have_zero_loss():
    student = torch.randn(4, 10, requires_grad=True)
    teacher = torch.randn(4, 10, requires_grad=True)
    distill(student, teacher).backward()
    assert student.grad is not None and teacher.grad is None
    torch.testing.assert_close(distill(student.detach(), student.detach()), torch.tensor(0.), atol=1e-6, rtol=0)


def test_consolidation_preserves_weights_and_penalty_restores_displacement():
    torch.manual_seed(2)
    model = Student(inputs=4, neurons=3, branches=2)
    x, y = torch.randn(8, 4), torch.arange(8) % 10
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cons = Consolidation()
    cons.consolidate(model, x, y, samples=8)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n])
    assert float(cons.penalty(model).detach()) == 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.01)
    cons.penalty(model).backward()
    assert all((p.grad >= 0).all() for p in model.parameters())
    assert any(p.grad.any() for p in model.parameters())


def test_frozen_teacher_survives_student_update():
    model = Student(inputs=4, neurons=3, branches=2)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    x, y = torch.randn(4, 4), torch.tensor([0, 1, 2, 3])
    expected = teacher(x).detach().clone()
    F.cross_entropy(model(x), y).backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-.1)
    torch.testing.assert_close(teacher(x), expected)
    assert not any(p.requires_grad for p in teacher.parameters())
