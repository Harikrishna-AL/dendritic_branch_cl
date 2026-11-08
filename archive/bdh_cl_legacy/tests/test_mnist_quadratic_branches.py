import pytest
import torch

from bdh_cl.scripts.mnist_quadratic_branches import InteractionBranches


@pytest.mark.parametrize("branch_count", [2, 4])
def test_zero_interaction_matches_normalized_control_and_counterfactual_is_exact(branch_count):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    control = InteractionBranches("normalized", branches=branch_count)
    torch.manual_seed(7)
    model = InteractionBranches("quadratic", branches=branch_count)
    images = torch.randn(5, 784)
    assert torch.equal(control(images), model(images))
    with torch.no_grad():
        model.alpha.fill_(.2)
    logits, alternatives = model(images, counterfactual=True)
    branches = model.responses(images)
    hidden = (branches * model.gates()).sum(-1)
    replacement = hidden.clone()
    replacement[:, 3] = branches[:, 3, branch_count - 1]
    assert torch.allclose(alternatives[:, 3, branch_count - 1], model.head(replacement), atol=1e-6)
    alternatives.square().mean().backward()
    assert model.u.weight.grad.abs().sum() > 0
    assert model.v.weight.grad.abs().sum() > 0
