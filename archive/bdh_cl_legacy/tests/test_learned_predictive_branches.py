import torch
from bdh_cl.scripts.learned_predictive_branches import Model


def test_random_branches_are_not_assigned_channels_and_memory_is_causal():
    torch.manual_seed(0);model=Model('memory_competition',streams=2)
    assert (model.weight!=0).all()
    x=torch.randn(2,16);_,q,_=model.predict(x)
    assert not model.seen and model.cross.count_nonzero()==0
    model.commit(q);assert model.seen and model.cross.count_nonzero()==0
    prior=model.previous.clone();_,nextq,_=model.predict(torch.randn(2,16))
    model.commit(nextq)
    assert torch.allclose(model.cross,.05*prior*nextq)
    assert not model.cross.requires_grad


def test_competition_updates_selected_branch_only_with_direct_loss():
    torch.manual_seed(0);model=Model('memory_competition',streams=2)
    x=torch.randn(2,16);pred,_,winner=model.predict(x)
    assert (winner==0).all()
    (pred-torch.ones_like(pred)).square().mean().backward()
    assert model.weight.grad[1:].count_nonzero()==0
    assert model.bias.grad[1:].count_nonzero()==0
