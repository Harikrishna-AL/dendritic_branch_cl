import torch
from torch import nn

from bdh_cl.scripts.continual_baseline_memory_suite import OnlineEWC, Replay, SI


def test_empirical_fisher_matches_closed_form_linear_classifier():
    model = nn.Linear(2,2,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2,-.3],[-.1,.4]]))
    x = torch.tensor([[1.,2.],[-2.,.5],[.4,-1.]])
    y = torch.tensor([0,1,0])
    residual = model(x).softmax(1).detach()-torch.nn.functional.one_hot(y,2)
    expected = (residual[:,:,None]*x[:,None,:]).square().mean(0)
    regularizer = OnlineEWC(model,1.)
    regularizer.consolidate(model,x,y)
    assert torch.allclose(regularizer.fisher['weight'],expected,atol=1e-7)
    regularizer.consolidate(model,x,y)
    assert torch.allclose(regularizer.fisher['weight'],2*expected,atol=1e-7)


def test_si_loss_descent_increases_importance_and_anchors_penalty():
    model = nn.Linear(1,1,bias=False)
    with torch.no_grad():
        model.weight.zero_()
    regularizer = SI(model,1.)
    before = [model.weight.detach().clone()]
    with torch.no_grad():
        model.weight.fill_(-.2)
    regularizer.observe(model,before,[torch.full_like(model.weight,.5)])
    regularizer.consolidate(model)
    assert torch.allclose(regularizer.omega[0],torch.full_like(model.weight,.1/.14))
    assert regularizer.penalty(model)==0
    with torch.no_grad():
        model.weight.add_(.1)
    assert regularizer.penalty(model)>0


def test_replay_contains_only_observed_examples_and_roundtrips_bytes():
    replay = Replay(4,1)
    assert replay.sample(2) is None
    raw = torch.stack([torch.full((784,),i,dtype=torch.uint8) for i in [0,64,128]])
    x = (raw.float()/255-.13062755)/.30810780
    replay.observe(x,torch.tensor([0,1,2]))
    assert replay.used==3 and replay.seen==3
    bx,by = replay.sample(20)
    restored = ((bx*.30810780+.13062755)*255).round().to(torch.uint8)
    assert torch.equal(restored,raw[by])
