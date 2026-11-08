import torch
from torch.nn import functional as F
from bdh_cl.scripts.rawpixel_gating_baselines import HAT, TWTA, FTN


def test_hat_protected_gradients_and_native_forward():
    torch.manual_seed(7)
    m=HAT();x=torch.randn(8,784)
    m.eval()
    actual=m(x,1)
    expected=m.net(torch.tensor([1]),x,s=400)[0][0]
    assert torch.equal(actual,expected)
    m.consolidate(0)
    F.cross_entropy(m(x,1),torch.arange(8)%10).backward()
    before=m.net.fc1.weight.grad.clone()
    view=m.net.get_view_for('fc1.weight',[m.used1,m.used2])
    m.restrict()
    assert torch.allclose(m.net.fc1.weight.grad,before*(1-view))


def test_twta_stochastic_competition_trains_current_posterior_only():
    torch.manual_seed(7)
    m=TWTA();x=torch.randn(8,784)
    F.cross_entropy(m(x,2,tau=.5),torch.arange(8)%10).backward()
    assert m.posterior.grad[2].abs().sum()>0
    assert m.posterior.grad[:2].count_nonzero()==0
    assert m.posterior.grad[3:].count_nonzero()==0
    gate=F.one_hot(m.posterior[2].argmax(-1),4).float()
    assert (gate.sum(-1)==1).all()
    assert torch.equal(m(x,2),m.head((m.first(x).reshape(-1,166,4)*gate).flatten(1)))


def test_ftn_mask_proposal_freezes_weights_and_is_sparse():
    torch.manual_seed(7)
    m=FTN();x=torch.randn(8,784);y=torch.arange(8)%10
    before={k:v.clone() for k,v in m.state_dict().items()}
    mask=m.configure(x,y)
    assert mask.sum()==8
    assert ((mask==0)|(mask==1)).all()
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    for p in m.parameters():assert p.grad is None
    m.eval()
    first=torch.zeros(64);first[:8]=1
    second=torch.zeros(64);second[8:16]=1
    opt=torch.optim.SGD(m.parameters(),lr=.1)
    old=m.forward_mask(x,first).detach().clone()
    F.cross_entropy(m.forward_mask(x,second),y).backward();opt.step()
    assert torch.equal(old,m.forward_mask(x,first))
