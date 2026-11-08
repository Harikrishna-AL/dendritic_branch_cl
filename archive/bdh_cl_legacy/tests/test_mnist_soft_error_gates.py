import torch
from bdh_cl.scripts.mnist_soft_error_gates import SoftErrorModel,set_temperature


def test_soft_error_gate_prefers_low_error_and_sharpens():
 m=SoftErrorModel('soft_0.1',101)
 m.error[:]=torch.tensor([.1,.2,.3,.4])
 m.temperature=.1;soft=m.gates();m.temperature=.001;sharp=m.gates()
 assert torch.allclose(soft.sum(-1),torch.ones(168))
 assert torch.all(soft[:,0]>soft[:,1])
 assert torch.all(sharp[:,0]>soft[:,0])
 assert torch.all(sharp[:,0]>.999)


def test_auxiliary_substitution_replaces_soft_mixture_exactly():
 torch.manual_seed(101);m=SoftErrorModel('soft_0.1',101);x=torch.randn(3,784)
 m.error[:]=torch.tensor([.1,.2,.3,.4]);_,alt=m(x,counterfactual=True)
 z=m.responses(x);h=(z*m.gates()).sum(-1);h[:,6]=z[:,6,2]
 assert torch.allclose(alt[:,6,2],m.head(h),atol=1e-6)


def test_schedule_restarts_without_resetting_error_memory():
 m=SoftErrorModel('anneal',101);before=m.error.clone()
 set_temperature(m,0,0,10);assert abs(m.temperature-.1)<1e-9
 set_temperature(m,1,9,10);assert abs(m.temperature-.001)<1e-9
 set_temperature(m,0,0,10);assert abs(m.temperature-.1)<1e-9
 assert torch.equal(before,m.error)
