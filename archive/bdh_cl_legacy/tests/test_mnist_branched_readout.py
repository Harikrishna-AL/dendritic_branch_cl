import torch
from torch.nn import functional as F
from bdh_cl.scripts.mnist_branched_readout import Branched,objective

def test_initial_branch_readout_matches_shared_for_arbitrary_gates():
 torch.manual_seed(101);a=Branched('shared',101)
 torch.manual_seed(101);b=Branched('branched',101);x=torch.randn(4,784)
 assert torch.allclose(a.responses(x),b.responses(x))
 z=a.responses(x);gate=torch.rand_like(z).softmax(-1)
 assert torch.allclose(torch.einsum('sik,cik->sc',z*gate,a.vectors())+a.head.bias,torch.einsum('sik,cik->sc',z*gate,b.vectors())+b.head.bias,atol=1e-6)

def test_branch_substitution_matches_direct_route():
 torch.manual_seed(101);m=Branched('branched',101)
 with torch.no_grad():m.head.weight.normal_()
 x=torch.randn(3,784);raw,z,v,selected,logits=m.components(x)
 alt=logits+z[:,2,3,None]*v[:,2,3][None]-selected[:,2,:]
 gate=m.gates().clone();gate[2]=F.one_hot(torch.tensor(3),4)
 direct=torch.einsum('sik,cik->sc',z*gate,v)+m.head.bias
 assert torch.allclose(alt,direct,atol=1e-5)
 loss,feedback=objective(m,x,torch.arange(3),1);loss.backward()
 assert torch.isfinite(m.head.weight.grad).all() and not feedback.requires_grad


def test_all_readouts_preserve_local_error_traces():
 for arm in ["shared","branched","branched_fixed"]:
  torch.manual_seed(101);m=Branched(arm,101);assert m.arm=="local"
  values=torch.arange(672).reshape(168,4).float();m.observe(values)
  assert not torch.equal(m.error[0],m.error[1])
