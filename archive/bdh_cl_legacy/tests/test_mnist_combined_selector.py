import torch
from bdh_cl.scripts.mnist_combined_selector import Model,evaluate


def test_hard_selector_receives_classification_gradient():
 torch.manual_seed(2);m=Model('selector_hard',101);x=torch.randn(4,784)
 logits=m.prediction(x);torch.nn.functional.cross_entropy(logits,torch.arange(4)).backward()
 assert m.selector[0].weight.grad.abs().sum()>0
 assert m.first.weight.grad.abs().sum()>0


def test_soft_and_hard_use_same_branch_outputs():
 torch.manual_seed(2);m=Model('selector_soft',101);x=torch.randn(3,784)
 with torch.no_grad():
  for p in m.selector.parameters():p.zero_()
  raw=m.responses(x,False)
  assert torch.allclose(m.prediction(x,raw),m.head(raw.tanh().mean(-1)),atol=1e-6)
  before={k:v.clone() for k,v in m.state_dict().items()}
  m.prediction(x)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
