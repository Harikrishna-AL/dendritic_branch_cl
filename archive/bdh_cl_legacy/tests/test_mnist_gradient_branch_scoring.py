import torch
from torch.nn import functional as F
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_gradient_branch_scoring import terms,exact_errors


def test_local_gradient_has_correct_per_example_scale():
 torch.manual_seed(101);m=Model('baseline',101);x=torch.randn(5,784);y=torch.arange(5)
 z,h,logits,ce,delta,linear,quad,a1,a2=terms(m,x,y)
 actual=torch.autograd.grad(ce,h,retain_graph=True)[0]*len(y)
 expected=(logits.detach().softmax(-1)-F.one_hot(y,10).float())@m.head.weight
 assert torch.allclose(actual,expected,atol=1e-6)
 base=F.cross_entropy(logits,y,reduction='none')[:,None,None]
 assert torch.allclose(a1,base.detach()+actual.detach()[:,:,None]*delta.detach(),atol=1e-6)


def test_second_order_reduces_error_for_small_substitutions():
 torch.manual_seed(7);m=Model('baseline',101);x=torch.randn(4,784);y=torch.arange(4)
 z,h,logits,*_=terms(m,x,y);delta=(z-h[:,:,None])*.01
 exact=exact_errors(m,y,h[:,:,None]+delta,h,logits)
 p=logits.softmax(-1);residual=p-F.one_hot(y,10).float();g=residual@m.head.weight;q=p@m.head.weight.square()-(p@m.head.weight).square();base=F.cross_entropy(logits,y,reduction='none')[:,None,None]
 first=base+g[:,:,None]*delta;second=first+.5*q[:,:,None]*delta.square()
 assert (second-exact).abs().mean()<(first-exact).abs().mean()


def test_exact_mode_equals_original_auxiliary_objective():
 from bdh_cl.scripts.mnist_gradient_branch_scoring import objective
 from bdh_cl.scripts.mnist_sparse_aux_ablation import objective as reference
 torch.manual_seed(101);m=Model('baseline',101);x=torch.randn(4,784);y=torch.arange(4)
 loss,feedback,exact=objective(m,x,y,'exact',0);expected,expected_feedback=reference(m,x,y,.5)
 assert exact and torch.allclose(loss,expected,atol=1e-7)
 assert torch.allclose(feedback,expected_feedback,atol=1e-7)
