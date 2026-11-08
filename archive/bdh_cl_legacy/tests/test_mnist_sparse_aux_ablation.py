import copy
import torch
from torch.nn import functional as F
from bdh_cl.scripts.mnist_combined_selector import Model
from bdh_cl.scripts.mnist_sparse_aux_ablation import objective


def test_zero_auxiliary_preserves_feedback_but_has_ce_only_gradients():
 torch.manual_seed(101);m=Model('baseline',101);other=copy.deepcopy(m);x=torch.randn(4,784);y=torch.tensor([0,1,2,3])
 loss,feedback=objective(m,x,y,0);_,aux_feedback=objective(other,x,y,.5)
 assert torch.allclose(feedback,aux_feedback,atol=1e-7)
 assert not feedback.requires_grad
 loss.backward();F.cross_entropy(other(x),y).backward()
 for a,b in zip(m.parameters(),other.parameters()):assert torch.allclose(a.grad,b.grad,atol=1e-7)


def test_auxiliary_arm_matches_existing_objective():
 torch.manual_seed(101);m=Model('baseline',101);x=torch.randn(4,784);y=torch.tensor([0,1,2,3]);loss,_=objective(m,x,y,.5)
 logits,alt=m(x,True);errors=F.cross_entropy(alt.reshape(-1,10),y[:,None,None].expand(alt.shape[:-1]).reshape(-1),reduction='none')
 assert torch.allclose(loss,F.cross_entropy(logits,y)+.5*errors.mean(),atol=1e-7)
