import torch
from torch.nn import functional as F
from bdh_cl.scripts.mnist_fewshot_recovery import coordinate_route,adapted_head


def test_error_route_search_does_not_increase_support_loss():
 torch.manual_seed(3);z=torch.randn(8,6,4).tanh();y=torch.arange(8)%10;w=torch.randn(10,6);b=torch.randn(10);initial=torch.zeros(6,dtype=torch.long)
 with torch.no_grad():
  before=F.cross_entropy(F.linear(z[:,:,0],w,b),y);route=coordinate_route(z,y,w,b,initial);h=z.gather(2,route[None,:,None].expand(8,-1,-1)).squeeze(-1);after=F.cross_entropy(F.linear(h,w,b),y)
 assert after<=before+1e-6
 assert torch.equal(initial,torch.zeros(6,dtype=torch.long))


def test_head_adaptation_leaves_original_weights_frozen():
 torch.manual_seed(2);x=torch.randn(6,8);y=torch.tensor([0,1,0,1,0,1]);w=torch.randn(10,8);b=torch.randn(10);oldw=w.clone();oldb=b.clone();aw,ab=adapted_head(x,y,w,b)
 assert torch.equal(w,oldw) and torch.equal(b,oldb)
 assert F.cross_entropy(F.linear(x,aw,ab),y)<F.cross_entropy(F.linear(x,w,b),y)
