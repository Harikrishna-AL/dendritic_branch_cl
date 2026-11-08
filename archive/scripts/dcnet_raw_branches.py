"""Raw-pixel branch replacement for DCNet's ResNet, with HAT protection.

Feature layer matches BranchBank: 3072 -> 128 groups x 10 tanh branches.
No CNN/pretrained encoder. DCNet's native projection and classifier modules
are retained, so this is an architectural ablation, not published DCNet.
"""
import torch
from torch import nn
from models.base_model import BaseModel


class RawBranchDCNet(BaseModel):
    def __init__(self, P, seed):
        P.embedding_dim=128
        super().__init__(128,P,num_classes=10)
        self.width,self.branches=128,10
        # Same first-layer initialization as the paired BranchBank pilot.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.first=nn.Linear(3072,1280)
        self.ec0=nn.ParameterList([nn.Parameter(torch.randn(1,1280)) for _ in range(P.n_tasks)])
        self.last_dim=128

    def mask(self,t,s=1):
        return torch.sigmoid(s*self.ec0[t])

    def penultimate(self,t,inputs,s,all_features=False):
        branches=torch.tanh(self.first(inputs.flatten(1)-.5)).reshape(-1,128,10)
        mask=self.mask(t,s)
        hidden=(branches*mask.reshape(1,128,10)).sum(-1)
        return hidden,[mask]


def update_masks(P,task,model,previous):
    current={'ec0':model.mask(task,P.smax).detach(),
             'simclr_layer.ec':model.simclr_layer.mask(task,P.smax).detach()}
    cumulative={k:v if previous is None else torch.maximum(previous[k],v) for k,v in current.items()}
    branch=cumulative['ec0'].reshape(1280)
    group=branch.reshape(128,10).max(-1).values
    projection=cumulative['simclr_layer.ec'].reshape(128)
    # Native HAT output/input connection protection. The first-layer mask can
    # remain broadcast-compressed because pixels have no learned input gates.
    back={'first.weight':1-branch[:,None], 'first.bias':1-branch,
          'simclr_layer.fc1.weight':1-torch.minimum(projection[:,None],group[None]),
          'simclr_layer.fc1.bias':1-projection}
    return cumulative,back


def verify_raw_model(model):
    assert not any(isinstance(m,nn.Conv2d) for m in model.modules())
    with torch.random.fork_rng():
        torch.manual_seed(model.P.seed)
        reference=nn.Linear(3072,1280)
    assert torch.equal(model.first.weight.detach().cpu(),reference.weight)
    assert torch.equal(model.first.bias.detach().cpu(),reference.bias)
    device=model.first.weight.device
    x=torch.rand(7,3,32,32,device=device)
    with torch.no_grad():
        manual=(torch.tanh(torch.nn.functional.linear(x.flatten(1)-.5,model.first.weight,model.first.bias)).reshape(-1,128,10)*model.mask(0,1).reshape(1,128,10)).sum(-1)
        actual,_=model.penultimate(0,x,1)
        assert torch.allclose(actual,manual,atol=1e-6)
        cumulative,back=update_masks(model.P,0,model,None)
        # Show that fully protected rows cannot change under the official SGD.
        from common.sgd_hat import SGD_hat
        parameter=nn.Parameter(torch.randn(5,4,device=device));before=parameter.detach().clone()
        parameter.hat=torch.tensor([[0],[0],[1],[1],[1]],device=device)
        opt=SGD_hat([parameter],lr=.1,momentum=.9)
        parameter.grad=torch.ones_like(parameter);opt.step(hat=True)
        assert torch.equal(parameter[:2],before[:2])
        assert not torch.equal(parameter[2:],before[2:])
    return {'no_convolutional_modules':True,'first_layer_matches_paired_branch_initialization':True,
            'direct_raw_branch_forward_matches':True,'broadcast_HAT_mask_protects_rows':True,
            'first_layer_parameters':sum(p.numel() for p in model.first.parameters())}
