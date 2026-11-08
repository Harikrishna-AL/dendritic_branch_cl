from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.scripts.strong_replay_comparison import Replay, ace_mask, make, objective, official_node


class FixedReplay:
    def __init__(self,method):
        self.method = method
        self.used = 2
        self.index = 0
        self.added = None
        self.samples = [
            {'x':torch.tensor([[1.,-1.],[-.5,.2]]),'y':torch.tensor([0,0]),
             'logits':torch.tensor([[.2,-.4,.1],[.1,.2,-.1]])},
            {'x':torch.tensor([[.2,.8],[-.3,.4]]),'y':torch.tensor([0,1]),
             'logits':torch.tensor([[.4,.2,.1],[.2,-.3,.4]])}]

    def draw(self,n):
        sample = self.samples[self.index]
        self.index += 1
        return sample

    def get_data(self,n,**kwargs):
        sample = self.draw(n)
        result = (sample['x'],sample['y'])
        return result+(sample['logits'],) if self.method=='derpp' else result

    def is_empty(self):
        return False

    def __len__(self):
        return self.used

    def add_data(self,**kwargs):
        self.added = kwargs


@pytest.mark.parametrize('method',['derpp','er_ace'])
def test_objective_and_update_match_official_observe(method):
    """Actual upstream source runs on fixed draws, rather than a retyped oracle."""
    torch.manual_seed(7)
    reference = nn.Linear(2,3)
    port = deepcopy(reference)
    x = torch.tensor([[.3,-.2],[-.1,.4]])
    y = torch.tensor([1,1])
    config = {'alpha':.5,'beta':1.}
    ref_buffer,port_buffer = FixedReplay(method),FixedReplay(method)
    optimizer = torch.optim.SGD(reference.parameters(),lr=.03)
    ref = SimpleNamespace(net=reference,opt=optimizer,loss=F.cross_entropy,
                          args=SimpleNamespace(alpha=.5,beta=1.,minibatch_size=2,task_free=True),
                          buffer=ref_buffer,transform=None,device='cpu',
                          current_task=1,seen_so_far=torch.tensor([0]))
    cls = 'Derpp' if method=='derpp' else 'ErACE'
    filename = 'derpp.py' if method=='derpp' else 'er_ace.py'
    official_observe = official_node(filename,cls,'observe')
    expected_loss = official_observe(ref,x,y,x.clone())
    opt = torch.optim.SGD(port.parameters(),lr=.03)
    opt.zero_grad()
    loss,outputs,seen,_ = objective(method,port,x,y,port_buffer,config,torch.tensor([0]))
    old_logits = outputs.detach().clone()
    loss.backward()
    opt.step()
    assert abs(loss.item()-expected_loss)<1e-7
    assert all(torch.allclose(a,b,atol=1e-7) for a,b in zip(reference.parameters(),port.parameters()))
    if method=='derpp':
        assert torch.equal(old_logits,ref_buffer.added['logits'])
        assert ref_buffer.index==port_buffer.index==2
    else:
        assert torch.equal(seen,ref.seen_so_far)
        assert ref_buffer.index==port_buffer.index==1


@pytest.mark.parametrize('method',['derpp','er_ace'])
def test_reverse_order_renaming_preserves_initial_predictions(method):
    forward,_,_ = make(method,19,list(range(5)))
    reverse,_,inverse = make(method,19,list(range(4,-1,-1)))
    x = torch.randn(4,784)
    assert torch.allclose(forward(x),reverse(x)[:,inverse],atol=1e-6)


def test_ace_suppresses_old_absent_class_gradient_but_keeps_unseen_class():
    logits = torch.tensor([[.3,-.2,.1,.4]],requires_grad=True)
    masked,seen = ace_mask(logits,torch.tensor([1]),torch.tensor([0]))
    F.cross_entropy(masked,torch.tensor([1])).backward()
    assert logits.grad[0,0]==0
    assert logits.grad[0,2]!=0 and logits.grad[0,3]!=0
    assert torch.equal(seen,torch.tensor([0,1]))


def test_logit_memory_and_reservoir_membership_are_auditable():
    dark = Replay(20,4,True)
    plain = Replay(20,4,False)
    assert dark.nbytes()-plain.nbytes()==20*10*4
    for stage in range(4):
        raw = torch.full((15,784),stage*50,dtype=torch.uint8)
        x = (raw.float()/255-.13062755)/.30810780
        y = torch.arange(15)%10
        if dark.used:
            dark.draw(10)
            dark.draw(10)
            plain.draw(10)
        dark.observe(x,y,torch.full((15,10),float(stage)))
        plain.observe(x,y,None)
    assert torch.equal(dark.images,plain.images) and torch.equal(dark.labels,plain.labels)
    assert dark.used==plain.used==20 and dark.seen==plain.seen==60
