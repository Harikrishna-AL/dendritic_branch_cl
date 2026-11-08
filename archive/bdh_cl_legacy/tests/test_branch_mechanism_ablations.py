import copy
import torch
from bdh_cl.scripts.branch_mechanism_ablations import Mechanism,DenseCredit,GenericExperts

def test_central_reexpression_is_exact_in_training_and_feedback():
    torch.manual_seed(7);a=Mechanism('full');a.reset()
    torch.manual_seed(7);b=Mechanism('central_exact');b.reset()
    oa=torch.optim.Adam(a.parameters(),lr=.003);ob=torch.optim.Adam(b.parameters(),lr=.003)
    for _ in range(40):
        x=torch.randn(16,16);y=torch.randn(16)
        with torch.no_grad():pa,ta=a.forecast(x);pb,tb=b.forecast(x)
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        a.observe(ta,y);b.observe(tb,y)
        for model,opt,t in ((a,oa,ta),(b,ob,tb)):
            opt.zero_grad();model.training_loss(t,y).backward();opt.step()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(a.error,b.error_state(),rtol=0,atol=0)

def test_dense_credit_keeps_forward_and_changes_inactive_updates():
    q=torch.randn(2,8,4,requires_grad=True);w=torch.zeros(2,8,dtype=torch.long)
    pred=DenseCredit.apply(q,w);torch.testing.assert_close(pred,torch.tanh(q[:,:,0]),rtol=0,atol=0)
    pred.sum().backward();assert (q.grad[:,:,1:].abs()>0).all()
    expected=(1-torch.tanh(q.detach()).square())/4;torch.testing.assert_close(q.grad,expected)

def test_generic_experts_have_matched_slow_budget_and_read_only_tickets():
    for d,budget in ((2,384),(25,1120)):
        m=GenericExperts('moe_sparse',d);m.reset();assert .95*budget<=sum(p.numel() for p in m.parameters())<=budget
        with torch.no_grad():_,ticket=m.forecast(torch.randn(16,16 if d==2 else d))
        state=copy.deepcopy(m.state_dict());m.training_loss(ticket,torch.randn(16)).backward()
        for k,v in state.items():torch.testing.assert_close(v,m.state_dict()[k])
