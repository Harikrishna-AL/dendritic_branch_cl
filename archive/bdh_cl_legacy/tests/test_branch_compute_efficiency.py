import torch
from bdh_cl.scripts.branch_compute_efficiency import cached_step,original_step,construct


def test_cached_and_chunked_match_feedback_loss_and_gradients():
    torch.set_num_threads(2)
    x=torch.randn(128,784);y=torch.arange(128)%10
    baseline=construct('branch_ce',7)
    loss,feedback=original_step(baseline,x,y);loss.backward()
    for variant in ['cached_exact','chunked_exact']:
        m=construct('branch_ce',7)
        actual,e=cached_step(m,x,y,0,variant);actual.backward()
        assert torch.equal(loss,actual)
        assert torch.allclose(feedback,e,atol=1e-6)
        for a,b in zip(baseline.parameters(),m.parameters()):assert torch.equal(a.grad,b.grad)


def test_periodic_feedback_only_on_scheduled_batches():
    m=construct('branch_ce',7);x=torch.randn(32,784);y=torch.arange(32)%10
    for step in [0,1,4]:
        _,e=cached_step(m,x,y,step,'feedback16_every4')
        assert (e is not None)==(step%4==0)
