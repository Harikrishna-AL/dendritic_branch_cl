import torch
from bdh_cl.models.vision_bdh import VisionBDH,BranchDistributions


def small(branches=0,mode='open'):
    return VisionBDH(width=16,neurons=16,heads=2,depth=2,dropout=0,branches=branches,groups=2,mode=mode)


def test_zero_insert_preserves_predictions_and_learns():
    torch.manual_seed(3);base=small();model=small(3)
    model.load_state_dict(base.state_dict(),strict=False)
    x=torch.randn(3,3,32,32)
    torch.testing.assert_close(base(x),model(x),rtol=0,atol=0)
    model(x).square().mean().backward()
    assert model.branch_gain.grad.abs().sum()>0


def test_gaussian_shapes_and_eval_immutability():
    torch.manual_seed(4);model=small(3,'gaussian');model.eval()
    x=torch.randn(4,3,32,32)
    with torch.no_grad():model(x,calibrate=True)
    state={k:v.clone() for k,v in model.router.state_dict().items()}
    result=model(x)
    assert result.shape==(4,100)
    assert not model.router.pending
    for k,v in state.items():torch.testing.assert_close(v,model.router.state_dict()[k],rtol=0,atol=0)
    model.train();model.router.threshold.fill_(-1e9)
    model(x).square().mean().backward()
    assert len(model.router.pending)==2
    model.commit();assert not model.router.pending
    assert not torch.equal(state['fast_mean'],model.router.fast_mean)


def test_frozen_backbone_really_stays_fixed():
    torch.manual_seed(5);model=small(3);model.set_backbone_trainable(False)
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(2):
        opt.zero_grad();model(torch.randn(3,3,32,32)).square().mean().backward();opt.step()
    for n,p in model.named_parameters():
        if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    assert model.branch_gain.abs().sum()>0


def test_rejection_excludes_updates():
    r=BranchDistributions(1,2,2,3,16)
    c=torch.randn(3,2,65,16);r.initialize(c,0);r.train();r.threshold.fill_(1e9)
    before=r.slow_mean.clone();g=r(c,0);r.commit()
    assert g.sum()==0
    torch.testing.assert_close(before,r.slow_mean,rtol=0,atol=0)
