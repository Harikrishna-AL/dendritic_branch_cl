import copy
import torch
from bdh_cl.scripts.usefulness_retention_synthetic import make,fit,audit
from bdh_cl.scripts.branch_short_memory_pilot import stream

def test_checkpoints_preserve_pending_feedback_and_audits_are_read_only():
    torch.manual_seed(4);m=make('quality_soft');m.reset();opt=torch.optim.Adam(m.parameters(),lr=.003)
    snap=fit(m,opt,9,(0,1),32,capture=True)
    assert snap['pending'] is not None and len(snap['batch'])==7
    before=copy.deepcopy(m.state_dict());p=audit(m,13,(0,1))
    assert all(torch.isfinite(torch.tensor(r['all_mse'])) for r in p)
    for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k])

def test_shared_and_branch_use_same_slow_initialization():
    torch.manual_seed(6);a=make('predict_winner');a.reset()
    torch.manual_seed(6);b=make('shared');b.reset()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)
    x=next(stream(14,1,1))[0]
    with torch.no_grad():
        pa,_=a.forecast(x);pb,_=b.forecast(x)
    torch.testing.assert_close(pa,pb)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==384
