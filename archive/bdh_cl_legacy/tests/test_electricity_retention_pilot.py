import copy
import numpy as np
import torch
from bdh_cl.scripts.electricity_retention_pilot import audit
from bdh_cl.scripts.electricity_usefulness_pilot import Useful,run

def test_audits_do_not_change_weights_optimizer_or_live_state():
    torch.manual_seed(7);m=Useful('quality_soft',2);m.reset()
    d={'y':torch.randn(1200,2),'year':np.full(1200,2012),'month':np.ones(1200,dtype=int)}
    before=copy.deepcopy(m.state_dict());r=audit(m,d,720,1100)
    assert r['hours']==356 and all(np.isfinite(v) for v in (r['mse'],r['after_week_mse']))
    for key,value in before.items():torch.testing.assert_close(value,m.state_dict()[key])
    a=audit(m,d,720,1100,error_feedback=False);assert np.isfinite(a['mse'])

def test_heldout_future_targets_cannot_change_training_checkpoint():
    torch.manual_seed(9);a=Useful('quality_soft',2);a.reset();b=copy.deepcopy(a)
    y=torch.randn(1200,2);other=y.clone();other[900:]+=100
    for model,values in ((a,y),(b,other)):
        d={'y':values,'year':np.full(1200,2012),'month':np.ones(1200,dtype=int)}
        run(model,torch.optim.Adam(model.parameters(),lr=.001),d,720,900,True,False)
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key])
