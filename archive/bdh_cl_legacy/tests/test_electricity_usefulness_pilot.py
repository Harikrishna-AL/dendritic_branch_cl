import copy
import numpy as np
import torch
from bdh_cl.scripts.electricity_usefulness_pilot import Useful,run

def test_feedback_uses_saved_predictions_and_does_not_move_weights():
    m=Useful('quality_all',2);m.reset()
    with torch.no_grad():_,ticket=m.forecast(torch.randn(2,25))
    target=torch.tensor([1.,-1.]);before=[p.clone() for p in m.parameters()]
    expected=.95*torch.ones_like(m.error)+.05*(ticket[3]-target[:,None]).square()
    with torch.no_grad():m.second.weight.add_(2)
    changed=[p.clone() for p in m.parameters()];m.observe(ticket,target)
    torch.testing.assert_close(m.error,expected)
    for a,b in zip(changed,m.parameters()):torch.testing.assert_close(a,b)

def test_feedback_waits24h_and_unseen_future_cannot_change_state():
    torch.manual_seed(4);m=Useful('quality_winner',2);m.reset();z=copy.deepcopy(m)
    y=torch.randn(300,2);future=y.clone();future[200:]+=100
    def data(x):return {'y':x,'year':np.full(300,2013),'month':np.ones(300,dtype=int)}
    run(m,None,data(y),144,168,False,False);torch.testing.assert_close(m.error,torch.ones_like(m.error))
    m=copy.deepcopy(z);run(m,None,data(y),144,200,False,False);run(z,None,data(future),144,200,False,False)
    for a,b in zip(m.buffers(),z.buffers()):torch.testing.assert_close(a,b)

def test_dense_training_control_keeps_identical_weights():
    torch.manual_seed(5);m=Useful('quality_all',2);m.reset()
    torch.manual_seed(5);c=Useful('predict_all',2);c.reset()
    y=torch.randn(220,2);d={'y':y,'year':np.full(220,2013),'month':np.ones(220,dtype=int)}
    for model in (m,c):run(model,torch.optim.Adam(model.parameters(),lr=.001),d,144,210,True,False)
    for a,b in zip(m.parameters(),c.parameters()):torch.testing.assert_close(a,b)
