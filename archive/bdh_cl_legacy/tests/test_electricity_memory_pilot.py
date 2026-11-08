import copy
import numpy as np
import torch
from bdh_cl.scripts.electricity_memory_pilot import Forecast,features,run

def test_observed_features_and_weekly_lag():
    y=torch.arange(300.).reshape(-1,1);x=features(y,200)
    torch.testing.assert_close(x[0,:24],y[177:201,0]);assert x[0,-1]==y[(200+24)-168,0]

def test_labels_are_delayed_and_future_values_cannot_change_weights():
    torch.manual_seed(9);a=Forecast('branch_linear',2);a.reset();b=copy.deepcopy(a)
    y=torch.randn(300,2);z=y.clone();z[200:]+=100
    year=np.full(300,2014);month=np.ones(300,dtype=int)
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    _,steps=run(a,oa,y,year,month,144,168,True,False);assert steps==0
    # Restart identical models for the full-prefix comparison.
    a=copy.deepcopy(b);oa=torch.optim.Adam(a.parameters(),lr=.001)
    _,sa=run(a,oa,y,year,month,144,200,True,False)
    _,sb=run(b,ob,z,year,month,144,200,True,False)
    assert sa==sb==4
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)

def test_recomputed_training_ticket_does_not_write_memory():
    for arm in ('branch_tanh','branch_linear','shared','history','gru'):
        m=Forecast(arm,2);m.reset();x=torch.randn(2,25)
        with torch.no_grad():_,ticket=m.forecast(x)
        state=[b.clone() for b in m.buffers()];history=copy.deepcopy(m.recent)
        m.ticket_prediction(ticket).square().mean().backward()
        for before,after in zip(state,m.buffers()):torch.testing.assert_close(before,after)
        for before,after in zip(history,m.recent):torch.testing.assert_close(before,after)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())
