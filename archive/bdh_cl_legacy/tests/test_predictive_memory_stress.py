import torch
from bdh_cl.scripts.predictive_memory_stress import stress_stream,CONDITIONS,evaluate
from bdh_cl.scripts.branch_short_memory_pilot import stream
from bdh_cl.scripts.shared_context_memory_pilot import Model

def test_reference_matches_existing_generator():
    for (x,y,_),(a,b,_,_) in zip(stream(13,128,64,order=(0,1)),stress_stream(13,64,.8,0,order=(0,1))):
        torch.testing.assert_close(x,a);torch.testing.assert_close(y,b)

def test_causal_prefix_and_gradual_targets():
    short=list(stress_stream(9,64,.8,32,order=(0,1)))
    long=list(stress_stream(9,64,.8,32,order=(0,1,0)))
    for (x,y,o,p),(a,b,c,q) in zip(short,long):
        torch.testing.assert_close(x,a);torch.testing.assert_close(y,b);torch.testing.assert_close(o,c);assert p==q
    assert all(torch.isfinite(v).all() for row in short for v in row[:3])

def test_all_observations_scored_and_eval_weights_unchanged():
    torch.manual_seed(0);m=Model('linear');weights=[p.clone() for p in m.parameters()]
    r=evaluate(m,5,'short',1);assert len(r['phases'])==8
    assert sum(len(p['binned_mse']) for p in r['phases'])==32
    assert r['phases'][0]['task']==1
    assert abs(r['all_mse']-sum(p['all_mse'] for p in r['phases'])/8)<1e-7
    for a,b in zip(weights,m.parameters()):torch.testing.assert_close(a,b)
