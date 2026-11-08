import torch
from bdh_cl.scripts.predictability_memory_pilot import Model,stream,evaluate


def test_temporal_memory_causal_and_fading():
    model=Model('branch_predictability',streams=2)
    x=torch.ones(2,16);model.predict(x)
    assert model.cross.count_nonzero()==0
    model.commit(x)
    assert torch.allclose(model.cross,torch.full((2,2),.05))
    model.commit(torch.zeros_like(x))
    assert torch.allclose(model.cross,torch.full((2,2),.0475))
    assert not model.cross.requires_grad


def test_central_state_matched_control_is_identical_and_routing_needs_no_id():
    a=Model('branch_predictability',streams=2);b=Model('central_matched_memory',streams=2)
    with torch.no_grad():a.slope.fill_(.9);b.slope.fill_(.9)
    for x,_,_ in stream(7,64,32,2,(0,1)):
        pa,ma=a.predict(x);pb,mb=b.predict(x)
        assert torch.equal(pa,pb) and torch.equal(ma,mb)
        a.commit(x);b.commit(x)
    assert torch.equal(a.cross,b.cross)


def test_frozen_evaluation_keeps_learned_weights():
    model=Model('branch_predictability')
    before={n:p.clone() for n,p in model.named_parameters()}
    evaluate(model,9,block=64)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
