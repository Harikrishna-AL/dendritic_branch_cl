import torch
from bdh_cl.scripts.branch_short_memory_pilot import Model,stream,evaluate


def test_memory_read_is_causal_and_commit_decays_without_targets():
    torch.manual_seed(0);model=Model('branch_memory',streams=2)
    x=torch.randn(2,16);before=model.memory.clone()
    pred,a,_=model.predict(x)
    assert torch.equal(before,model.memory)
    model.commit(a)
    assert torch.allclose(model.memory,.05*torch.tanh(a),atol=1e-7)
    previous=model.memory.clone();model.commit(torch.zeros_like(a))
    assert torch.allclose(model.memory,.95*previous)
    assert not model.memory.requires_grad


def test_neuron_control_shares_state_and_competition_memory_changes_winner():
    model=Model('neuron_memory_competition',streams=2)
    model.commit(torch.rand(2,32,4))
    assert torch.equal(model.memory[:,:,0],model.memory[:,:,3])
    model=Model('branch_memory_competition',streams=2)
    with torch.no_grad():model.weight.zero_();model.bias.fill_(.1);model.memory[:,:,2]=1
    _,_,winner=model.predict(torch.zeros(2,16))
    assert (winner==2).all()
    _,_,erased=model.predict(torch.zeros(2,16),True)
    assert (erased==0).all()


def test_stream_is_continuous_and_evaluation_restores_state():
    model=Model('branch_memory',streams=2)
    data=list(stream(3,4,2,2,order=(0,1)))
    assert all(x.shape==(2,16) and y.shape==(2,) for x,y,_ in data)
    # Adjacent histories overlap across the regime switch, rather than resetting.
    assert torch.equal(data[1][0].reshape(2,8,2)[:,1:],data[2][0].reshape(2,8,2)[:,:-1])
    model.memory.fill_(.3);saved=model.memory.clone();params={n:p.clone() for n,p in model.named_parameters()}
    evaluate(model,5,64)
    assert torch.equal(saved,model.memory)
    assert all(torch.equal(params[n],p) for n,p in model.named_parameters())
