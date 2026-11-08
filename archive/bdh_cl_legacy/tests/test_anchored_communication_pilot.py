import torch
from bdh_cl.scripts.anchored_communication_pilot import Layer,Model,fit,policy,make_data


def test_evidence_is_retained_at_long_horizons_and_exchange_is_local():
    layer=Layer(16)
    with torch.no_grad():
        layer.context.weight.zero_();layer.context.bias.zero_();layer.context.bias[40*4]=1
        layer.evidence.weight.copy_(torch.eye(4))
        layer.self_mix.weight.zero_();layer.neighbor_mix.weight.zero_()
    _,state=layer(torch.zeros(1,16),64,False)
    assert state['context'][0,40,0]>.5
    with torch.no_grad():layer.neighbor_mix.weight.copy_(torch.eye(4))
    _,state=layer(torch.zeros(1,16),8,True)
    affected=torch.where(state['context'][0].abs().sum(-1)>1e-9)[0]
    assert (affected>=32).all() and (affected<=48).all()
    assert state['context'][0,39].abs().sum()>0


def test_variable_training_has_finite_gradients_and_inference_keeps_weights():
    torch.set_num_threads(2);torch.manual_seed(0);model=Model()
    x,y=make_data(5,32)[0];readout=model.readout.clone()
    fit(model,[(x,y)]*4,True,0,[2,4])
    assert torch.equal(readout,model.readout)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before={n:p.clone() for n,p in model.state_dict().items()}
    base=policy(model,x,'8',0)[0];off=policy(model,x,'adaptive',float('inf'))[0]
    assert torch.equal(base,off)
    long=policy(model,x,'32',0)[0];on=policy(model,x,'adaptive',float('-inf'))[0]
    assert torch.equal(long,on)
    assert all(torch.equal(before[n],p) for n,p in model.state_dict().items())
