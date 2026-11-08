import torch
from bdh_cl.scripts.context_circuit_pilot import CircuitNet, fit, make_data


def test_separated_circuits_are_disjoint_and_keep_old_prediction_exactly():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=CircuitNet('separated');data=make_data(1,64)
    _,a=model.encode(data[0][0]);_,b=model.encode(data[1][0])
    for (ma,_),(mb,_) in zip(a,b):
        assert torch.equal(ma.sum(-1),torch.full((64,),16.))
        assert (ma*mb).sum()==0
    fit(model,*data[0],2,0)
    old=model(data[0][0]).detach().clone();readout=model.readout.clone()
    fit(model,*data[1],2,1)
    assert torch.equal(old,model(data[0][0]))
    assert torch.equal(readout,model.readout)


def test_learned_routes_select_half_and_receive_context_gradient():
    torch.manual_seed(0)
    model=CircuitNet('context_leader');x,y=make_data(2,64)[0]
    _,states=model.encode(x)
    for mask,_ in states:
        assert torch.allclose(mask.sum(-1),torch.full((64,),16.))
        assert torch.allclose(mask,mask.round(),atol=1e-6)
    (model(x)-y).square().mean().backward()
    for layer in model.layers:
        g=layer.compatibility.grad
        assert g is not None and torch.isfinite(g).all() and g[:,:,0].abs().sum()>0
        assert g[:,:,1].abs().sum()==0
