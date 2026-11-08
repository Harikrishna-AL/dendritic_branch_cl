import torch
from bdh_cl.scripts.signed_local_history_pilot import Layer,Model,fit_stream,stream_batches
from bdh_cl.scripts.taskfree_local_pilot import make_data


def test_signed_messages_excite_and_inhibit_with_local_reach():
    layer=Layer(16)
    with torch.no_grad():
        layer.input.weight.zero_();layer.input.bias.fill_(.1);layer.input.bias[40]=2
        layer.keys.zero_();layer.keys[:,0]=-1;layer.keys[40,0]=1
    _,state=layer(torch.zeros(1,16),'leader_signed')
    assert state['winner'].item()==40
    gate=state['gate'][0]
    assert gate[40]>.5
    assert gate[39]<.5 and gate[41]<.5
    assert gate[0]==.5 and gate[100]==.5
    _,off=layer(torch.zeros(1,16),'leader_signed',True)
    assert torch.equal(off['gate'],torch.full((1,128),.5))


def test_suppressed_branch_can_be_leader_and_inhibit():
    layer=Layer(16)
    with torch.no_grad():
        layer.input.weight.zero_();layer.input.bias.fill_(.1);layer.input.bias[40]=-2
        layer.keys.fill_(1)
    _,state=layer(torch.zeros(1,16),'leader_signed')
    assert state['winner'].item()==40
    assert state['gate'][0,39]<.5


def test_stream_passes_only_observations_targets_and_keeps_readout_fixed():
    torch.set_num_threads(2)
    model=Model('leader_signed');readout=model.readout.clone();data=make_data(7,64)
    batches=list(stream_batches(data,1,0,[0,1]))
    assert len(batches)==2 and all(len(b)==2 and b[0].shape==(64,16) for b in batches)
    states,metrics=fit_stream(model,iter(batches),[1,2],lambda m:float(m(data[0][0]).mean().detach()))
    assert len(states)==2 and len(metrics)==2
    assert torch.equal(readout,model.readout)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
