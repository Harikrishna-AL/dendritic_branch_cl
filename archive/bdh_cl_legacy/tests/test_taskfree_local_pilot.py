import torch
from bdh_cl.scripts.taskfree_local_pilot import LocalLayer, Model, make_data, fit


def test_no_task_id_in_input_and_history_is_only_observation():
    data=make_data(10,64)
    assert all(x.shape==(64,16) and y.shape==(64,) for x,y in data)
    assert torch.equal(data[0][0],make_data(10,64)[0][0])
    assert not torch.equal(data[0][0],make_data(11,64)[0][0])


def test_local_recruitment_has_bounded_reach_and_fixed_budget():
    layer=LocalLayer(16)
    with torch.no_grad():
        layer.weight.zero_();layer.bias.fill_(.1);layer.bias[10,2]=4
    _,mask,w=layer(torch.zeros(2,16),'leader_local')
    assert w.tolist()==[42,42]
    assert torch.allclose(mask.sum(-1),torch.full((2,),16.))
    selected=torch.where(mask[0]>.5)[0]
    distance=torch.minimum((selected-10)%32,(10-selected)%32)
    assert (distance<=8).all()


def test_branch_compartment_graph_has_bounded_reach():
    layer=LocalLayer(16,branch_graph=True)
    with torch.no_grad():
        layer.weight.zero_();layer.bias.fill_(.1);layer.bias[10,2]=4
    _,mask,w=layer(torch.zeros(2,16),'leader_local')
    assert w.tolist()==[42,42]
    assert torch.allclose(mask.sum(-1),torch.full((2,),16.))
    selected=torch.where(mask[0]>.5)[0]
    assert (torch.minimum((selected-10)%32,(10-selected)%32)<=8).all()


def test_fixed_readout_and_finite_leader_gradients():
    torch.set_num_threads(2);torch.manual_seed(0)
    model=Model('leader_local');x,y=make_data(4,64)[0]
    original=model.readout.clone()
    (model(x)-y).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    fit(model,x,y,1,0)
    assert torch.equal(original,model.readout)
