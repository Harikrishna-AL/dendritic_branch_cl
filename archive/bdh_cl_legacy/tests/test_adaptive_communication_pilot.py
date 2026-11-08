import torch
from bdh_cl.scripts.adaptive_communication_pilot import Model,run,adaptive,make_data


def test_eight_rounds_reproduce_checkpoint_model_and_weights_stay_fixed():
    torch.set_num_threads(2);torch.manual_seed(0)
    model=Model('learned');x,_=make_data(5,16)[0]
    before={n:p.detach().clone() for n,p in model.state_dict().items()}
    prediction,_=run(model,x,8)
    assert torch.equal(prediction,model(x))
    run(model,x,32);adaptive(model,x,.1)
    assert all(torch.equal(before[n],p) for n,p in model.state_dict().items())


def test_adaptive_uses_inputs_only_and_matches_selected_fixed_horizons():
    torch.manual_seed(0);model=Model('learned');x,_=make_data(6,16)[0]
    base,_=run(model,x,8);long,_=run(model,x,32)
    off,_,trigger=adaptive(model,x,float('inf'))
    assert not trigger.any() and torch.equal(off,base)
    on,_,trigger=adaptive(model,x,float('-inf'))
    assert trigger.all() and torch.equal(on,long)
    selfonly,_=run(model,x,8,False)
    model.disabled=True
    assert torch.equal(selfonly,model(x))
