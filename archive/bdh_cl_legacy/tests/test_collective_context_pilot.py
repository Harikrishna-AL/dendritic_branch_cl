import torch
from bdh_cl.scripts.collective_context_pilot import Layer,Model
from bdh_cl.scripts.taskfree_local_pilot import make_data
from bdh_cl.scripts.signed_local_history_pilot import stream_batches,fit_stream


def test_exchange_stays_local_and_independent_has_no_exchange():
    layer=Layer(16)
    with torch.no_grad():
        for mix in (layer.self_mix,layer.left_mix,layer.right_mix):mix.weight.copy_(torch.eye(4))
    cue=torch.zeros(1,128,4);cue[:,40,0]=1
    for arm in ('average','learned'):
        out=layer.communicate(cue,arm)
        affected=torch.where(out[0].abs().sum(-1)>1e-8)[0]
        assert (affected>=32).all() and (affected<=48).all()
        assert out[0,39].abs().sum()>0
    out=layer.communicate(cue,'independent')
    assert torch.where(out[0].abs().sum(-1)>1e-8)[0].tolist()==[40]
    assert torch.equal(out,layer.communicate(cue,'learned',True))


def test_matched_parameters_finite_gradients_and_fixed_readout():
    torch.set_num_threads(2)
    counts=[]
    for arm in ('independent','average','learned'):
        torch.manual_seed(0);model=Model(arm);data=make_data(3,64)
        counts.append(sum(p.numel() for p in model.parameters()))
        readout=model.readout.clone()
        fit_stream(model,stream_batches(data,1,0,[0,1]),[],lambda m:None)
        assert torch.equal(model.readout,readout)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(set(counts))==1
