import torch
from bdh_cl.scripts.orchestrator_signal_pilot import SignalNet, make_data, train, evaluate


def test_context_changes_target_without_changing_observed_signals():
    data=make_data(7,64)
    assert torch.equal(data[0][0][:,:8],data[1][0][:,:8])
    assert torch.equal(data[0][0][:,-2:],data[1][0][:,-2:].flip(1))
    assert not torch.equal(data[0][1],data[1][1])
    assert torch.equal(data[0][0],make_data(7,64)[0][0])
    assert not torch.equal(data[0][0],make_data(8,64)[0][0])


def test_training_keeps_readout_fixed_and_intervention_restores_gain():
    torch.set_num_threads(2)
    model=SignalNet('winner',4)
    readout=model.readout.clone()
    decoder=model.decoder.weight.detach().clone()
    data=make_data(9,64)
    train(model,*data[0],2,0)
    assert torch.equal(model.readout,readout)
    assert 'readout' not in dict(model.named_parameters())
    assert torch.equal(model.decoder.weight,decoder)
    before=model(data[0][0]).detach().clone()
    result=evaluate(model,data)
    assert model.gain==4
    assert torch.equal(model(data[0][0]),before)
    assert all(torch.isfinite(torch.tensor(r['mse'])) for r in result)
