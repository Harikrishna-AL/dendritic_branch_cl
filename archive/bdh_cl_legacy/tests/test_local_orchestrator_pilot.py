import torch
from bdh_cl.scripts.local_orchestrator_pilot import Layer, Network


def test_winner_influence_is_local_and_winner_is_raw_activation():
    layer = Layer(3, neurons=4, branches=4, context=2)
    with torch.no_grad():
        layer.input.weight.zero_()
        layer.input.bias.fill_(.1)
        layer.input.bias[5] = 2
        layer.context.weight.zero_()
        layer.context.bias.zero_()
        layer.keys.fill_(1)
    x = torch.zeros(2, 3)
    _, base, _ = layer(x, 'context', 2)
    _, routed, winner = layer(x, 'winner', 2)
    assert winner.tolist() == [5, 5]
    affected = torch.where((routed-base)[0].abs()>1e-7)[0].tolist()
    assert affected == [3, 4, 5, 6, 7]


def test_all_routing_arms_support_finite_encoder_gradients():
    for arm in ('context', 'local', 'winner', 'fixed'):
        model = Network(arm)
        model(torch.rand(4,196)).square().mean().backward()
        for p in model.layers.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_zero_gain_equals_context_and_diagnostic_restores_gain():
    from bdh_cl.scripts.local_orchestrator_diagnostic import diagnose
    torch.manual_seed(3)
    model = Network('winner', gain=4)
    x = torch.rand(8,196)
    model.gain = 0
    off = model(x)
    model.arm = 'context'
    assert torch.equal(off, model(x))
    model.arm = 'winner'
    model.gain = 4
    before = model(x).detach().clone()
    result = diagnose(model, x, torch.zeros(8, dtype=torch.long), [0,1])
    assert model.gain == 4
    assert torch.equal(before, model(x))
    assert all(d['gate_change_mean'] >= 0 for d in result['layers'])
