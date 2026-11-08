import torch
from bdh_cl.scripts.numenta_context_trace_ablation import make


def test_native_forward_matches_upstream():
    torch.set_num_threads(2)
    m=make('native_max',7);m.eval()
    x=torch.randn(4,784);c=torch.randn(4,784)
    assert torch.equal(m(x,c,False),m.net(x,c))


def test_counterfactuals_match_full_interventions_and_receive_gradients():
    torch.set_num_threads(2)
    m=make('trace_aux',7);m.eval()
    x=torch.randn(2,784);c=torch.randn(2,784)
    alts=m.alternatives(x,c)
    assert alts.shape==(2,2,64,5,10)
    for target,n,branch in [(0,7,3),(1,11,4)]:
        h=x
        for i,l in enumerate(m.net._layers):
            y=l.module(h);d=l.segments(c);ids=m.selected(d,i).clone()
            if i==target:ids[:,n]=branch
            h=m.activate(y*d.gather(2,ids[...,None]).squeeze(-1).sigmoid(),i,False)
        expected=m.net._output_layers[0](h)
        assert torch.allclose(alts[:,target,n,branch],expected,atol=1e-6)
    alts.square().mean().backward()
    assert m.net._layers[0].segments.weights.grad.abs().sum()>0
    assert m.net._layers[1].segments.weights.grad.abs().sum()>0


def test_context_enters_network_without_archive_or_error_updates():
    m=make('trace_ce',7);m.eval()
    x=torch.randn(4,784);a=torch.randn(4,784);b=torch.randn(4,784)
    before=m.error.clone()
    assert not torch.equal(m(x,a,False),m(x,b,False))
    assert torch.equal(before,m.error)
