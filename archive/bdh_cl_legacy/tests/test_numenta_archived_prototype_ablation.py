import torch
from bdh_cl.scripts.matched_numenta_comparison import construct
from bdh_cl.scripts.numenta_archived_prototype_ablation import archive, archived_forward


def test_archives_reproduce_native_at_capture_and_frozen_gate_survives_dendrite_change():
    torch.set_num_threads(2)
    m=construct('numenta',7);m.eval()
    x=torch.randn(8,784);contexts=torch.randn(5,784)
    pairs=[archive(m,c) for c in contexts]
    indices=torch.stack([p[0] for p in pairs]);mods=torch.stack([p[1] for p in pairs])
    ids=torch.arange(8)%5
    expected=archived_forward(m,x,contexts,ids,indices,mods,'native')
    for mode in ['archived_indices','archived_modulation']:
        assert torch.allclose(expected,archived_forward(m,x,contexts,ids,indices,mods,mode),atol=1e-6)
    frozen_before=archived_forward(m,x,contexts,ids,indices,mods,'archived_modulation').detach().clone()
    with torch.no_grad():
        for l in m._layers:l.segments.weights.mul_(.1)
    assert torch.equal(frozen_before,archived_forward(m,x,contexts,ids,indices,mods,'archived_modulation'))
    assert not torch.equal(expected,archived_forward(m,x,contexts,ids,indices,mods,'archived_indices'))
