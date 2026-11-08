import inspect

import torch

from bdh_cl.scripts.encoder_drift_pilot import Encoder, Learner, cka, probes


def test_parameter_matching_and_latent_normalization():
    encoders=[Encoder(k) for k in ('mlp','sum','max')]
    counts=[sum(p.numel() for p in e.parameters()) for e in encoders]
    assert max(counts)-min(counts)==5
    x=torch.rand(8,196)
    for encoder in encoders:
        z=encoder(x)
        assert z.shape==(8,64)
        torch.testing.assert_close(z.norm(dim=-1),torch.full((8,),3.))


def test_frozen_encoder_remains_fixed_but_moving_encoder_updates():
    x,y=torch.rand(8,196),torch.arange(8)
    for frozen in (True,False):
        learner=Learner('max',frozen,True,0)
        old=[p.detach().clone() for p in learner.encoder.parameters()]
        teacher_free_optimizer=learner.optimizer
        features=learner.encoder(x).detach().half()
        learner.update(x,y)
        assert learner.optimizer is teacher_free_optimizer
        assert any(not torch.equal(p,q) for p,q in zip(old,learner.encoder.parameters())) == (not frozen)
        torch.testing.assert_close(learner.memory.z[:8],features)
        assert not learner.memory.z.requires_grad


def test_update_interface_and_balanced_evaluator_probes():
    assert list(inspect.signature(Learner.update).parameters)==['self','x','y']
    x=torch.rand(200,196);y=torch.cat((torch.zeros(100),torch.ones(100))).long()
    xx,yy=probes((x,y))
    assert xx.shape[0]==64 and int((yy==0).sum())==32 and int((yy==1).sum())==32


def test_cka_is_one_for_identical_nonconstant_features():
    x=torch.randn(64,10)
    assert abs(cka(x,x)-1)<1e-6
