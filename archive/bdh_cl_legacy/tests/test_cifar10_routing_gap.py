from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.scripts.cifar10_routing_gap import VisionBranch,VisionPEC,ContextRidge,normalize,state_hash


def test_cnn_counterfactuals_equal_direct_single_neuron_replacements():
    torch.set_num_threads(2);model=VisionBranch(4)
    x=torch.randn(3,3,32,32)
    logits,candidates=model(x,True)
    z=torch.tanh(model.branch.first(model.encoder(x))).reshape(-1,122,4)
    pooled=(z*model.branch.gates()).sum(2)
    assert logits.shape==(3,10) and candidates.shape==(3,122,4,10)
    for neuron,branch in [(0,0),(17,2),(121,3)]:
        replaced=pooled.clone();replaced[:,neuron]=z[:,neuron,branch]
        assert torch.allclose(candidates[:,neuron,branch],model.branch.head(replaced),atol=1e-6)


def test_archived_routes_equal_explicit_gate_restoration_and_preserve_state():
    model=VisionBranch(5);x=torch.randn(4,3,32,32)
    routes=torch.randint(4,(5,122));before=state_hash(model)
    outputs=model.route_logits(x,routes)
    assert outputs.shape==(4,5,10) and state_hash(model)==before
    for i in range(5):
        copied=deepcopy(model)
        copied.branch.error.fill_(1.)
        copied.branch.error.scatter_(1,routes[i,:,None],0.)
        assert torch.allclose(outputs[:,i],copied(x),atol=1e-6)


def test_online_context_statistics_match_batch_fit_without_examples():
    router=ContextRidge(8);x=torch.randn(20,3,32,32)
    router.observe(x[:10],2);router.observe(x[10:],3)
    phi=router.features(x);targets=F.one_hot(torch.tensor([2]*10+[3]*10),5).float()
    assert torch.allclose(router.gram,phi.T@phi,atol=1e-5)
    assert torch.allclose(router.cross,phi.T@targets,atol=1e-6)
    assert set(router.state())=={'projection','gram','cross'}
    expected=torch.linalg.solve(phi.T@phi+torch.eye(129),phi.T@targets)
    assert torch.allclose(router.solve(1.),expected,atol=2e-5)


def test_full_cnn_teacher_and_students_fit_declared_total_parameter_budget():
    branch=VisionBranch(3);pec=VisionPEC(3)
    a=sum(p.numel() for p in branch.parameters());b=sum(p.numel() for p in pec.parameters())
    assert a==525014 and b==522927 and abs(a-b)/a<.004
    assert all(not p.requires_grad for p in pec.teacher.parameters())
    x=torch.zeros(2,3,32,32,dtype=torch.uint8)
    assert normalize(x).shape==(2,3,32,32) and pec(normalize(x)).shape==(2,10)


def test_frozen_teacher_targets_are_independent_of_minibatch_partition():
    model=VisionPEC(6);x=torch.randn(8,3,32,32)
    with torch.no_grad():
        whole=model.teacher(x)
        partitioned=torch.cat([model.teacher(b) for b in x.split(2)])
    assert torch.allclose(whole,partitioned,atol=2e-6)
