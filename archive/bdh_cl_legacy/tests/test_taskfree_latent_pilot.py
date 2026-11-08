import inspect

import torch

from bdh_cl.scripts.taskfree_latent_pilot import LatentMemory, Learner


def test_reservoir_is_bounded_and_contains_only_observed_latents():
    memory = LatentMemory('reservoir', width=4, capacity=5)
    x = torch.arange(80).reshape(20,4).float()
    y = torch.arange(20)
    memory.observe(x,y)
    assert memory.size == 5 and memory.seen == 20
    assert memory.bytes() == 5*(4*2+8)
    for z,label in zip(memory.z,memory.y):
        torch.testing.assert_close(z.float(),x[label])
    assert memory.z.dtype == torch.float16


def test_centroids_are_streaming_means_without_stored_examples():
    memory=LatentMemory('centroids',width=2)
    memory.observe(torch.tensor([[1.,2.],[3.,4.],[9.,8.]]),torch.tensor([0,0,1]))
    torch.testing.assert_close(memory.mean[0],torch.tensor([2.,3.]))
    assert memory.z is None and memory.y is None
    sampled=memory.sample(32)
    assert set(sampled[1].tolist()) <= {0,1}
    assert memory.bytes() == 10*2*4+10*8


def test_update_has_no_task_boundary_argument_or_optimizer_reset():
    assert list(inspect.signature(Learner.update).parameters) == ['self','z','y']
    learner=Learner('max','reservoir',False,0,capacity=5)
    optimizer=learner.optimizer
    learner.update(torch.randn(4,64),torch.tensor([0,1,0,1]))
    assert learner.replayed == 0
    learner.update(torch.randn(4,64),torch.tensor([2,3,2,3]))
    assert learner.replayed == 16 and learner.updates == 2
    assert learner.optimizer is optimizer and learner.memory.size == 5


def test_ema_teacher_updates_without_a_task_snapshot():
    learner=Learner('max','none',True,0)
    original=[p.clone() for p in learner.teacher.parameters()]
    learner.update(torch.randn(8,64),torch.arange(8))
    for old,teacher,student in zip(original,learner.teacher.parameters(),learner.model.parameters()):
        torch.testing.assert_close(teacher,.99*old+.01*student)
        assert not teacher.requires_grad
