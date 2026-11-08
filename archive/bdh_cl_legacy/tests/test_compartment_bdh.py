import torch
from bdh_cl.config import BDHSpec
from bdh_cl.models.compartment_bdh import CompartmentBDH


def spec():
    return BDHSpec(n_layer=2,n_embd=16,n_head=2,mlp_internal_dim_multiplier=4,dropout=0.,
                   neuron_unit='dendritic',neuron_unit_kwargs={'n_branches':4,'soma':'sum'})


def test_shared_and_uniform_are_same_memory_with_different_views():
    torch.manual_seed(3)
    a,b=CompartmentBDH(spec(),'shared'),CompartmentBDH(spec(),'uniform')
    b.load_state_dict(a.state_dict())
    x=torch.randint(256,(3,8))
    assert a.memory_bytes()==b.memory_bytes()
    for _ in range(3):
        la=a(x,update_memory=True)[0]
        lb=b(x,update_memory=True)[0]
        torch.testing.assert_close(la,lb,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a.slow_state,b.slow_state,atol=1e-7,rtol=1e-5)


def test_current_write_does_not_change_current_prediction_and_eval_does_not_write():
    a=CompartmentBDH(spec(),'selective')
    b=CompartmentBDH(spec(),'selective')
    b.load_state_dict(a.state_dict())
    x=torch.randint(256,(2,7))
    before=a.slow_state.clone()
    expected=b(x)[0]
    logits=a(x,update_memory=True)[0]
    torch.testing.assert_close(logits,expected)
    assert not torch.equal(a.slow_state,before)
    saved=a.slow_state.clone()
    a.eval();a(x)
    torch.testing.assert_close(a.slow_state,saved)
    logits.square().mean().backward()
    assert a.neuron_unit_x.weight.grad is not None


def test_all_memory_arms_have_identical_parameter_and_state_budgets():
    models=[CompartmentBDH(spec(),a) for a in ['none','shared','uniform','selective']]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    assert models[0].memory_bytes()==0
    assert len({m.memory_bytes() for m in models[1:]})==1


def test_no_memory_arm_matches_native_dendritic_bdh():
    from bdh_cl.models.bdh_wrapper import BDHWrapper
    a=CompartmentBDH(spec(),'none')
    b=BDHWrapper(spec())
    b.load_state_dict({k:v for k,v in a.state_dict().items() if k not in ('compression','router')})
    x=torch.randint(256,(2,8))
    torch.testing.assert_close(a(x)[0],b(x)[0],atol=0,rtol=0)


def test_domain_shift_preserves_labels_and_applies_same_transform_to_both_splits():
    import numpy as np
    from bdh_cl.data.mnist import MNISTPool
    from bdh_cl.scripts.compartment_recurrence_domain import permuted_environment
    images=np.arange(2*7*7,dtype=np.uint8).reshape(2,7,7)
    labels=np.array([0,1])
    pool=MNISTPool('A',(0,1),'d',7,images,labels,images.copy(),labels.copy())
    shifted=permuted_environment(pool)
    np.testing.assert_array_equal(shifted.train_labels,labels)
    np.testing.assert_array_equal(shifted.train_images,shifted.test_images)
    np.testing.assert_array_equal(np.sort(shifted.train_images.reshape(2,-1),axis=1),np.sort(images.reshape(2,-1),axis=1))
    assert not np.array_equal(shifted.train_images,images)
