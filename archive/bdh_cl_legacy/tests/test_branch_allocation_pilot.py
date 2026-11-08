import inspect

import numpy as np

from bdh_cl.scripts.branch_allocation_pilot import BranchMemory, assemblies


def test_fixed_storage_and_sparse_contact_budget():
    model=BranchMemory('competitive_stable',0)
    size=model.bytes()
    for x in assemblies(8,320,0,0):model.update(x)
    assert model.bytes()==size
    assert ((model.weights>0).sum(1)<=40).all()
    np.testing.assert_allclose(np.linalg.norm(model.weights,axis=1),1,atol=1e-6)


def test_competition_keeps_branches_available():
    x=assemblies(8,320,0,0)[0]
    independent=BranchMemory('independent',0,noise=0)
    competitive=BranchMemory('competitive',0,noise=0)
    for _ in range(30):
        independent.update(x);competitive.update(x)
    assert (independent.usage>0).sum()==12
    assert (competitive.usage>0).sum()==1
    assert competitive.scores(x).max()>.9


def test_generic_slots_are_identical_not_a_dendrite_specific_advantage():
    a=BranchMemory('competitive_stable',2)
    b=BranchMemory('generic_slots',2)
    for x in assemblies(8,320,.3,1):
        for _ in range(5):a.update(x);b.update(x)
    np.testing.assert_array_equal(a.weights,b.weights)
    assert list(inspect.signature(BranchMemory.update).parameters)==['self','x']


def test_background_turnover_is_reduced_for_established_branches():
    a=BranchMemory('competitive',0)
    b=BranchMemory('competitive_stable',0)
    p=assemblies(8,320,0,0)
    for _ in range(100):a.update(p[0]);b.update(p[0])
    aa=a.weights[0].copy();bb=b.weights[0].copy()
    for _ in range(100):a.update(p[1]);b.update(p[1])
    assert np.linalg.norm(b.weights[0]-bb)<np.linalg.norm(a.weights[0]-aa)
