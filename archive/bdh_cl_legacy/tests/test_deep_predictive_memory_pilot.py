import torch
from bdh_cl.scripts.deep_predictive_memory_pilot import Model


def test_central_and_branch_budgets_and_predictions_match():
    torch.manual_seed(0);branch=Model('branch_memory',streams=2)
    torch.manual_seed(0);central=Model('central_memory',streams=2)
    assert sum(p.numel() for p in branch.parameters())==384
    assert sum(b.numel()*b.element_size() for b in branch.buffers())==sum(b.numel()*b.element_size() for b in central.buffers())
    for _ in range(16):
        x=torch.randn(2,16);a,qa,wa=branch.predict(x);b,qb,wb=central.predict(x)
        assert torch.equal(a,b)
        assert all(torch.equal(i,j) for i,j in zip(wa,wb))
        branch.commit(qa);central.commit(qb)


def test_memory_is_causal_and_both_layers_receive_finite_gradients():
    torch.manual_seed(0);model=Model(streams=2);x=torch.randn(2,16)
    old=[m.previous.clone() for m in model.memories]
    pred,q,_=model.predict(x)
    assert all(torch.equal(o,m.previous) for o,m in zip(old,model.memories))
    model.commit(q)
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not m.cross.requires_grad for m in model.memories)


def test_erasure_clears_both_layers():
    model=Model(streams=2)
    for _ in range(3):_,q,_=model.predict(torch.randn(2,16));model.commit(q)
    model.commit(q,True)
    assert all(m.previous.count_nonzero()==0 and m.cross.count_nonzero()==0 for m in model.memories)
