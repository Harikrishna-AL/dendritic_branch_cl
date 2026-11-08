import torch
from torch import nn
from bdh_cl.scripts.mnist_sparse_reproducibility import CompactProjection,SparseBranches


def test_unconnected_input_does_not_change_branch_response():
    torch.manual_seed(4)
    projection=CompactProjection(nn.Linear(20,6),5,101)
    x=torch.randn(3,20);changed=x.clone()
    connected=projection.indices[0].tolist();unused=next(i for i in range(20) if i not in connected)
    changed[:,unused]+=100
    assert torch.equal(projection(x)[:,0],projection(changed)[:,0])
    projection(x).square().sum().backward()
    assert projection.weight.grad.shape==(6,5)
    assert torch.isfinite(projection.weight.grad).all()


def test_dense_variant_reproduces_original_initialization():
    from bdh_cl.scripts.mnist_quadratic_branches import InteractionBranches
    torch.manual_seed(101);base=InteractionBranches('original',branches=4)
    torch.manual_seed(101);control=SparseBranches(False,784,101)
    x=torch.randn(2,784)
    assert torch.equal(base(x),control(x))
