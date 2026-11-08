import torch
from bdh_cl.models.protected_branches import ProtectedBranches


def test_actual_projected_update_preserves_protected_preactivations():
    torch.manual_seed(4)
    m = ProtectedBranches(d=8, neurons=3, rank=3)
    q = torch.linalg.qr(torch.randn(8, 3)).Q
    m.basis[:] = q
    m.ranks[:] = 3
    old = m.weight.detach().clone()
    delta = m.project(torch.randn_like(old))
    with torch.no_grad():
        m.weight.add_(delta)
    inputs = torch.randn(7, 3) @ q.T
    before = torch.einsum('bd,nkd->bnk', inputs, old)
    after = torch.einsum('bd,nkd->bnk', inputs, m.weight)
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)


def test_online_basis_is_local_orthonormal_and_bounded():
    torch.manual_seed(2)
    m = ProtectedBranches(d=8, neurons=3, rank=3)
    for _ in range(8):
        x = torch.randn(16, 8)
        winners = torch.zeros(16, 3, dtype=torch.long)
        m.consolidate(x, winners, torch.ones(16, dtype=torch.bool))
    assert (m.ranks[:, 0] == 3).all()
    assert (m.ranks[:, 1:] == 0).all()
    for q in m.basis[:, 0]:
        torch.testing.assert_close(q.T@q, torch.eye(3), atol=1e-6, rtol=1e-6)


def test_recruitment_requires_both_error_and_restricted_updates():
    m = ProtectedBranches(d=8, neurons=3)
    raw = torch.ones_like(m.weight)
    for _ in range(25):
        assert m.maybe_recruit(2., .8, raw) == []
    assert not m.enabled[:, -1].any()
    for _ in range(19):
        assert m.maybe_recruit(2., .1, raw) == []
    assert len(m.maybe_recruit(2., .1, raw)) == 3
    assert m.enabled.all()
