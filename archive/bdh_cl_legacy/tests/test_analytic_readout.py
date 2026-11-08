import numpy as np
import torch

from bdh_cl.scripts import analytic_readout as A


def _data(n_per, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(100, d, generator=g) * 3
    y = torch.arange(100).repeat_interleave(n_per)
    return centres[y] + torch.randn(len(y), d, generator=g), y


def test_streaming_ridge_equals_batch_ridge():
    x, y = _data(5)
    st = A.Stats(16, 100, "cpu")
    for c in range(0, 100, 10):          # add task by task
        m = (y >= c) & (y < c + 10)
        st.add(x[m], y[m])
    seen = list(range(100))
    W = A.ridge_weights(st, seen)
    xa = A.aug(x)
    Y = torch.nn.functional.one_hot(y, 100).double()
    scale = torch.linalg.eigvalsh(xa.T @ xa).clamp(min=0).mean()
    for a in (1e-2, 1.0):
        batch = torch.linalg.solve(xa.T @ xa + a * scale * torch.eye(17, dtype=torch.float64), xa.T @ Y)
        assert torch.allclose(W[a], batch, atol=1e-6)


def test_slda_within_class_covariance_matches_direct():
    x, y = _data(6)
    st = A.Stats(16, 100, "cpu")
    st.add(x, y)
    seen = list(range(100))
    mu = torch.stack([x[y == c].mean(0) for c in seen]).double()
    centred = x.double() - mu[y]
    cov = centred.T @ centred / (len(x) - 100)
    evals, Q = torch.linalg.eigh(cov)
    a = 1e-2
    prec = Q @ torch.diag(1 / (evals.clamp(min=0) + a * evals.clamp(min=0).mean())) @ Q.T
    pm, b = A.slda_weights(st, seen)[a]
    assert torch.allclose(pm, prec @ mu.T, atol=1e-6)
    assert torch.allclose(b, -0.5 * ((mu @ prec) * mu).sum(1), atol=1e-6)


def test_maps_shapes_and_binary_pop():
    x = torch.randn(7, 16)
    assert A.apply(A.make_map("raw", 16, 0, "cpu"), x).shape == (7, 16)
    r = A.apply(A.make_map("relu_30", 16, 0, "cpu"), x)
    assert r.shape == (7, 30) and (r >= 0).all()
    p = A.apply(A.make_map("pop_40_k4_K3", 16, 0, "cpu"), x)
    assert p.shape == (7, 40) and torch.equal(p.sum(1), torch.full((7,), 4.0))
    assert torch.equal(p, A.apply(A.make_map("pop_40_k4_K3", 16, 0, "cpu"), x))


def test_run_one_selects_on_valid_and_reports_test():
    data = tuple(_data(n, seed=s) for n, s in ((6, 0), (2, 1), (2, 2)))
    tasks = [list(range(i * 10, i * 10 + 10)) for i in range(3)]
    res = A.run_one("relu_40", 0, data, tasks, "cpu")
    for r in ("ncm", "ridge", "slda"):
        assert len(res[r]["test_matrix"]) == 3
        assert 0 <= res[r]["test"]["final_avg_acc"] <= 1
    grid = res["ridge"]["valid_grid"]
    assert res["ridge"]["valid"]["final_avg_acc"] == max(grid.values())
