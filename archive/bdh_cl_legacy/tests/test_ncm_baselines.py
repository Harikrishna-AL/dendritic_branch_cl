import torch

from bdh_cl.scripts.ncm_baselines import (
    RandDend,
    RandKWTA,
    kwta,
    ncm_class_incremental,
    summarize,
)


def test_kwta_exactly_k_binary():
    out = kwta(torch.randn(7, 50), 5)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert (out.sum(1) == 5).all()


def test_random_maps_shapes_and_determinism():
    x = torch.rand(4, 64)
    for make in (lambda s: RandKWTA(64, 40, 4, s, "cpu"), lambda s: RandDend(64, 40, 4, 3, 5, s, "cpu")):
        a, b, c = make(0)(x), make(0)(x), make(1)(x)
        assert a.shape == (4, 40) and (a.sum(1) == 4).all()
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


def _clusters(n_cls, per, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n_cls, d, generator=g) * 5
    y = torch.arange(n_cls).repeat_interleave(per)
    return centres[y] + 0.1 * torch.randn(len(y), d, generator=g), y


def test_separable_clusters_perfect_and_matrix_shape():
    f, y = _clusters(6, 20)
    ft, yt = _clusters(6, 10, seed=0)
    tasks = [[0, 1], [2, 3], [4, 5]]
    acc = ncm_class_incremental(f, y, ft, yt, tasks)
    assert [len(r) for r in acc] == [1, 2, 3]
    assert summarize(acc)["final_avg_acc"] == 1.0


def test_argmax_is_over_all_seen_classes_not_task():
    # Classes 0 and 2 share a centre: class-incremental eval must confuse
    # them once both are seen, which a task-restricted eval would hide.
    d = 8
    c = torch.eye(d)[:3] * 5
    c[2] = c[0]
    y = torch.arange(3).repeat_interleave(10)
    f = c[y]
    acc = ncm_class_incremental(f, y, f, y, [[0, 1], [2]])
    assert acc[0][0] == 1.0          # before class 2 arrives
    # Once class 2 is stored, 0 and 2 tie; argmax resolves to the first, so
    # task [2] is misread as class 0 -- only possible if 0 is a candidate.
    assert acc[1][1] == 0.0
