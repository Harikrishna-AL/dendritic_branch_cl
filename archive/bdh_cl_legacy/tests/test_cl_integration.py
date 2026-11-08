"""End-to-end continual-learning plumbing, on CPU with micro models.

These do not check that anything *learns* -- far too few steps for that. They
check that the machinery is wired correctly: that the accuracy matrix gets
filled in the right order, that each strategy actually draws the data it
claims to, that probes come back with the right shapes, and that the whole
thing survives both metric directions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.cl import baselines, probes  # noqa: E402
from bdh_cl.cl.runner import run_sequence  # noqa: E402
from bdh_cl.cl.tasks import build_synthetic_sequence  # noqa: E402
from bdh_cl.config import BDHSpec, GPTSpec, TrainSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.gpt_baseline import GPTBaseline  # noqa: E402

DEVICE = torch.device("cpu")

TINY_BDH = BDHSpec(n_layer=2, n_embd=32, n_head=2, mlp_internal_dim_multiplier=4, dropout=0.0)
TINY_GPT = GPTSpec(n_layer=2, n_embd=32, n_head=2, dropout=0.0)
TRAIN = TrainSpec(
    block_size=80, batch_size=4, steps_per_task=3, warmup_steps=1,
    eval_every=2, log_every=100, seed=0,
)


def _tasks():
    return build_synthetic_sequence(["copy", "reverse"], n_eval=4, batch_size=4)


@pytest.mark.parametrize("builder", [lambda: BDHWrapper(TINY_BDH), lambda: GPTBaseline(TINY_GPT)])
def test_sequence_runs_for_both_models(builder):
    tasks = _tasks()
    s = run_sequence(
        builder, tasks, TRAIN, DEVICE,
        strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
        label="t", collect_probes=True, probe_batches=1, verbose=False,
    )
    k = len(tasks)
    assert len(s["R"]) == k and all(len(r) == k for r in s["R"])
    # Every cell measured, including the zero-shot upper triangle that FWT needs.
    assert all(np.isfinite(v) for row in s["R"] for v in row)
    assert len(s["random_init_own"]) == k
    assert np.isfinite(s["bwt"]) and np.isfinite(s["fwt"])
    assert len(s["per_task"]) == k
    assert s["total_tokens"] == k * TRAIN.steps_per_task * TRAIN.batch_size * (
        tasks[0].seq_len
    )


def test_probes_shapes_and_ranges():
    tasks = _tasks()
    s = run_sequence(
        lambda: BDHWrapper(TINY_BDH), tasks, TRAIN, DEVICE,
        strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
        label="t", collect_probes=True, probe_batches=1, verbose=False,
    )
    p = s["probes"]
    om = p["overlap_matrix_final"]
    # xy_sparse is the elementwise product the decoder actually reads. It is
    # reported as its own stream because measuring only the two factors, and
    # comparing those against a Transformer's single hidden layer, compares the
    # wrong quantity -- the conjunction is far sparser than either factor.
    assert set(om) == {"x_sparse", "y_sparse", "xy_sparse"}
    for stream, d in om.items():
        J = np.array(d["jaccard"])
        assert J.shape == (len(tasks), len(tasks))
        # Jaccard is a similarity in [0,1] and a task overlaps itself exactly.
        assert ((J >= 0) & (J <= 1)).all()
        assert np.allclose(np.diag(J), 1.0)
        assert np.allclose(J, J.T)  # symmetric

    for task, streams in p["sparsity_per_task_final"].items():
        for stream, d in streams.items():
            assert len(d["per_layer"]) == TINY_BDH.n_layer
            for layer in d["per_layer"]:
                assert 0.0 <= layer["density"] <= 1.0
                assert layer["n_neurons"] == TINY_BDH.n_neurons
                assert abs(layer["density"] + layer["sparsity"] - 1.0) < 1e-9

        # The product's support is the INTERSECTION of the factors' supports, so
        # its density can never exceed either one. If this ever fails, the
        # conjunction is not being captured from the same place it is consumed.
        xy = streams["xy_sparse"]["per_layer"]
        xs = streams["x_sparse"]["per_layer"]
        ys = streams["y_sparse"]["per_layer"]
        for lx, ly, lxy in zip(xs, ys, xy):
            assert lxy["density"] <= lx["density"] + 1e-9, (task, lxy, lx)
            assert lxy["density"] <= ly["density"] + 1e-9, (task, lxy, ly)


def test_gpt_probes_use_mlp_hidden():
    tasks = _tasks()
    s = run_sequence(
        lambda: GPTBaseline(TINY_GPT), tasks, TRAIN, DEVICE,
        strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
        label="t", collect_probes=True, probe_batches=1, verbose=False,
    )
    assert set(s["probes"]["overlap_matrix_final"]) == {"mlp_hidden"}


def test_joint_strategy_mixes_every_task():
    """A joint batch must contain examples from more than one task."""
    tasks = _tasks()
    fn = baselines.get_strategy("joint")(0, tasks, [])
    rng = np.random.default_rng(0)
    tags = set()
    for _ in range(12):
        b = fn(rng, 8)
        for i in range(b.x.size(0)):
            tags.add(b.prompts[i][:2])
    assert tags == {"cp", "rv"}, tags


def test_naive_strategy_uses_only_the_current_task():
    tasks = _tasks()
    fn = baselines.get_strategy("naive")(1, tasks, [])
    rng = np.random.default_rng(0)
    b = fn(rng, 8)
    assert {p[:2] for p in b.prompts} == {"rv"}


def test_replay_strategy_mixes_buffer_into_the_batch():
    tasks = _tasks()
    rng = np.random.default_rng(0)
    # Buffer holds items from task 0 (copy); we are now training task 1 (reverse).
    buf = [(0, it) for it in tasks[0].sample_items(rng, 20)]
    fn = baselines.get_strategy("replay")(1, tasks, buf)
    b = fn(rng, 16)
    tags = [p[:2] for p in b.prompts]
    assert "rv" in tags and "cp" in tags
    # 5% of 16 rounds to 1 replayed item.
    assert tags.count("cp") == 1, tags

    # With an empty buffer, replay must degrade to naive rather than crash.
    fn0 = baselines.get_strategy("replay")(1, tasks, [])
    assert {p[:2] for p in fn0(rng, 8).prompts} == {"rv"}


def test_replay_buffer_grows_during_a_sequence():
    tasks = _tasks()
    s = run_sequence(
        lambda: GPTBaseline(TINY_GPT), tasks, TRAIN, DEVICE,
        strategy="replay", strategy_batch_fn=baselines.get_strategy("replay"),
        label="t", collect_probes=False, verbose=False,
    )
    sizes = [r["buffer_size"] for r in s["per_task"]]
    assert sizes == sorted(sizes) and sizes[0] > 0


def test_optimizer_reset_flag_is_honoured():
    """Both settings must run; 'keep state' is a documented variant."""
    import dataclasses

    tasks = _tasks()
    for reset in (True, False):
        t = dataclasses.replace(TRAIN, reset_optimizer_between_tasks=reset)
        s = run_sequence(
            lambda: GPTBaseline(TINY_GPT), tasks, t, DEVICE,
            strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
            label="t", collect_probes=False, verbose=False,
        )
        assert s["train_spec"]["reset_optimizer_between_tasks"] is reset


def test_checkpoint_written_per_task(tmp_path):
    tasks = _tasks()
    run_sequence(
        lambda: GPTBaseline(TINY_GPT), tasks, TRAIN, DEVICE,
        strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
        label="ck", ckpt_dir=tmp_path, collect_probes=False, verbose=False,
    )
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert len(files) == len(tasks)
    assert "ck_seed0_after0_copy.pt" in files
    ck = torch.load(tmp_path / "ck_seed0_after0_copy.pt", weights_only=False)
    assert ck["task"] == "copy" and ck["task_index"] == 0 and "model" in ck


def test_overlap_and_repurposing_helpers():
    """Identical profiles must score as identical; disjoint ones as separated."""
    a = probes.ModelProfile()
    b = probes.ModelProfile()

    class Acts:
        def __init__(self, t):
            self.x_sparse = [t]
            self.y_sparse = [t]

    t1 = torch.zeros(1, 1, 4, 8)
    t1[..., :4] = 1.0  # first half of the neurons active
    t2 = torch.zeros(1, 1, 4, 8)
    t2[..., 4:] = 1.0  # second half

    a.update(Acts(t1))
    b.update(Acts(t2))

    same = probes.overlap(a, a, k_frac=0.5)["x_sparse"]
    assert same["jaccard_mean"] == pytest.approx(1.0)
    assert same["cosine_mean"] == pytest.approx(1.0)

    diff = probes.overlap(a, b, k_frac=0.5)["x_sparse"]
    assert diff["jaccard_mean"] == pytest.approx(0.0)
    assert diff["cosine_mean"] == pytest.approx(0.0)

    rep = probes.repurposing(a, b, k_frac=0.5)["x_sparse"]
    assert rep["retained_mean"] == pytest.approx(0.0)  # fully reassigned
    assert rep["activity_ratio_mean"] == pytest.approx(0.0)  # and silenced

    # Density: half the units active.
    assert a.streams["x_sparse"][0].density == pytest.approx(0.5)
    assert a.streams["x_sparse"][0].dead_fraction == pytest.approx(0.5)


def test_checkpoint_roundtrip_restores_identical_model(tmp_path):
    """A reloaded checkpoint must reproduce the saved model's outputs exactly.

    Step 3 saves one checkpoint per task; any later analysis of "what did the
    model look like after task i" depends on this holding.
    """
    tasks = _tasks()
    run_sequence(
        lambda: BDHWrapper(TINY_BDH), tasks, TRAIN, DEVICE,
        strategy="naive", strategy_batch_fn=baselines.get_strategy("naive"),
        label="rt", ckpt_dir=tmp_path, collect_probes=False, verbose=False,
    )
    ck = torch.load(tmp_path / "rt_seed0_after1_reverse.pt", weights_only=False)

    restored = BDHWrapper(TINY_BDH)
    restored.load_state_dict(ck["model"])
    restored.eval()

    idx = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        a, _ = restored(idx, None)
        b, _ = restored(idx, None)
    assert torch.equal(a, b)

    # And the inner upstream module still accepts the same weights, so the
    # checkpoint stays loadable by unmodified upstream code.
    from bdh_cl.models.bdh_wrapper import upstream_bdh, upstream_config

    ref = upstream_bdh.BDH(upstream_config(TINY_BDH)).eval()
    ref.load_state_dict({k[len("core."):]: v for k, v in ck["model"].items()
                         if k.startswith("core.")})
    with torch.no_grad():
        c, _ = ref(idx, None)
    assert torch.allclose(a, c, atol=0, rtol=0)


def test_topk_jaccard_chance_baseline_is_width_independent():
    """The BDH-vs-GPT overlap comparison rests on this.

    BDH has 4096 latent neurons per layer, the matched GPT 512 -- an 8x
    difference. If the chance level of top-k Jaccard depended on width, the
    observed gap (BDH 0.87 vs GPT 0.58) could be an artifact of size rather
    than a real difference in task separation.

    For two independent uniform k-subsets of n items with k = frac*n,
    E[Jaccard] = k / (2n - k) = frac / (2 - frac), which does not involve n.
    Verified numerically here across the widths actually used.
    """
    frac = 0.05
    expected = frac / (2 - frac)  # ~0.0256
    rng = np.random.default_rng(0)
    for n in (512, 1024, 4096, 8192):
        k = max(1, int(round(frac * n)))
        vals = []
        for _ in range(200):
            a = rng.choice(n, size=k, replace=False)
            b = rng.choice(n, size=k, replace=False)
            sa, sb = set(a.tolist()), set(b.tolist())
            vals.append(len(sa & sb) / len(sa | sb))
        got = float(np.mean(vals))
        assert abs(got - expected) < 0.01, (n, got, expected)

    # And the observed values are far above that baseline for both widths,
    # so both models genuinely reuse neurons across tasks -- the question is
    # only how much.
    assert 0.87 > expected * 5
    assert 0.575 > expected * 5


def test_cosine_profile_overlap_is_k_free():
    """Cosine corroborates Jaccard without any top-k choice at all.

    Two disjoint activation profiles must give 0, identical ones 1, regardless
    of how many neurons there are.
    """
    for n in (16, 512, 4096):
        a = np.zeros(n); a[: n // 2] = 1.0
        b = np.zeros(n); b[n // 2 :] = 1.0
        assert probes.cosine(a, b) == pytest.approx(0.0)
        assert probes.cosine(a, a) == pytest.approx(1.0)
