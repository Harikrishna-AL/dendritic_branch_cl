"""Split MNIST rendering and task wiring (Step 2C).

The load-dependent tests skip when the IDX cache is absent, so the suite still
passes on a machine that has not fetched the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.cl.tasks import build_mnist_sequence  # noqa: E402
from bdh_cl.data import mnist  # noqa: E402
from bdh_cl.data.bytes import collate, encode  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "data_cache" / "mnist"
have_data = all((CACHE / n).exists() for n in mnist.FILES.values())
needs_data = pytest.mark.skipif(not have_data, reason="MNIST IDX cache not present")


# -- rendering -------------------------------------------------------------


def test_every_pixel_is_exactly_one_byte():
    """The whole point of quantising to hex: UTF-8 must not expand anything.

    A raw pixel value above 127 would encode to two bytes and silently break
    the pixel-index / sequence-position correspondence.
    """
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    ex = mnist.render(img, 7)
    assert len(encode(ex.prompt)) == len(ex.prompt)
    assert len(encode(ex.output)) == len(ex.output)
    # tag + '|' + 256 pixels + '>'
    assert len(encode(ex.prompt)) == 1 + 1 + 256 + 1


def test_render_is_reversible_to_grey_level():
    img = np.array([[0, 15], [240, 255]], dtype=np.uint8)
    ex = mnist.render(img, 3)
    pixels = ex.prompt.split("|")[1].rstrip(">")
    assert pixels == "00ff"  # 0->0, 15->0, 240->15, 255->15
    assert ex.output == "3\n"


def test_rendered_len_matches_actual():
    for side in (7, 14, 28):
        img = np.zeros((side, side), dtype=np.uint8)
        ex = mnist.render(img, 0)
        assert mnist.rendered_len(side) == len(ex.prompt) + len(ex.output)


def test_downsample_block_means():
    img = np.zeros((1, 28, 28), dtype=np.uint8)
    img[0, :2, :2] = 200  # one 2x2 block of the 14x14 grid
    out = mnist.downsample(img, 14)
    assert out.shape == (1, 14, 14)
    assert out[0, 0, 0] == 200
    assert out[0, 0, 1] == 0


@pytest.mark.parametrize("side", [5, 8, 9])
def test_downsample_rejects_non_divisor(side):
    """28 = 2^2 * 7, so only 1/2/4/7/14/28 are valid. 8 in particular is not,
    which is the mistake this guard caught the first time the suite ran."""
    with pytest.raises(ValueError, match="must divide"):
        mnist.downsample(np.zeros((1, 28, 28), dtype=np.uint8), side)


def test_downsample_identity_at_full_res():
    img = np.random.default_rng(0).integers(0, 256, (3, 28, 28), dtype=np.uint8)
    assert np.array_equal(mnist.downsample(img, 28), img)


def test_no_overflow_on_bright_blocks():
    """Block-mean in float, not uint8 -- 4 saturated pixels must average to 255."""
    img = np.full((1, 28, 28), 255, dtype=np.uint8)
    assert mnist.downsample(img, 14).min() == 255


# -- class-incremental vs task-aware --------------------------------------


@needs_data
def test_default_is_class_incremental():
    """Every task shares one tag, so the input reveals nothing about the task."""
    pools = mnist.build_pools(CACHE, side=7)
    tags = {p.tag for p in pools}
    assert tags == {mnist.SHARED_TAG}


@needs_data
def test_task_tags_are_distinct_when_requested():
    pools = mnist.build_pools(CACHE, side=7, task_tags=True)
    tags = [p.tag for p in pools]
    assert len(set(tags)) == len(pools)


# -- pools ----------------------------------------------------------------


@needs_data
def test_pools_contain_only_their_own_classes():
    for pool in mnist.build_pools(CACHE, side=7):
        assert set(np.unique(pool.train_labels)) == set(pool.classes)
        assert set(np.unique(pool.test_labels)) == set(pool.classes)


@needs_data
def test_eval_set_is_deterministic_and_duplicate_free():
    pool = mnist.build_pools(CACHE, side=7)[0]
    a = pool.eval_set(64, seed=0)
    b = pool.eval_set(64, seed=0)
    assert [e.prompt for e in a] == [e.prompt for e in b]
    assert pool.eval_set(64, seed=1) != a
    assert len({e.prompt for e in a}) == len(a)  # drawn without replacement


@needs_data
def test_eval_comes_from_the_test_split():
    """A held-out image must never appear in the training pool."""
    pool = mnist.build_pools(CACHE, side=14)[0]
    train_seen = {mnist.render(i, 0).prompt for i in pool.train_images[:2000]}
    for ex in pool.eval_set(32, seed=0):
        assert ex.prompt not in train_seen


# -- task wiring ----------------------------------------------------------


@needs_data
def test_sequence_shares_one_seq_len_and_collates():
    tasks = build_mnist_sequence(CACHE, side=14, n_eval=64, batch_size=8)
    assert len(tasks) == 5
    assert len({t.seq_len for t in tasks}) == 1
    rng = np.random.default_rng(0)
    for t in tasks:
        b = t.train_batch(rng, 4)
        assert b.x.shape == (4, t.seq_len)
        assert b.loss_mask is not None
        # exactly the digit byte and the newline are supervised
        assert b.loss_mask.sum(dim=1).unique().tolist() == [2.0]


@needs_data
def test_loss_mask_targets_the_label_byte():
    tasks = build_mnist_sequence(CACHE, side=7, n_eval=16, batch_size=4)
    t = tasks[0]
    ex = t._eval[0]
    b = collate([ex], seq_len=t.seq_len)
    pos = b.loss_mask[0].nonzero().flatten().tolist()
    targets = b.y[0][pos].tolist()
    assert targets == [ord(ex.output[0]), ord("\n")]


@needs_data
def test_task_names_are_unique_and_describe_their_classes():
    tasks = build_mnist_sequence(CACHE, side=7, n_eval=16)
    names = [t.name for t in tasks]
    assert names == ["mnist_01", "mnist_23", "mnist_45", "mnist_67", "mnist_89"]


def test_missing_cache_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_mnist"):
        mnist.build_pools(tmp_path)
