"""Data pipeline: masking correctness is the thing that matters most here.

A leak in ``collate`` -- supervising a prompt byte, or letting an output byte
into the input at the wrong offset -- would silently inflate every exact-match
number in the report.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.data import synthetic  # noqa: E402
from bdh_cl.data.bytes import (  # noqa: E402
    PAD,
    VOCAB_SIZE,
    ByteStream,
    Example,
    collate,
    decode,
    encode,
)


def test_encode_decode_roundtrip_including_multibyte():
    for s in ["hello", "", "Grüße", "മലയാളം", "a\nb\t"]:
        arr = encode(s)
        assert arr.dtype == np.uint8
        assert decode(arr) == s
        assert arr.max(initial=0) < VOCAB_SIZE


def test_collate_supervises_exactly_the_output_bytes():
    ex = [Example("ad|12+34>", "46\n"), Example("cp|ab>", "ab\n")]
    b = collate(ex, seq_len=32)

    for i, e in enumerate(ex):
        m = b.loss_mask[i].bool()
        # the supervised targets are exactly the output
        assert decode(b.y[i][m]) == e.output
        assert int(m.sum()) == len(encode(e.output))

        # the input at those positions is the output shifted right by one, i.e.
        # the model predicts output[0] from the last prompt byte
        first = int(m.nonzero()[0])
        assert b.x[i][first].item() == encode(e.prompt)[-1]
        assert b.y[i][first].item() == encode(e.output)[0]

        # nothing beyond the example is supervised
        L = len(encode(e.text))
        assert b.loss_mask[i][L - 1 :].sum() == 0
        assert (b.x[i][L - 1 :] == PAD).all()


def test_collate_x_y_are_a_shifted_pair():
    ex = [Example("lk|a:01;a>", "01\n")]
    b = collate(ex, seq_len=24)
    full = encode(ex[0].text).astype(np.int64)
    L = len(full)
    assert torch.equal(b.x[0][: L - 1], torch.from_numpy(full[:-1]))
    assert torch.equal(b.y[0][: L - 1], torch.from_numpy(full[1:]))


def test_collate_rejects_too_small_seq_len():
    with pytest.raises(ValueError):
        collate([Example("abcdefgh>", "xyz\n")], seq_len=4)


@pytest.mark.parametrize("name", sorted(synthetic.TASKS))
def test_synthetic_tasks_are_wellformed_and_correct(name):
    """Check the task's own answer really is the right answer."""
    task = synthetic.get_task(name)
    rng = np.random.default_rng(3)
    for split in ("train", "longgen"):
        for _ in range(40):
            e = task.example(rng, split)
            assert e.prompt.startswith(f"{task.tag}|")
            assert e.prompt.endswith(">")
            assert e.output.endswith("\n")
            inp = e.prompt[len(task.tag) + 1 : -1]
            out = e.output[:-1]
            if name == "copy":
                assert out == inp
            elif name == "reverse":
                assert out == inp[::-1]
            elif name == "sort":
                assert out == "".join(sorted(inp))
            elif name == "add":
                a, bb = inp.split("+")
                assert int(out) == int(a) + int(bb)
            elif name == "lookup":
                table, q = inp.split(";")
                d = dict(p.split(":") for p in table.split(","))
                assert out == d[q]


@pytest.mark.parametrize("name", sorted(synthetic.TASKS))
def test_longgen_inputs_are_strictly_longer_than_training(name):
    """The length-generalization split must not overlap training lengths."""
    task = synthetic.get_task(name)
    assert task.longgen_lengths[0] > task.train_lengths[1], name


def test_eval_sets_are_reproducible_and_split_dependent():
    a = synthetic.make_eval_set("sort", 20, seed=0)
    b = synthetic.make_eval_set("sort", 20, seed=0)
    c = synthetic.make_eval_set("sort", 20, seed=1)
    assert [e.text for e in a] == [e.text for e in b]
    assert [e.text for e in a] != [e.text for e in c]
    lg = synthetic.make_eval_set("sort", 20, seed=0, split="longgen")
    assert [e.text for e in a] != [e.text for e in lg]


def test_sequence_seq_len_fits_every_task_and_split():
    names = synthetic.DEFAULT_SEQUENCE
    L = synthetic.sequence_seq_len(names)
    rng = np.random.default_rng(5)
    for n in names:
        t = synthetic.get_task(n)
        for split in ("train", "longgen"):
            for _ in range(100):
                e = t.example(rng, split)
                assert len(encode(e.text)) - 1 <= L, (n, split, e.text)
            # must actually collate at that width
            collate(t.batch(rng, 4, split), seq_len=L)


def test_bytestream_split_is_disjoint_and_val_sweep_is_deterministic():
    data = np.arange(256, dtype=np.uint8).repeat(20)  # 5120 bytes
    s = ByteStream(data, block_size=32, val_frac=0.1)
    assert s.n_bytes("train") + s.n_bytes("val") == len(data)

    rng = np.random.default_rng(0)
    b = s.batch("train", 4, rng)
    assert b.x.shape == (4, 32) and b.y.shape == (4, 32)
    # y is x shifted by one
    assert torch.equal(b.x[:, 1:], b.y[:, :-1])

    one = [bb.x for bb in s.iter_val_blocks(2)]
    two = [bb.x for bb in s.iter_val_blocks(2)]
    assert len(one) == len(two) and all(torch.equal(a, c) for a, c in zip(one, two))


def test_bytestream_rejects_short_corpus():
    with pytest.raises(ValueError):
        ByteStream(np.zeros(10, dtype=np.uint8), block_size=64)
