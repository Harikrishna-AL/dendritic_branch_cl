"""Split MNIST as byte sequences (Step 2C).

An MNIST image is already bytes -- 784 pixels in 0..255 -- so in principle it
drops straight into a vocab-256 model. Two adjustments are needed in practice:

**Grey levels are quantised to hex characters, not passed as raw bytes.**
:func:`bdh_cl.data.bytes.encode` is UTF-8, so any pixel value above 127 would
expand to *two* bytes and silently destroy the alignment between pixel index
and sequence position. Quantising to 16 levels and rendering them as
``0123456789abcdef`` keeps every pixel exactly one byte, so ``collate``,
``exact_match`` and the length-bucketing in ``evaluate.py`` all work unchanged.
MNIST is classifiable from far less than 4 bits of grey, so nothing of
substance is lost.

**Images are downsampled to 14x14 by default.** 28x28 renders a 784-byte
prompt, which costs ~4x more per step than 196 bytes for no benefit on this
task. ``side=28`` is available for the full-resolution check.

Rendered form, matching the synthetic family's ``<tag>|<input>><output>\\n``::

    d|8ba0...f1>7\\n

**Task-agnostic by default.** The synthetic tasks each carry their *own* tag,
so a model there can in principle tell the tasks apart from the input -- those
runs are task-aware. Split MNIST uses one constant tag for every task, which
makes it *class-incremental*: at test time nothing says which pair of digits an
image came from, and the model may emit any digit. That is the standard and
strictly harder setting. Pass ``task_tags=True`` for the task-aware variant,
which is the apples-to-apples comparison against the synthetic results.
"""

from __future__ import annotations

import dataclasses
import gzip
import struct
from pathlib import Path

import numpy as np

from .bytes import Example

#: 16 grey levels, one byte each, all ASCII.
GREY_CHARS = "0123456789abcdef"

#: Constant tag for the class-incremental setting: identical across tasks, so
#: it carries no information about which task an example came from.
SHARED_TAG = "d"

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

#: Split MNIST: five binary tasks, the standard partition.
SPLITS: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


# --------------------------------------------------------------------------
# Raw IDX loading -- no torchvision dependency
# --------------------------------------------------------------------------


def _read_idx(path: Path) -> np.ndarray:
    """Parse one gzipped IDX file, checking the magic number.

    Written out rather than taken from torchvision so this works in the
    cluster venv, which has torch but not torchvision.
    """
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic, count = struct.unpack(">II", raw[:8])
    if magic == IMAGE_MAGIC:
        rows, cols = struct.unpack(">II", raw[8:16])
        arr = np.frombuffer(raw[16:], dtype=np.uint8)
        expect = count * rows * cols
        if arr.size != expect:
            raise ValueError(f"{path.name}: expected {expect} pixel bytes, got {arr.size}")
        return arr.reshape(count, rows, cols)
    if magic == LABEL_MAGIC:
        arr = np.frombuffer(raw[8:], dtype=np.uint8)
        if arr.size != count:
            raise ValueError(f"{path.name}: expected {count} labels, got {arr.size}")
        return arr
    raise ValueError(f"{path.name}: unexpected IDX magic {magic}")


def load_raw(cache_dir: str | Path) -> dict[str, np.ndarray]:
    """Load the four IDX files from ``cache_dir``.

    The files are *not* downloaded here. They are fetched once and shipped to
    wherever the run happens, so a compute node never needs the network.
    """
    d = Path(cache_dir)
    missing = [n for n in FILES.values() if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"MNIST files missing from {d}: {missing}. Fetch them with "
            "bdh_cl/scripts/fetch_mnist.sh"
        )
    out = {k: _read_idx(d / v) for k, v in FILES.items()}
    if len(out["train_images"]) != len(out["train_labels"]):
        raise ValueError("train image/label count mismatch")
    if len(out["test_images"]) != len(out["test_labels"]):
        raise ValueError("test image/label count mismatch")
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def downsample(images: np.ndarray, side: int) -> np.ndarray:
    """Block-mean an ``(N, 28, 28)`` uint8 stack down to ``(N, side, side)``."""
    n, h, w = images.shape
    if side == h == w:
        return images
    if h % side or w % side:
        raise ValueError(f"side={side} must divide {h}x{w} evenly")
    f = h // side
    # mean over each f x f block, in float to avoid uint8 overflow
    blocks = images.reshape(n, side, f, side, f).astype(np.float32)
    return blocks.mean(axis=(2, 4)).round().astype(np.uint8)


def render(image: np.ndarray, label: int, tag: str = SHARED_TAG) -> Example:
    """One image -> one :class:`Example`.

    ``image`` is uint8 of any square side; it is quantised to 16 grey levels
    and flattened row-major.
    """
    q = image.reshape(-1) >> 4  # 0..255 -> 0..15
    pixels = "".join(GREY_CHARS[v] for v in q)
    return Example(prompt=f"{tag}|{pixels}>", output=f"{label}\n")


def rendered_len(side: int, tag: str = SHARED_TAG) -> int:
    """Total rendered byte length, for sizing a fixed ``seq_len``."""
    return len(tag) + 1 + side * side + 1 + 1 + 1  # tag | pixels > label \n


# --------------------------------------------------------------------------
# Split MNIST pools
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MNISTPool:
    """The images and labels of one split-MNIST task, train and test.

    Images are pre-downsampled once at construction, because rendering happens
    every training step and block-meaning 60k images per step would dominate
    the step time.
    """

    name: str
    classes: tuple[int, ...]
    tag: str
    side: int
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        idx = rng.integers(0, len(self.train_labels), size=n)
        return [
            render(self.train_images[i], int(self.train_labels[i]), self.tag) for i in idx
        ]

    def eval_set(self, n: int, seed: int) -> list[Example]:
        """A fixed held-out set drawn from the MNIST *test* split.

        Deterministic in ``(task, n, seed)``, and drawn without replacement so
        the same image never appears twice in one evaluation.
        """
        rng = np.random.default_rng([seed, abs(hash(self.name)) % (2**31)])
        take = min(n, len(self.test_labels))
        idx = rng.permutation(len(self.test_labels))[:take]
        return [
            render(self.test_images[i], int(self.test_labels[i]), self.tag) for i in idx
        ]


def build_pools(
    cache_dir: str | Path,
    *,
    side: int = 14,
    splits: tuple[tuple[int, ...], ...] = SPLITS,
    task_tags: bool = False,
) -> list[MNISTPool]:
    """One :class:`MNISTPool` per split, sharing one downsampling pass.

    ``task_tags=False`` (the default) gives every task the same tag, i.e. the
    class-incremental setting -- see the module docstring.
    """
    raw = load_raw(cache_dir)
    tr_x = downsample(raw["train_images"], side)
    te_x = downsample(raw["test_images"], side)
    tr_y, te_y = raw["train_labels"], raw["test_labels"]

    pools = []
    for i, classes in enumerate(splits):
        name = "mnist_" + "".join(str(c) for c in classes)
        tag = GREY_CHARS[i] if task_tags else SHARED_TAG
        trm = np.isin(tr_y, classes)
        tem = np.isin(te_y, classes)
        pools.append(
            MNISTPool(
                name=name,
                classes=tuple(classes),
                tag=tag,
                side=side,
                train_images=tr_x[trm],
                train_labels=tr_y[trm],
                test_images=te_x[tem],
                test_labels=te_y[tem],
            )
        )
    return pools
