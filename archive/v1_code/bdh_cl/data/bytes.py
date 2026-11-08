"""Byte-level tokenization (vocab 256), shared by every task family.

Everything in this project is a byte sequence, which is what makes a single
model comparable across English text, Python source, German, MNIST pixels and
algorithmic puzzles. There is no tokenizer to train and no vocabulary drift
between tasks -- a genuine advantage when the point is to isolate *weight*
interference from *vocabulary* interference.

Two batch shapes are used throughout:

``LMBatch``   -- next-byte prediction over a long stream (Step 2A language).
                Loss on every position, exactly like upstream ``train.py``.
``MaskedBatch`` -- fixed-width examples with a per-position loss mask
                (Step 2B synthetic, Step 2C MNIST, Step 4 ICL). Only the
                positions the task cares about contribute to the loss.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

VOCAB_SIZE = 256
PAD = 0  # byte 0 never appears in our ASCII payloads, so it is a safe pad


def encode(text: str) -> np.ndarray:
    """str -> uint8 array of its UTF-8 bytes."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def decode(arr) -> str:
    """Byte array/tensor -> str, tolerating invalid UTF-8 (model output)."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().to(torch.uint8).cpu().numpy()
    return bytes(np.asarray(arr, dtype=np.uint8)).decode(errors="backslashreplace")


# --------------------------------------------------------------------------
# Streams: next-byte prediction over a corpus
# --------------------------------------------------------------------------


@dataclasses.dataclass
class LMBatch:
    x: torch.Tensor  # (B, T) int64
    y: torch.Tensor  # (B, T) int64
    loss_mask: torch.Tensor | None = None

    def to(self, device) -> "LMBatch":
        return LMBatch(
            self.x.to(device),
            self.y.to(device),
            None if self.loss_mask is None else self.loss_mask.to(device),
        )


class ByteStream:
    """A byte corpus with a deterministic held-out tail.

    Sampling mirrors upstream ``train.py``: uniformly random ``block_size``
    windows. The split is by position (first 90% train) rather than at random,
    so train and val never share a window.
    """

    def __init__(self, data: np.ndarray, *, block_size: int, val_frac: float = 0.1):
        data = np.asarray(data, dtype=np.uint8)
        if len(data) < block_size + 2:
            raise ValueError(f"corpus of {len(data)} bytes too short for block_size={block_size}")
        self.block_size = block_size
        cut = int((1.0 - val_frac) * len(data))
        self.splits = {"train": data[:cut], "val": data[cut:]}

    @classmethod
    def from_text(cls, text: str, **kw) -> "ByteStream":
        return cls(encode(text), **kw)

    def n_bytes(self, split: str) -> int:
        return len(self.splits[split])

    def batch(self, split: str, batch_size: int, rng: np.random.Generator) -> LMBatch:
        data = self.splits[split]
        hi = len(data) - self.block_size - 1
        if hi <= 0:
            raise ValueError(f"split {split!r} has only {len(data)} bytes")
        ix = rng.integers(0, hi, size=batch_size)
        x = np.stack([data[i : i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(np.int64)
        return LMBatch(torch.from_numpy(x), torch.from_numpy(y))

    def iter_val_blocks(self, batch_size: int, max_batches: int | None = None):
        """Deterministic, non-overlapping sweep of the val split.

        Used for bits-per-byte, where random sampling would add variance
        between evaluations of the same model and make forgetting curves noisy.
        """
        data = self.splits["val"]
        T = self.block_size
        n = (len(data) - 1) // T
        starts = [i * T for i in range(n)]
        for b0 in range(0, len(starts), batch_size):
            chunk = starts[b0 : b0 + batch_size]
            if not chunk:
                break
            x = np.stack([data[i : i + T] for i in chunk]).astype(np.int64)
            y = np.stack([data[i + 1 : i + 1 + T] for i in chunk]).astype(np.int64)
            yield LMBatch(torch.from_numpy(x), torch.from_numpy(y))
            if max_batches is not None and b0 // batch_size + 1 >= max_batches:
                break


# --------------------------------------------------------------------------
# Masked examples: fixed-width, loss on selected positions only
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MaskedBatch:
    """``x``/``y`` are the usual shifted pair; ``loss_mask`` marks the target
    positions that count. ``prompt_len`` and ``gold`` support exact-match
    scoring by free-running generation."""

    x: torch.Tensor  # (B, T) int64
    y: torch.Tensor  # (B, T) int64
    loss_mask: torch.Tensor  # (B, T) float32
    prompt_len: torch.Tensor | None = None  # (B,) int64
    gold: list[str] | None = None
    prompts: list[str] | None = None

    def to(self, device) -> "MaskedBatch":
        return MaskedBatch(
            self.x.to(device),
            self.y.to(device),
            self.loss_mask.to(device),
            None if self.prompt_len is None else self.prompt_len.to(device),
            self.gold,
            self.prompts,
        )

    def __len__(self) -> int:
        return self.x.size(0)


@dataclasses.dataclass
class Example:
    """One rendered example: ``prompt`` then ``output``, both plain text.

    ``prompt + output`` is the full byte string the model sees. The loss covers
    exactly the ``output`` bytes.
    """

    prompt: str
    output: str

    @property
    def text(self) -> str:
        return self.prompt + self.output


def collate(examples: list[Example], seq_len: int | None = None) -> MaskedBatch:
    """Pack examples into a shifted, masked, right-padded batch.

    For an example with prompt ``p`` and output ``o``, the model is trained to
    predict every byte of ``o``. In shifted coordinates the supervised target
    positions are ``len(p)-1 .. len(p)+len(o)-2``, so predicting the first
    output byte is conditioned on the whole prompt and nothing is leaked.
    """
    encoded = [(encode(e.prompt), encode(e.output)) for e in examples]
    full_lens = [len(p) + len(o) for p, o in encoded]
    need = max(full_lens)
    T = need - 1 if seq_len is None else seq_len
    if T < need - 1:
        raise ValueError(f"seq_len={seq_len} too small; need at least {need - 1}")

    B = len(examples)
    x = np.full((B, T), PAD, dtype=np.int64)
    y = np.full((B, T), PAD, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float32)

    for b, (p, o) in enumerate(encoded):
        full = np.concatenate([p, o]).astype(np.int64)
        L = len(full)
        x[b, : L - 1] = full[:-1]
        y[b, : L - 1] = full[1:]
        # target index j holds full[j+1]; output bytes are full[len(p):]
        mask[b, len(p) - 1 : L - 1] = 1.0

    return MaskedBatch(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        loss_mask=torch.from_numpy(mask),
        prompt_len=torch.tensor([len(p) for p, _ in encoded], dtype=torch.long),
        gold=[e.output for e in examples],
        prompts=[e.prompt for e in examples],
    )
