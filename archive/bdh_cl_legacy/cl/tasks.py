"""A uniform Task interface over every data family. (Addition to the Step 1
layout, noted in NOTES.md -- the runner needs one contract to drive them all.)

A ``Task`` must provide:

``sample_items(rng, n)``    -> n training items, drawn fresh
``collate_items(items)``    -> a model-ready batch
``train_batch(rng, bs)``    -> shorthand for the two above
``eval_batches()``          -> deterministic held-out batches (probes, bpb)
``evaluate(model, device)``  -> a metric dict including ``metric_name``
``metric_name`` / ``higher_is_better``

Splitting sampling from collation is what makes the replay baseline simple: a
replay buffer is just a list of *items*, and mixing replay into a batch is
concatenating item lists before collating.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import numpy as np
import torch

from ..data import mnist, synthetic
from ..data.bytes import ByteStream, Example, LMBatch, collate
from . import evaluate as ev


class Task(Protocol):  # pragma: no cover - structural
    name: str
    metric_name: str
    higher_is_better: bool

    def sample_items(self, rng: np.random.Generator, n: int) -> list[Any]: ...
    def collate_items(self, items: list[Any]): ...
    def eval_batches(self) -> list[Any]: ...
    def evaluate(self, model, device, *, quick: bool = False,
                 with_longgen: bool = False) -> dict: ...


# --------------------------------------------------------------------------
# Step 2B: synthetic algorithmic tasks
# --------------------------------------------------------------------------


@dataclasses.dataclass
class SyntheticCLTask:
    """One algorithmic task. Items are :class:`Example`; metric is exact match."""

    name: str
    seq_len: int
    n_eval: int = 200
    eval_seed: int = 0
    batch_size: int = 32
    n_quick: int = 32
    metric_name: str = "exact_match"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        self.task = synthetic.get_task(self.name)
        self._eval = synthetic.make_eval_set(self.name, self.n_eval, self.eval_seed, "train")
        self._longgen = synthetic.make_eval_set(
            self.name, self.n_eval, self.eval_seed, "longgen"
        )

    # -- training ----------------------------------------------------------
    def sample_items(self, rng: np.random.Generator, n: int) -> list[Example]:
        return self.task.batch(rng, n, "train")

    def collate_items(self, items: list[Example]):
        return collate(items, seq_len=self.seq_len)

    def train_batch(self, rng: np.random.Generator, batch_size: int):
        return self.collate_items(self.sample_items(rng, batch_size))

    # -- evaluation --------------------------------------------------------
    def eval_batches(self) -> list:
        return [
            collate(self._eval[i : i + self.batch_size], seq_len=self.seq_len)
            for i in range(0, len(self._eval), self.batch_size)
        ]

    def evaluate(
        self, model, device, *, quick: bool = False, with_longgen: bool = False
    ) -> dict:
        """Full held-out evaluation, or a cheap subset for training curves.

        ``with_longgen`` roughly doubles the cost, so the runner only asks for
        the length-generalization split on the final row of the matrix, where
        it is actually reported.

        ``quick=True`` scores exact match on the first ``n_quick`` examples only
        and skips the length-generalization split. It is used for the dense
        curves that plasticity is read off, where the full evaluation would cost
        more than the training it is measuring.
        """
        if quick:
            subset = self._eval[: self.n_quick]
            out = ev.exact_match(model, subset, device, batch_size=self.batch_size)
            out["metric_name"] = self.metric_name
            out["value"] = out["exact_match"]
            out["quick"] = True
            return out

        out = ev.evaluate_synthetic(
            model, self._eval, device, seq_len=self.seq_len, batch_size=self.batch_size
        )
        if with_longgen:
            lg = ev.exact_match(model, self._longgen, device, batch_size=self.batch_size)
            out["longgen_exact_match"] = lg["exact_match"]
            out["longgen_samples"] = lg["samples"][:4]
        out["metric_name"] = self.metric_name
        out["value"] = out["exact_match"]
        out["quick"] = False
        return out


# --------------------------------------------------------------------------
# Step 2A: language domain streams
# --------------------------------------------------------------------------


@dataclasses.dataclass
class LanguageCLTask:
    """One text domain. Items are (x, y) byte windows; metric is bits-per-byte."""

    name: str
    stream: ByteStream
    batch_size: int = 8
    eval_batches_max: int = 24
    metric_name: str = "bits_per_byte"
    higher_is_better: bool = False

    def sample_items(self, rng: np.random.Generator, n: int) -> list[tuple]:
        b = self.stream.batch("train", n, rng)
        return [(b.x[i], b.y[i]) for i in range(n)]

    def collate_items(self, items: list[tuple]):
        x = torch.stack([i[0] for i in items])
        y = torch.stack([i[1] for i in items])
        return LMBatch(x=x, y=y)

    def train_batch(self, rng: np.random.Generator, batch_size: int):
        return self.stream.batch("train", batch_size, rng)

    def eval_batches(self) -> list:
        return list(
            self.stream.iter_val_blocks(self.batch_size, max_batches=self.eval_batches_max)
        )

    def evaluate(
        self, model, device, *, quick: bool = False, with_longgen: bool = False
    ) -> dict:
        """bits-per-byte on the held-out tail. ``quick`` uses fewer batches."""
        batches = list(
            self.stream.iter_val_blocks(
                self.batch_size,
                max_batches=4 if quick else self.eval_batches_max,
            )
        )
        out = ev.bits_per_byte(model, batches, device)
        out["metric_name"] = self.metric_name
        out["value"] = out["bits_per_byte"]
        out["quick"] = quick
        return out


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def build_synthetic_sequence(
    names: list[str],
    *,
    n_eval: int = 200,
    eval_seed: int = 0,
    batch_size: int = 32,
    seq_len: int | None = None,
) -> list[SyntheticCLTask]:
    """All tasks in a sequence share one seq_len so the probes stay comparable."""
    L = seq_len or synthetic.sequence_seq_len(names)
    return [
        SyntheticCLTask(
            name=n, seq_len=L, n_eval=n_eval, eval_seed=eval_seed, batch_size=batch_size
        )
        for n in names
    ]


# --------------------------------------------------------------------------
# Step 2C: split MNIST as byte sequences
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MNISTCLTask:
    """One split-MNIST task. Items are :class:`Example`; metric is exact match.

    Because the rendered output is a single digit byte plus a newline, exact
    match on free-running greedy generation *is* classification accuracy -- the
    same metric and the same code path as the synthetic tasks, so the two
    families are directly comparable.

    There is no length-generalization split here: every image renders to the
    same length, so ``with_longgen`` is accepted and ignored.
    """

    pool: mnist.MNISTPool
    seq_len: int
    n_eval: int = 200
    eval_seed: int = 0
    batch_size: int = 32
    n_quick: int = 32
    metric_name: str = "exact_match"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        self.name = self.pool.name
        self._eval = self.pool.eval_set(self.n_eval, self.eval_seed)

    # -- training ----------------------------------------------------------
    def sample_items(self, rng: np.random.Generator, n: int) -> list[Example]:
        return self.pool.sample(rng, n)

    def collate_items(self, items: list[Example]):
        return collate(items, seq_len=self.seq_len)

    def train_batch(self, rng: np.random.Generator, batch_size: int):
        return self.collate_items(self.sample_items(rng, batch_size))

    # -- evaluation --------------------------------------------------------
    def eval_batches(self) -> list:
        return [
            collate(self._eval[i : i + self.batch_size], seq_len=self.seq_len)
            for i in range(0, len(self._eval), self.batch_size)
        ]

    def evaluate(
        self, model, device, *, quick: bool = False, with_longgen: bool = False
    ) -> dict:
        examples = self._eval[: self.n_quick] if quick else self._eval
        if quick:
            out = ev.exact_match(model, examples, device, batch_size=self.batch_size)
        else:
            out = ev.evaluate_synthetic(
                model, examples, device, seq_len=self.seq_len, batch_size=self.batch_size
            )
        out["metric_name"] = self.metric_name
        out["value"] = out["exact_match"]
        out["quick"] = quick
        return out


def build_mnist_sequence(
    cache_dir,
    *,
    side: int = 14,
    n_eval: int = 200,
    eval_seed: int = 0,
    batch_size: int = 32,
    task_tags: bool = False,
    splits=mnist.SPLITS,
) -> list[MNISTCLTask]:
    """The five split-MNIST tasks, all sharing one seq_len."""
    pools = mnist.build_pools(
        cache_dir, side=side, splits=splits, task_tags=task_tags
    )
    L = mnist.rendered_len(side) - 1  # collate stores the shifted pair
    tasks = [
        MNISTCLTask(
            pool=p,
            seq_len=L,
            n_eval=n_eval,
            eval_seed=eval_seed,
            batch_size=batch_size,
        )
        for p in pools
    ]
    # A REVISIT schedule repeats a split, e.g. [[0,1],[2,3],[0,1]], so the same
    # environment is seen twice. The two visits share data and eval set -- they
    # are the same environment -- but need distinct names, or the results dict
    # and the accuracy matrix silently collapse them into one column.
    seen: dict[str, int] = {}
    for t in tasks:
        base = t.pool.name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            t.name = f"{base}#{seen[base]}"
    return tasks
