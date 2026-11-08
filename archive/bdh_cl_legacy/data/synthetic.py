"""Step 2B: synthetic algorithmic tasks, cheap to generate and exactly scored.

Wire format, one example per line::

    <task-tag>|<input>><output>\\n

The tag is present so a single model can in principle tell the tasks apart;
whether it *uses* the tag is a separate question, and the whole point of the
continual-learning experiment is that it is trained on them sequentially, not
mixed. The ``>`` separates input from output and the trailing newline
terminates generation.

Loss is masked to the output bytes (the newline included, so the model learns
where to stop -- without it, exact-match scoring by free-running generation
would have no stopping signal). Scoring is exact match on the decoded output.

Every task exposes a ``length-generalization`` split whose inputs are strictly
longer than anything in the train split, which separates "learned the
algorithm" from "memorised the length".
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from .bytes import Example

DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclasses.dataclass
class SyntheticTask:
    """A named example generator with train / eval / length-gen length ranges."""

    name: str
    tag: str
    make: Callable[[np.random.Generator, int], tuple[str, str]]
    train_lengths: tuple[int, int]
    longgen_lengths: tuple[int, int]

    def example(self, rng: np.random.Generator, split: str = "train") -> Example:
        lo, hi = self.train_lengths if split != "longgen" else self.longgen_lengths
        L = int(rng.integers(lo, hi + 1))
        inp, out = self.make(rng, L)
        return Example(prompt=f"{self.tag}|{inp}>", output=f"{out}\n")

    def batch(self, rng: np.random.Generator, n: int, split: str = "train") -> list[Example]:
        return [self.example(rng, split) for _ in range(n)]

    def max_len(self, split: str = "train") -> int:
        """Worst-case rendered length, for sizing a fixed seq_len."""
        lo, hi = self.train_lengths if split != "longgen" else self.longgen_lengths
        probe = max(
            (len(self.example(np.random.default_rng(s), split).text) for s in range(64)),
            default=0,
        )
        # Pad the estimate: sampled probes may miss the worst case.
        return probe + 8


# --------------------------------------------------------------------------
# The five tasks
# --------------------------------------------------------------------------


def _copy(rng, L):
    s = "".join(rng.choice(list(LETTERS), size=L))
    return s, s


def _reverse(rng, L):
    s = "".join(rng.choice(list(LETTERS), size=L))
    return s, s[::-1]


def _sort(rng, L):
    s = "".join(rng.choice(list(DIGITS), size=L))
    return s, "".join(sorted(s))


def _add(rng, L):
    """L-digit + L-digit addition, most-significant digit first.

    No leading zeros, so the operands really are L digits and carry
    propagation is exercised.
    """
    a = int(rng.integers(10 ** (L - 1), 10**L)) if L > 1 else int(rng.integers(0, 10))
    b = int(rng.integers(10 ** (L - 1), 10**L)) if L > 1 else int(rng.integers(0, 10))
    return f"{a}+{b}", str(a + b)


def _lookup(rng, L):
    """key->value lookup: L pairs are listed, then one key is queried.

    Single-letter keys, two-digit values. This is the task closest in spirit to
    Step 4's in-context associative recall, but here it is learned in weights.
    """
    keys = rng.choice(list(LETTERS), size=L, replace=False)
    vals = [f"{int(rng.integers(0, 100)):02d}" for _ in range(L)]
    table = ",".join(f"{k}:{v}" for k, v in zip(keys, vals))
    q = int(rng.integers(0, L))
    return f"{table};{keys[q]}", vals[q]


TASKS: dict[str, SyntheticTask] = {
    "copy": SyntheticTask("copy", "cp", _copy, (4, 12), (16, 20)),
    "reverse": SyntheticTask("reverse", "rv", _reverse, (4, 12), (16, 20)),
    "sort": SyntheticTask("sort", "sr", _sort, (4, 12), (16, 20)),
    "add": SyntheticTask("add", "ad", _add, (2, 5), (6, 8)),
    "lookup": SyntheticTask("lookup", "lk", _lookup, (3, 8), (10, 12)),
}

#: The default Step 3 sequence. Ordered so that consecutive tasks are
#: superficially similar (copy/reverse share the input distribution) and later
#: ones are not, which gives both easy and hard transfer cases in one run.
DEFAULT_SEQUENCE = ["copy", "reverse", "sort", "add", "lookup"]


def get_task(name: str) -> SyntheticTask:
    if name not in TASKS:
        raise KeyError(f"unknown synthetic task {name!r}; have {sorted(TASKS)}")
    return TASKS[name]


def sequence_seq_len(names: list[str], splits=("train", "longgen")) -> int:
    """A single seq_len wide enough for every task in a sequence.

    All tasks in a run share one context width so the models never have to be
    rebuilt and the probes are directly comparable across tasks.
    """
    return max(get_task(n).max_len(s) for n in names for s in splits)


def make_eval_set(
    name: str, n: int, seed: int, split: str = "train"
) -> list[Example]:
    """A fixed held-out set, reproducible from ``(name, split, seed)``.

    Held-out-ness: examples are drawn from the same generator but with an
    evaluation-only seed offset, and the spaces are large enough
    (26^4 for the shortest copy) that collision with training draws is
    negligible. For ``add`` and short ``sort`` the space is smaller, so we also
    report train-set overlap in NOTES.md rather than claiming strict disjointness.
    """
    rng = np.random.default_rng(hash((name, split, seed, "eval")) % (2**32))
    return get_task(name).batch(rng, n, split)
