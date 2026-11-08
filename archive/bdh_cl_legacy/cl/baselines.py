"""Reference points for the Step 3 comparison.

Three strategies, each expressed as a ``strategy_batch_fn(i, tasks, buffer)``
that returns the ``batch_fn`` the runner should use for task ``i``:

``naive``   sequential finetuning, nothing to stop forgetting. The lower bound.
``joint``   every batch drawn from *all* tasks at once. The upper bound; note
            that it is not a continual learner at all, it just consumes the
            same total step budget with i.i.d. data.
``replay``  a small buffer holding 5% of each finished task's training items,
            mixed into every batch. The standard cheap baseline.

All three see the same number of optimizer steps and the same batch size, so
differences are about data ordering, not compute. Where the replay buffer
changes the *effective* number of new-task examples per step, that is stated
in NOTES.md rather than corrected for -- it is the usual definition.
"""

from __future__ import annotations

import numpy as np


def naive_strategy():
    """Sequential finetune: task i's batches come only from task i."""

    def make(i, tasks, buffer):
        return tasks[i].train_batch

    return make


def joint_strategy():
    """Joint training: every batch is a mixture over all tasks.

    Each example in the batch picks a task uniformly at random, so over a batch
    the mixture is even. Tasks must share a collation (they do: one seq_len per
    sequence), otherwise mixing is impossible.
    """

    def make(i, tasks, buffer):
        def batch_fn(rng: np.random.Generator, batch_size: int):
            which = rng.integers(0, len(tasks), size=batch_size)
            items = []
            for t_idx in which:
                items.extend(tasks[int(t_idx)].sample_items(rng, 1))
            return tasks[0].collate_items(items)

        return batch_fn

    return make


def replay_strategy(replay_frac: float = 0.05, min_replay: int = 1):
    """Experience replay from a buffer of past tasks' items.

    ``replay_frac`` is the *share of each batch* drawn from the buffer, chosen
    to match the 5% figure in the brief. The buffer itself is also filled at 5%
    of each task's training examples (the runner does that), so 5% appears in
    both senses; NOTES.md records this.
    """

    def make(i, tasks, buffer):
        task = tasks[i]
        if i == 0 or not buffer:
            return task.train_batch

        def batch_fn(rng: np.random.Generator, batch_size: int):
            n_replay = max(min_replay, int(round(replay_frac * batch_size)))
            n_replay = min(n_replay, batch_size - 1, len(buffer))
            n_new = batch_size - n_replay
            items = task.sample_items(rng, n_new)
            pick = rng.integers(0, len(buffer), size=n_replay)
            items.extend(buffer[int(p)][1] for p in pick)
            return task.collate_items(items)

        return batch_fn

    return make


STRATEGIES = {
    "naive": naive_strategy,
    "joint": joint_strategy,
    "replay": replay_strategy,
}


def get_strategy(name: str, **kwargs):
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
