"""Held-out evaluation. (An addition to the Step 1 layout; noted in NOTES.md.)

Two metrics, matching the brief:

* **bits-per-byte** for the language streams (Step 2A). Mean cross-entropy in
  nats over held-out bytes, divided by ln 2. Because the vocabulary *is* the
  byte alphabet, this is directly bits/byte with no tokenizer correction.
* **exact-match accuracy** for the synthetic tasks (Step 2B) and MNIST
  (Step 2C), measured by *free-running greedy generation* from the prompt --
  not teacher forcing. Teacher forcing would credit a model that can continue
  a correct prefix it could never have produced itself.

A cheap teacher-forced masked accuracy is also provided, for the dense
training curves that plasticity needs, where running generation every few
steps would dominate the runtime.

Batched generation detail: rows in a batch have different prompt lengths, and
neither BDH nor our GPT baseline takes a padding mask, so left-padding would
silently corrupt the context. Instead we bucket examples by exact prompt length
and generate one bucket at a time.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from ..data.bytes import PAD, Example, collate, decode, encode

NEWLINE = 10


# --------------------------------------------------------------------------
# Bits per byte
# --------------------------------------------------------------------------


@torch.no_grad()
def bits_per_byte(model, batches, device) -> dict:
    """Token-weighted mean cross-entropy over held-out bytes, in bits."""
    was_training = model.training
    model.eval()
    total_nats, total_bytes = 0.0, 0
    try:
        for batch in batches:
            b = batch.to(device)
            logits, _ = model(b.x, None)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), b.y.reshape(-1), reduction="none"
            )
            mask = getattr(b, "loss_mask", None)
            if mask is None:
                total_nats += float(ce.sum())
                total_bytes += ce.numel()
            else:
                m = mask.reshape(-1).to(ce.dtype)
                total_nats += float((ce * m).sum())
                total_bytes += int(m.sum())
    finally:
        model.train(was_training)
    if total_bytes == 0:
        return {"bits_per_byte": float("nan"), "n_bytes": 0}
    nats = total_nats / total_bytes
    return {
        "bits_per_byte": nats / math.log(2),
        "nats_per_byte": nats,
        "n_bytes": total_bytes,
    }


# --------------------------------------------------------------------------
# Teacher-forced masked accuracy (cheap curve metric)
# --------------------------------------------------------------------------


@torch.no_grad()
def masked_token_accuracy(model, batches, device) -> dict:
    """Fraction of supervised target positions whose argmax is correct.

    An upper bound on exact-match: every byte of every output must be right
    for an exact match, but here each byte is scored given the *true* prefix.
    """
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    seq_correct, seq_total = 0, 0
    try:
        for batch in batches:
            b = batch.to(device)
            logits, _ = model(b.x, None)
            pred = logits.argmax(-1)
            m = b.loss_mask.bool()
            hit = (pred == b.y) & m
            correct += int(hit.sum())
            total += int(m.sum())
            # per-sequence: all supervised positions correct
            per_row_ok = (hit.sum(1) == m.sum(1))
            seq_correct += int(per_row_ok.sum())
            seq_total += int(per_row_ok.numel())
    finally:
        model.train(was_training)
    return {
        "token_accuracy": correct / total if total else float("nan"),
        "teacher_forced_seq_accuracy": seq_correct / seq_total if seq_total else float("nan"),
        "n_tokens": total,
    }


# --------------------------------------------------------------------------
# Free-running greedy generation -> exact match
# --------------------------------------------------------------------------


@torch.no_grad()
def greedy_continue(model, prompts: list[str], max_new: int, device) -> list[str]:
    """Greedy-decode ``max_new`` bytes after each prompt.

    All prompts in one call must have the same byte length (the caller buckets).
    Generation stops early once every row has emitted a newline.
    """
    lens = {len(encode(p)) for p in prompts}
    assert len(lens) == 1, f"greedy_continue needs equal-length prompts, got {lens}"

    idx = torch.stack(
        [torch.from_numpy(encode(p).astype("int64")) for p in prompts]
    ).to(device)
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=device)
    out = [[] for _ in prompts]

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new):
            logits, _ = model(idx, None)
            nxt = logits[:, -1, :].argmax(-1)  # greedy
            # Rows already finished emit padding and are ignored downstream.
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            for r, tok in enumerate(nxt.tolist()):
                if not done[r]:
                    out[r].append(tok)
            done = done | (nxt == NEWLINE)
            idx = torch.cat([idx, nxt.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break
    finally:
        model.train(was_training)

    return [decode(torch.tensor(o, dtype=torch.long)) for o in out]


def exact_match(
    model,
    examples: list[Example],
    device,
    *,
    batch_size: int = 32,
    max_new_slack: int = 4,
) -> dict:
    """Exact-match accuracy by free-running greedy generation.

    Comparison ignores the terminating newline on both sides, so a model that
    produces the right answer counts as correct; failing to terminate at all
    still counts as wrong, because the generated text will then include extra
    bytes.
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        buckets[len(encode(e.prompt))].append(i)

    n_correct = 0
    per_example: list[dict] = []

    for _plen, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = [examples[i] for i in idxs[s : s + batch_size]]
            max_new = max(len(encode(e.output)) for e in chunk) + max_new_slack
            gens = greedy_continue(model, [e.prompt for e in chunk], max_new, device)
            for e, g in zip(chunk, gens):
                got = g.split("\n")[0]
                want = e.output.rstrip("\n")
                ok = got == want
                n_correct += ok
                per_example.append(
                    {"prompt": e.prompt, "gold": want, "pred": got, "correct": bool(ok)}
                )

    return {
        "exact_match": n_correct / len(examples) if examples else float("nan"),
        "n_examples": len(examples),
        "samples": per_example[:8],
    }


def evaluate_synthetic(
    model, examples: list[Example], device, *, seq_len: int, batch_size: int = 32
) -> dict:
    """Both metrics for one synthetic held-out set."""
    batches = [
        collate(examples[i : i + batch_size], seq_len=seq_len)
        for i in range(0, len(examples), batch_size)
    ]
    out = {}
    out.update(masked_token_accuracy(model, batches, device))
    out.update(bits_per_byte(model, batches, device))
    out.update(exact_match(model, examples, device, batch_size=batch_size))
    return out
