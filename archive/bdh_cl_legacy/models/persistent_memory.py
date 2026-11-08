"""A slow synaptic state that survives across sequences.

Milestone 1 of the persistent-memory direction. The claim being tested is
narrow on purpose: does a decaying outer-product memory, with no dendrites and
no task signal of any kind, buy anything measurable when an environment recurs?

Why this is a small change to BDH's equations and a large one to its semantics.
Upstream computes attention as ``scores = (QR @ KR.mT).tril(-1)`` then
``scores @ V``, which for token t is::

    y_t = q_t * sum_{t' < t} k_{t'}^T v_{t'}  =  q_t S_t

so BDH ALREADY maintains this state, as an (N, D) matrix per head -- it just
never materialises it (``tril(-1)`` forces the computation through a T x T
token matrix) and it resets at every sequence boundary. This module keeps the
same object with a decay and no reset::

    Sigma <- lambda * Sigma + eta * mean_bt( x_sparse^T (x) v )
    read   = x_sparse @ Sigma

Three design decisions, each of which would break the thing if taken the other
way:

**Position-free.** BDH applies RoPE to Q and K with absolute position. A state
carried across sequences would be mixing incompatible rotation frames, so the
slow path reads and writes the UN-rotated sparse code. The fast within-sequence
path keeps RoPE and is untouched.

**Hebbian, not backpropagated.** Sigma is a buffer, not a parameter: it is
written under ``no_grad`` and never receives a gradient. Gradients still reach
the encoders through the READ, so the model can learn to use the memory, but
there is no backpropagation through time and no truncation window to tune. This
is what the fast-weight literature does and it is the simpler half of the
design space.

**Causal at the batch level.** The read uses the state as it stood BEFORE this
batch, and the write is applied after. Reading a state that already contains
the current batch would leak the answer into the prediction.

Stability: the read is added to ``yKV`` before upstream's ``ln``, so the
layer-norm that was already there bounds the contribution for free. That is why
no separate normalisation of Sigma is needed at this scale.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class MemorySpec:
    """Configuration for the slow state."""

    enabled: bool = False
    #: lambda: per-write retention. 1.0 never forgets and saturates; small
    #: values forget within a task and cannot bridge a revisit.
    decay: float = 0.999
    #: eta: write strength.
    eta: float = 1.0
    #: Scale applied to the read before it joins the fast path. Starts small so
    #: an untrained memory cannot swamp the within-sequence attention.
    read_scale: float = 1.0
    #: Write only every n-th step; a cheap way to slow the state down without
    #: pushing decay to the point of numerical trouble.
    write_every: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if self.eta < 0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")
        if self.write_every < 1:
            raise ValueError(f"write_every must be >= 1, got {self.write_every}")


class PersistentMemory(nn.Module):
    """Sigma, shaped (n_head, N, D), shared across layers as BDH shares weights.

    Registered as a buffer so it moves with ``.to(device)`` and is saved in the
    ``state_dict`` -- a checkpoint of a model with memory has to include what it
    remembers, or reloading it silently discards the thing under test.
    """

    def __init__(self, spec: MemorySpec, n_head: int, N: int, d: int):
        super().__init__()
        self.spec = spec
        self.n_head, self.N, self.d = n_head, N, d
        self.register_buffer("sigma", torch.zeros(n_head, N, d))
        self.register_buffer("writes", torch.zeros((), dtype=torch.long))
        #: Staged write for the current forward, applied by commit(). Not a
        #: buffer: per-forward scratch, not state worth checkpointing.
        self._pending: torch.Tensor | None = None

    def reset(self) -> None:
        """Wipe the state. Used between independent runs, never between tasks --
        surviving a task boundary is the whole point."""
        self.sigma.zero_()
        self.writes.zero_()
        self._pending = None

    @torch.no_grad()
    def snapshot(self) -> torch.Tensor:
        """The state as it stands now, as a tensor the forward pass may read.

        A CLONE, not a view, and that is load-bearing twice over:

        * Autograd records the tensor ``read`` consumed. Mutating that same
          storage in place later in the same forward bumps its version counter,
          and backward then refuses with "one of the variables needed for
          gradient computation has been modified by an inplace operation".
          The first version of this file did exactly that and every
          memory-enabled run died on the first backward pass.
        * BDH reuses one parameter set for every layer, so without a single
          snapshot taken once per forward, layer 2 would read a state already
          containing layer 1's write FROM THE SAME BATCH -- a within-batch leak,
          and a correctness problem independent of the crash.
        """
        return self.sigma.detach().clone()

    def read(self, code: torch.Tensor, snap: torch.Tensor) -> torch.Tensor:
        """``(B, nh, T, N) -> (B, nh, T, D)``, the recalled contribution."""
        return self.spec.read_scale * torch.einsum("bhtn,hnd->bhtd", code, snap)

    @torch.no_grad()
    def stage(self, code: torch.Tensor, value: torch.Tensor) -> None:
        """Accumulate a write without touching ``sigma`` yet.

        Deferring the mutation to :meth:`commit` is what keeps ``sigma``'s
        version counter stable for the whole forward pass.

        ``value`` is the residual stream, ``(B, 1, T, D)`` or ``(B, nh, T, D)``.
        Averaging over batch and time keeps the write magnitude independent of
        batch size and sequence length, so ``eta`` means the same thing across
        configurations.
        """
        v = value if value.shape[1] == self.n_head else value.expand(-1, self.n_head, -1, -1)
        outer = torch.einsum("bhtn,bhtd->hnd", code.detach(), v.detach())
        contribution = outer / (code.shape[0] * code.shape[2])
        self._pending = (
            contribution if self._pending is None else self._pending + contribution
        )

    @torch.no_grad()
    def commit(self) -> None:
        """``Sigma <- lambda Sigma + eta * (staged writes)``, once per forward."""
        if self._pending is None:
            return
        self.writes += 1
        if int(self.writes) % self.spec.write_every == 0:
            self.sigma.mul_(self.spec.decay).add_(self._pending, alpha=self.spec.eta)
        self._pending = None

    def extra_repr(self) -> str:
        return (
            f"n_head={self.n_head}, N={self.N}, d={self.d}, "
            f"decay={self.spec.decay}, eta={self.spec.eta}"
        )
