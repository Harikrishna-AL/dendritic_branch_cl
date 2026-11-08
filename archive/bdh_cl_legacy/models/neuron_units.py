"""Pluggable neuron nonlinearities for BDH (Step 7: structure only).

Upstream ``bdh.py`` applies the same elementwise rule at two places in every
level of the layer loop:

    x_sparse = relu(x            @ encoder  )     # the "x" / query path
    y_sparse = relu(ln(yKV)      @ encoder_v)     # the "y" / value path

Both are "project into the latent neuron space, then rectify". A ``NeuronUnit``
factors that step out so a different neuron model -- in particular a dendritic
one -- can be dropped in without touching the surrounding attention or
residual maths.

Contract
--------
``unit(x, weight)`` maps ``x`` of shape ``(B, nh|1, T, D)`` to latent
activations of shape ``(B, nh, T, N)``.

``weight`` is the host module's projection parameter, shape ``(nh, D, N)``.
Units with ``owns_parameters = False`` (the default unit) *use* it, so that
the upstream ``BDH`` module stays the sole parameter owner and its
``state_dict`` remains upstream-compatible. Units with
``owns_parameters = True`` hold their own parameters and ignore ``weight``.

Every unit must return non-negative activations, because the Step 3 sparsity
and neuron-overlap probes and BDH's own ``x_sparse * y_sparse`` gating both
assume a positive, sparse code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NeuronUnit(nn.Module):
    """Base class. See module docstring for the contract."""

    #: whether this unit carries its own projection parameters
    owns_parameters: bool = False

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def extra_param_count(self, n_head: int, d: int, N: int) -> int:
        """Parameters this unit adds *beyond* the host's ``(nh, D, N)`` weight.

        Used by :mod:`bdh_cl.models.param_accounting` to size variants at
        matched parameter counts.
        """
        return 0


class DefaultNeuronUnit(NeuronUnit):
    """Exactly upstream: ``relu(x @ weight)``.

    Must stay bit-identical to ``bdh.py``; ``tests/test_neuron_units.py``
    asserts that.
    """

    owns_parameters = False

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        if weight is None:
            raise ValueError("DefaultNeuronUnit needs the host's projection weight")
        return F.relu(x @ weight)


class DendriticNeuronUnit(NeuronUnit):
    """Each latent neuron gets ``k`` dendritic branches with their own input
    weights, a per-branch nonlinearity, and a soma that pools them.

    Computation, per head ``h`` and latent neuron ``j``::

        b[..., c, j] = x @ W[h, :, c, j]            # branch c pre-activation
        a[..., c, j] = g(b[..., c, j] - theta[h,c,j])   # branch nonlinearity
        out[..., j]  = soma_c( a[..., c, j] )       # pooled at the soma

    The point is that a branch can be active while its siblings are silent, so
    one neuron can hold several input-specific sub-receptive-fields instead of
    a single compromise.

    ``soma``:

    * ``sum`` -- ``sum_c a_c``. Closest to upstream; with ``k=1`` and no
      threshold this is *exactly* ``relu(x @ W)``, which is asserted in
      ``tests/test_neuron_units.py``.
    * ``max`` -- ``max_c a_c``. Winner-take-all across branches: gradient flows
      to one branch per (token, neuron), which is what makes branch selection a
      real routing decision rather than a reparameterised sum.
    * ``gated`` -- ``sum_c softmax_c(beta_j * a_c) * a_c`` with a learnable
      per-neuron inverse temperature ``beta``. This is the general form: as
      ``beta -> inf`` it becomes ``max`` (hard routing to one branch), as
      ``beta -> 0`` it becomes the mean over branches (the neuron behaves like
      one ordinary unit). Each neuron learns its own ``beta``, so some can act
      as routers and others as plain neurons.

      The gate is deliberately a function of the branch activations, i.e. of
      the input. A *static* per-neuron gate (a bare ``(nh, k, N)`` parameter)
      would be useless here: with fixed positive weights,
      ``sigma_c * relu(b_c) == relu(x @ (sigma_c * w_c))``, so the gate folds
      into the branch weights and the unit collapses to ``sum``. Static gating
      cannot route different inputs to different branches.

    Output stays ``(B, nh, T, N)`` and non-negative, so every metric and probe
    in ``cl/`` works unchanged.

    Parameter accounting: the projection is ``(nh, D, k, N)`` instead of
    ``(nh, D, N)``, so ``encoder`` and ``encoder_v`` grow by ``k`` while
    ``decoder`` -- which reads one soma output per neuron -- does not. See
    :func:`bdh_cl.models.param_accounting.dendritic_budget` for the neuron
    count that keeps the total matched to baseline BDH.
    """

    owns_parameters = True

    def __init__(
        self,
        n_head: int,
        d: int,
        N: int,
        n_branches: int = 4,
        soma: str = "max",
        branch_nonlinearity: str = "relu",
        learn_threshold: bool = False,
        init_std: float = 0.02,
        soma_topk: int = 2,
        wta_frac: float | None = None,
    ) -> None:
        super().__init__()
        if soma not in ("sum", "max", "gated", "topk"):
            raise ValueError(f"unknown soma {soma!r}; want sum|max|gated|topk")
        if branch_nonlinearity not in ("relu",):
            raise ValueError(f"unknown branch nonlinearity {branch_nonlinearity!r}")
        if n_branches < 1:
            raise ValueError("n_branches must be >= 1")

        self.n_head, self.d, self.N = n_head, d, N
        self.k = n_branches
        self.soma = soma
        self.branch_nonlinearity = branch_nonlinearity
        #: soma="topk": how many of the k branches contribute. 1 == max,
        #: k == sum, so this fills the gap those two leave.
        self.soma_topk = max(1, min(int(soma_topk), n_branches))
        #: k-WTA ACROSS NEURONS, applied after the soma: keep this fraction of
        #: the N units per (token, head) and zero the rest. Distinct from the
        #: branch competition above -- that picks among one neuron's branches,
        #: this picks among the whole layer's neurons. It is the form of
        #: winner-take-all the sparse-representation literature ties to
        #: forgetting resistance (Ahmad & Scheinkman; Numenta's active
        #: dendrites), and it adds no parameters, so matched comparisons hold.
        if wta_frac is not None and not 0.0 < wta_frac <= 1.0:
            raise ValueError(f"wta_frac must be in (0, 1], got {wta_frac}")
        self.wta_frac = wta_frac

        # (nh, D, k*N): one matmul, reshaped to branches afterwards.
        self.weight = nn.Parameter(
            torch.zeros(n_head, d, n_branches * N).normal_(std=init_std)
        )
        self.threshold = (
            nn.Parameter(torch.zeros(n_head, n_branches, N)) if learn_threshold else None
        )
        # Per-neuron inverse temperature, via softplus so beta > 0.
        # softplus(0) ~ 0.69: starts between mean and max, free to sharpen.
        self.log_beta = (
            nn.Parameter(torch.zeros(n_head, 1, N)) if soma == "gated" else None
        )

        #: set True to retain branch activations for the probes. Off by
        #: default: the branch tensor is k times the size of the output.
        self.capture = False
        self.last_branches: torch.Tensor | None = None

    def branch_activations(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, nh, T, k, N)``, post-nonlinearity, non-negative."""
        B = x.shape[0]
        T = x.shape[-2]
        z = x @ self.weight  # B, nh, T, k*N
        z = z.view(B, self.n_head, T, self.k, self.N)
        if self.threshold is not None:
            z = z - self.threshold.view(1, self.n_head, 1, self.k, self.N)
        return F.relu(z)

    def kwta(self, out: torch.Tensor) -> torch.Tensor:
        """Keep the top ``wta_frac`` of neurons per (batch, head, token).

        Competition is across the N neurons of the layer, independently at each
        token, so which units win is input-dependent. Ties at the threshold may
        keep slightly more than the nominal count, which is harmless here and
        cheaper than breaking them arbitrarily.
        """
        if self.wta_frac is None:
            return out
        keep = max(1, int(round(self.wta_frac * self.N)))
        if keep >= self.N:
            return out
        thresh = out.topk(keep, dim=-1).values[..., -1:]
        return out * (out >= thresh)

    def pool(self, a: torch.Tensor) -> torch.Tensor:
        """Combine the k branch activations into one value per neuron."""
        if self.soma == "sum":
            return a.sum(dim=-2)
        if self.soma == "max":
            return a.amax(dim=-2)
        if self.soma == "topk":
            # Winner-take-most: the strongest soma_topk branches contribute and
            # the rest are silent. soma_topk=1 reproduces max exactly and
            # soma_topk=k reproduces sum, so this interpolates between them.
            return a.topk(self.soma_topk, dim=-2).values.sum(dim=-2)
        # gated: softmax over branches, sharpened by a learnable per-neuron beta
        beta = F.softplus(self.log_beta).view(1, self.n_head, 1, 1, self.N)
        w = torch.softmax(a * beta, dim=-2)
        return (a * w).sum(dim=-2)

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        a = self.branch_activations(x)
        if self.capture:
            self.last_branches = a.detach()
        return self.kwta(self.pool(a))

    def extra_param_count(self, n_head: int, d: int, N: int) -> int:
        """Parameters beyond the host's ``(nh, D, N)`` weight, which we replace."""
        own = self.weight.numel()
        own += self.threshold.numel() if self.threshold is not None else 0
        own += self.log_beta.numel() if self.log_beta is not None else 0
        return own - n_head * d * N


class DeepNeuronUnit(NeuronUnit):
    """``relu(relu(x @ A) @ B)`` -- extra nonlinear depth, NO per-neuron grouping.

    THE control for the dendritic claim. Every dendritic result so far is an
    end-to-end improvement with no mechanism behind it: beta never trains, the
    gate is task-blind, representation quality is unchanged, the soma choice
    barely matters, and `sum` (no routing at all) wins on language modelling.
    All of that is consistent with a plainer explanation -- a sum of ``k``
    rectified branches is simply a richer positive function of the input than
    one rectifier, and ANY equal-parameter increase in per-neuron nonlinearity
    might do as well.

    This unit tests exactly that. It spends the same extra parameters on one
    hidden layer with a LEARNED projection instead of ``k`` parallel branches
    with fixed pooling, and it is matched exactly rather than approximately:

        dendritic encoder   nh * d * (k * N)
        this encoder        nh * (d * M + M * N) = nh * M * (d + N)
        equal when          M = d * k * N / (d + N)

    The decoder is ``nh * N * d`` in both, because both emit one value per
    neuron for the same ``N``. So with ``match_branches=k`` the two models agree
    on parameter count, neuron count and output dimensionality, and differ ONLY
    in how the encoder's extra capacity is spent.

    If this matches the dendritic numbers, "dendritic" was never the right
    description and the finding is about nonlinear depth per parameter. If it
    does not, the per-neuron grouping is doing real work.
    """

    owns_parameters = True

    def __init__(
        self,
        n_head: int,
        d: int,
        N: int,
        hidden: int | None = None,
        match_branches: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden is None:
            if match_branches is None:
                raise ValueError(
                    "DeepNeuronUnit needs `hidden`, or `match_branches=k` to "
                    "derive the width that matches a k-branch dendritic unit"
                )
            # M = d*k*N/(d+N); derived here rather than written into a config so
            # the match cannot drift when d, N or k change.
            hidden = int(round(d * match_branches * N / (d + N)))
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")

        self.n_head, self.d, self.N = n_head, d, N
        self.hidden = hidden
        self.match_branches = match_branches
        self.A = nn.Parameter(torch.zeros(n_head, d, hidden).normal_(std=init_std))
        self.B = nn.Parameter(torch.zeros(n_head, hidden, N).normal_(std=init_std))

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Two rectifiers, so the unit is a genuinely deeper positive function,
        # and the output stays non-negative as the probes and BDH's own
        # x_sparse * y_sparse gate both require.
        return F.relu(F.relu(x @ self.A) @ self.B)

    def extra_param_count(self, n_head: int, d: int, N: int) -> int:
        """Parameters beyond the host's ``(nh, D, N)`` weight, which we replace."""
        return self.A.numel() + self.B.numel() - n_head * d * N

    def extra_repr(self) -> str:
        return (
            f"n_head={self.n_head}, d={self.d}, N={self.N}, hidden={self.hidden}, "
            f"match_branches={self.match_branches}"
        )


class SparseDendriticNeuronUnit(DendriticNeuronUnit):
    """Dendritic unit whose branches each contact only a few input coordinates,
    optionally with structural rewiring (Poirazi & Mel 2001; Limbacher &
    Legenstein 2020).

    Every branch column of ``weight`` has exactly ``contacts`` active synapses,
    chosen at random at init. With ``n_branches=1`` this is a sparse point
    neuron, which is the control: same neurons, same synapse count, no
    grouping into separately rectified branches.

    Rewiring is SET-style, a simplification of DEEP R (Bellec et al. 2018):
    every ``rewire_every`` optimizer steps each branch prunes its
    ``rewire_frac`` smallest-magnitude active synapses and activates the same
    number of currently dormant ones at random, at weight zero. The per-branch
    budget never changes. No sign constraint and no Langevin noise, unlike the
    paper. Adam moments of every touched entry are zeroed, because a regrown
    synapse inheriting a stale second moment would get a mis-scaled first step.

    ``rewire_mode="coactivity"`` replaces the random regrow target with a
    Hebbian/coactivity trace (Limbacher & Legenstein 2020): an EMA of
    ``input_i * branch_activity_c``, accumulated every forward pass while
    training. Regrowth then goes to the dormant synapse whose input already
    correlates most with what the branch is doing, instead of a uniform
    random dormant synapse -- the real-BDH analogue of the toy lab's
    "clustered" rewiring.
    """

    def __init__(
        self,
        n_head: int,
        d: int,
        N: int,
        contacts: int = 32,
        rewire_every: int = 0,
        rewire_frac: float = 0.1,
        mask_seed: int = 0,
        rewire_mode: str = "random",
        coactivity_decay: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(n_head, d, N, **kwargs)
        if not 1 <= contacts <= d:
            raise ValueError(f"contacts must be in [1, {d}], got {contacts}")
        if rewire_every and contacts == d:
            raise ValueError("rewiring needs dormant synapses: contacts < d")
        if rewire_mode not in ("random", "coactivity"):
            raise ValueError(f"rewire_mode must be 'random' or 'coactivity', got {rewire_mode!r}")
        self.contacts = contacts
        self.rewire_every = int(rewire_every)
        self.rewire_frac = float(rewire_frac)
        self.rewire_mode = rewire_mode
        self.coactivity_decay = float(coactivity_decay)
        self.n_rewire = max(1, round(rewire_frac * contacts)) if rewire_every else 0
        self.n_rewire = min(self.n_rewire, d - contacts) if rewire_every else 0
        # Separate generator, so the mask does not depend on (or perturb) the
        # global RNG stream that the weight init and data order use.
        # Offset by the run seed so the wiring varies across seeds like the
        # weights do; mask_seed separates the x and y paths within one run.
        gen = torch.Generator().manual_seed((mask_seed + 1000 * torch.initial_seed()) % (2**63))
        cols = n_head * self.k * N
        ranks = torch.rand(cols, d, generator=gen).argsort(dim=1).argsort(dim=1)
        mask = (ranks < contacts).view(n_head, self.k * N, d).transpose(1, 2)
        self.register_buffer("mask", mask.contiguous())
        with torch.no_grad():
            self.weight.mul_(self.mask)
        self.register_buffer("coactivity", torch.zeros_like(self.weight))
        self.steps = 0
        self.rewired_total = 0

    def branch_activations(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[-2]
        z = x @ (self.weight * self.mask)
        z = z.view(B, self.n_head, T, self.k, self.N)
        if self.threshold is not None:
            z = z - self.threshold.view(1, self.n_head, 1, self.k, self.N)
        out = F.relu(z)
        if self.rewire_mode == "coactivity" and self.training:
            self._update_coactivity(x, out.reshape(B, self.n_head, T, self.k * self.N))
        return out

    @torch.no_grad()
    def _update_coactivity(self, x: torch.Tensor, activity: torch.Tensor) -> None:
        """EMA of input-branch coactivation, ``(nh, D, k*N)``, used by
        ``rewire_mode="coactivity"`` to pick regrow targets."""
        x_exp = x.detach().expand(-1, self.n_head, -1, -1)
        co = torch.einsum("bhtd,bhtc->hdc", x_exp, activity.detach())
        co = co / (x.shape[0] * x.shape[-2])
        self.coactivity.mul_(self.coactivity_decay).add_(co, alpha=1 - self.coactivity_decay)

    @torch.no_grad()
    def rewire(self, optimizer=None) -> int:
        """Prune the weakest active synapses of every branch and regrow as many
        dormant ones. Returns the number of synapses moved."""
        n = self.n_rewire
        if n == 0:
            return 0
        mag = self.weight.abs()
        weakest = torch.where(self.mask, mag, torch.full_like(mag, float("inf")))
        prune = torch.zeros_like(self.mask)
        prune.scatter_(1, weakest.topk(n, dim=1, largest=False).indices, True)
        # Regrow among synapses that were dormant BEFORE this prune, so a
        # just-pruned synapse cannot come straight back.
        if self.rewire_mode == "coactivity":
            score = torch.where(
                self.mask, torch.full_like(self.coactivity, float("-inf")), self.coactivity
            )
        else:
            score = torch.rand(mag.shape, device=mag.device)
            score = torch.where(self.mask, torch.full_like(score, -1.0), score)
        grow = torch.zeros_like(self.mask)
        grow.scatter_(1, score.topk(n, dim=1).indices, True)

        self.mask.logical_and_(~prune).logical_or_(grow)
        touched = prune | grow
        self.weight.masked_fill_(touched, 0.0)
        if optimizer is not None:
            st = optimizer.state.get(self.weight, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key].masked_fill_(touched, 0.0)
        moved = int(grow.sum())
        self.rewired_total += moved
        return moved

    def post_step(self, optimizer=None) -> None:
        """Called by the training loop after every optimizer step."""
        self.steps += 1
        if self.rewire_every and self.steps % self.rewire_every == 0:
            self.rewire(optimizer)

    def active_synapses(self) -> int:
        return int(self.mask.sum())

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, soma={self.soma}, contacts={self.contacts}, "
            f"rewire_every={self.rewire_every}, n_rewire={self.n_rewire}, "
            f"rewire_mode={self.rewire_mode}"
        )


#: name -> class, selected by ``BDHSpec.neuron_unit``
UNITS: dict[str, type[NeuronUnit]] = {
    "default": DefaultNeuronUnit,
    "dendritic": DendriticNeuronUnit,
    "sparse_dendritic": SparseDendriticNeuronUnit,
    "deep": DeepNeuronUnit,
}


def build_neuron_unit(name: str, **kwargs) -> NeuronUnit:
    """Instantiate a unit by config name."""
    if name not in UNITS:
        raise KeyError(f"unknown neuron_unit {name!r}; have {sorted(UNITS)}")
    cls = UNITS[name]
    if not cls.owns_parameters:
        return cls()
    return cls(**kwargs)
