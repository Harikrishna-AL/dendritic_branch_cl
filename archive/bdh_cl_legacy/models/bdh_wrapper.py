"""Wrapper around the *unmodified* upstream ``bdh.py``.

Design constraints (from the brief):

* Never edit the maths in upstream ``bdh.py``. We keep the clone at
  ``bdh-d/upstream_bdh`` untouched and import it.
* The upstream ``BDH`` instance is the sole parameter owner (``self.core``),
  so checkpoints stay upstream-compatible.
* We still need per-layer ``x_sparse`` / ``y_sparse`` for the Step 3 probes,
  and a pluggable :class:`~bdh_cl.models.neuron_units.NeuronUnit` for Step 7.
  Neither is reachable through upstream's ``forward``: the ReLUs are
  functional calls, not submodules, so forward hooks cannot see them.

So :meth:`BDHWrapper.forward` re-walks the same layer loop using
``self.core``'s parameters and ``self.core.attn``. That is a re-expression of
upstream's forward, which the brief allows only with proof: ``tests/
test_bdh_equivalence.py`` asserts ``torch.allclose`` against
``self.core.forward`` on fixed seeds, in both eval and train mode.

The only line that differs in substance is the neuron nonlinearity, which is
routed through the NeuronUnit. With ``neuron_unit="default"`` it is literally
``F.relu(x @ weight)``, i.e. the upstream expression.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..config import BDHSpec
from .neuron_units import build_neuron_unit
from .persistent_memory import MemorySpec, PersistentMemory

# --------------------------------------------------------------------------
# Import the untouched upstream module
# --------------------------------------------------------------------------

UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "upstream_bdh"

if not (UPSTREAM_DIR / "bdh.py").exists():  # pragma: no cover
    raise ImportError(
        f"upstream BDH not found at {UPSTREAM_DIR}. Clone it with:\n"
        f"  git clone https://github.com/pathwaycom/bdh.git {UPSTREAM_DIR}"
    )

if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

import bdh as upstream_bdh  # noqa: E402  (must follow the sys.path insert)


def upstream_config(spec: BDHSpec) -> "upstream_bdh.BDHConfig":
    """Translate our BDHSpec into upstream's dataclass, dropping our extras."""
    fields = {f.name for f in dataclasses.fields(upstream_bdh.BDHConfig)}
    return upstream_bdh.BDHConfig(
        **{k: v for k, v in dataclasses.asdict(spec).items() if k in fields}
    )


# --------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------


@dataclasses.dataclass
class BDHActivations:
    """Per-layer latent activations, each ``(B, nh, T, N)``, non-negative.

    ``xy_sparse`` is the elementwise PRODUCT that upstream actually feeds to the
    decoder. It is recorded separately because it is the functionally relevant
    code and it is far sparser than either factor: two codes at ~20% density
    give a conjunction near 4% if they are close to independent. Measuring only
    the factors and comparing them to a Transformer's single hidden layer
    compares the wrong quantity -- which is the mistake this field exists to
    stop repeating. It is also the quantity the BDH paper's "~5% sparsity"
    refers to (Empirical Finding 1, stated for the y vectors of BDH-GPU).
    """

    x_sparse: list[torch.Tensor]
    y_sparse: list[torch.Tensor]
    xy_sparse: list[torch.Tensor] = dataclasses.field(default_factory=list)


class BDHWrapper(nn.Module):
    """Upstream BDH plus optional activation introspection.

    ``forward(idx, targets=None, return_activations=False)`` matches upstream's
    ``(logits, loss)`` return. With ``return_activations=True`` it returns
    ``(logits, loss, BDHActivations)``.
    """

    def __init__(self, spec: BDHSpec) -> None:
        super().__init__()
        self.spec = spec
        self.core = upstream_bdh.BDH(upstream_config(spec))
        # TWO units: upstream has two independent projections (encoder for the
        # query path, encoder_v for the value path). Sharing one unit between
        # them would halve a parameter-owning unit's weights and silently break
        # both the maths and the parameter matching.
        kw = dict(
            n_head=spec.n_head, d=spec.n_embd, N=spec.N,
            **(spec.neuron_unit_kwargs or {}),
        )
        # Per-path overrides. Needed because BDH's output is the PRODUCT
        # x_sparse * y_sparse, so a sparsity constraint on both paths compounds
        # multiplicatively; applying it to one path is the control that
        # separates the constraint's own cost from that compounding.
        kw_x = dict(kw, **(spec.neuron_unit_kwargs_x or {}))
        kw_y = dict(kw, **(spec.neuron_unit_kwargs_y or {}))
        self.neuron_unit_x = build_neuron_unit(spec.neuron_unit, **kw_x)
        self.neuron_unit_y = build_neuron_unit(spec.neuron_unit, **kw_y)

        # Slow state. Absent by default, so every existing result and the
        # upstream-equivalence test are unaffected.
        mem_cfg = dict(spec.persistent_memory or {})
        self.memory = None
        if mem_cfg.pop("enabled", False):
            self.memory = PersistentMemory(
                MemorySpec(enabled=True, **mem_cfg),
                n_head=spec.n_head, N=spec.N, d=spec.n_embd,
            )
        if self.neuron_unit_x.owns_parameters:
            # A unit with its own projection replaces the core's. Leaving
            # upstream's encoder/encoder_v allocated would add two dead
            # (nh, D, N) tensors that are never read -- wasted memory, and it
            # would silently break parameter matching against baseline BDH.
            for name in ("encoder", "encoder_v"):
                if name in self.core._parameters:
                    del self.core._parameters[name]
                    setattr(self.core, name, None)

    # -- convenience passthroughs ------------------------------------------
    @property
    def neuron_unit(self):
        """The query-path unit. Kept for callers that predate the x/y split."""
        return self.neuron_unit_x

    @property
    def config(self):
        return self.core.config

    def generate(self, *a, **kw):
        return self.core.generate(*a, **kw)

    # -- the instrumented forward ------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_activations: bool = False,
        return_hidden: bool = False,
        loss_mask: torch.Tensor | None = None,
    ):
        """Re-expression of upstream ``BDH.forward``; see module docstring.

        ``return_hidden`` additionally returns the final residual stream, the
        representation the LM head reads. A prototype (nearest-class-mean)
        readout needs it, because the point of that readout is to classify
        WITHOUT the learned output layer.

        ``loss_mask`` is ours, not upstream's: a ``(B, T)`` bool/float mask
        selecting which positions contribute to the loss. Step 2B needs it to
        score only the output bytes of a synthetic example. When it is None the
        loss is exactly upstream's mean cross-entropy over all positions.
        """
        core = self.core
        C = core.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = core.embed(idx).unsqueeze(1)
        x = core.ln(x)  # B, 1, T, D

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        xys: list[torch.Tensor] = []

        # One snapshot for the whole forward pass. Every layer reads the state as
        # it stood BEFORE this batch: reading a state that already contains the
        # current batch would leak the target into the prediction, and because
        # BDH reuses one parameter set across layers, a per-layer read of a
        # live-mutating state would leak layer 1's write into layer 2. The read
        # is un-RoPE'd on purpose -- a state carried across sequences cannot
        # live in an absolute-position frame.
        mem_snap = self.memory.snapshot() if self.memory is not None else None

        for _level in range(C.n_layer):
            x_sparse = self.neuron_unit_x(x, core.encoder)  # B, nh, T, N

            yKV = core.attn(Q=x_sparse, K=x_sparse, V=x)
            if self.memory is not None:
                yKV = yKV + self.memory.read(x_sparse, mem_snap)
                if self.training:
                    self.memory.stage(x_sparse, x)
            yKV = core.ln(yKV)

            y_sparse = self.neuron_unit_y(yKV, core.encoder_v)  # B, nh, T, N
            xy_sparse = x_sparse * y_sparse

            if return_activations:
                xs.append(x_sparse.detach())
                ys.append(y_sparse.detach())
                xys.append(xy_sparse.detach())

            xy_sparse = core.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ core.decoder
            )  # B, 1, T, D
            y = core.ln(yMLP)
            x = core.ln(x + y)

        # Apply the staged writes now that every layer has read the snapshot.
        if self.memory is not None and self.training:
            self.memory.commit()

        hidden = x.view(B, T, D)
        logits = hidden @ core.lm_head

        loss = None
        if targets is not None:
            loss = masked_cross_entropy(logits, targets, loss_mask)

        if return_activations:
            return logits, loss, BDHActivations(x_sparse=xs, y_sparse=ys, xy_sparse=xys)
        if return_hidden:
            return logits, loss, hidden.detach()
        return logits, loss


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cross-entropy, optionally over masked positions only.

    With ``loss_mask=None`` this is identical to upstream's
    ``F.cross_entropy(logits.view(-1, V), targets.view(-1))``.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if loss_mask is None:
        return F.cross_entropy(flat_logits, flat_targets)
    m = loss_mask.reshape(-1).to(flat_logits.dtype)
    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    denom = m.sum()
    # An all-masked-out batch would divide by zero; clamp and let the caller
    # decide (it should never happen -- the task builders always emit outputs).
    return (per_tok * m).sum() / denom.clamp(min=1.0)
