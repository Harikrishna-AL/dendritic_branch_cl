"""Branch-granular synaptic consolidation (the user's idea 1).

The problem it targets: under naive sequential training every model tested so
far collapses to the floor (exactly 1/5 on split MNIST, 1/3 on the 3-task
synthetic sequence) because nothing in the loss rewards leaving an old
parameter alone. Replay supplies that pressure implicitly by keeping old tasks
in the objective; this supplies it explicitly, and needs no replay buffer.

The mechanism, per task:

1. After training task t, estimate an **importance** ``omega`` for every unit
   of the latent code, from the trained model on that task's own data.
2. Merge it into a running ``Omega`` across tasks, with decay so that rigidity
   does not accumulate without bound.
3. For every later task, scale the gradient of the encoders and the decoder
   down wherever ``Omega`` is high: ``g <- g / (1 + lam * Omega_hat)``.

Two choices are separated because comparing them *is* the experiment:

``signal``
    ``activity``  -- mean latent activation magnitude on the task. For a
                     dendritic unit this is measured per BRANCH, which is the
                     finer unit only this architecture offers.
    ``synaptic``  -- the norm of BDH's own linear-attention state. Upstream
                     materialises attention as a T x T token matrix
                     (``scores = (QR @ KR.mT).tril(-1)``, bdh.py) and throws it
                     away, but the equivalent state is
                     ``S = sum_t KR[t]^T (x) V[t]`` with shape (N, D) per head:
                     neuron x embedding. Its row norms are a per-neuron
                     importance. This is closest to "the synapses that survive
                     to the end of the task", but note it is inherently
                     per-NEURON: attention runs on the post-soma code, so the
                     branch structure is already pooled away by the time the
                     state exists.

``granularity``
    ``branch`` -- omega has shape (nh, k, N); branch c of neuron j can be
                  frozen while its siblings stay plastic.
    ``neuron`` -- omega has shape (nh, N), broadcast over branches. What a
                  non-dendritic network is limited to.

So ``signal=synaptic, granularity=neuron`` is the literal version of the idea,
and ``signal=activity, granularity=branch`` is the version that uses the
dendrites. If branch granularity does not beat neuron granularity, the
dendritic structure is not buying anything here and that should be reported.

Relation to prior work, so the novelty claim stays honest: importance-weighted
gradients are MAS / SI territory and hard masking is PackNet / HAT. What is new
here is deriving importance from the architecture's own attention state and
applying it at branch granularity. This needs task BOUNDARIES (it estimates at
the end of each task) but never task IDS -- at evaluation nothing is told which
task an input came from.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class ConsolidationSpec:
    enabled: bool = False
    #: how the importance is enforced. THIS MATTERS MORE THAN lam:
    #:
    #: ``scale``   -- multiply the gradient by 1/(1+lam*Omega). Intuitive, and
    #:                MEASURED TO DO NOTHING under AdamW: the update is
    #:                lr * m_hat / sqrt(v_hat), so scaling g by a constant c
    #:                scales m by c and sqrt(v) by c as well and cancels. The
    #:                lam calibration confirmed this -- final accuracy was
    #:                0.197 / 0.199 at lam = 1 / 10, i.e. the naive floor,
    #:                indistinguishable from no consolidation at all.
    #: ``penalty`` -- add (lam/2) * sum Omega_i (theta_i - theta*_i)^2 to the
    #:                loss, the EWC / SI form. Survives Adam because it changes
    #:                the gradient DIRECTION and creates a restoring force
    #:                toward the stored anchor.
    #: ``mask``    -- zero the gradient where Omega exceeds ``threshold``
    #:                (PackNet-style). Also survives Adam, because zero is
    #:                scale-invariantly zero.
    mode: str = "penalty"
    signal: str = "activity"      # activity | synaptic
    granularity: str = "branch"   # branch | neuron
    lam: float = 10.0             # strength; g <- g / (1 + lam * Omega_hat)
    decay: float = 0.9            # Omega <- decay * Omega, before merging a new task
    cap: float | None = None      # protect at most this fraction of units per task
    batches: int = 8              # batches used to estimate omega
    protect_decoder: bool = True  # also shield the neuron -> output mapping

    threshold: float = 0.5        # mode="mask": freeze units with Omega above this

    def __post_init__(self) -> None:
        if self.mode not in ("scale", "penalty", "mask"):
            raise ValueError(f"mode must be scale|penalty|mask, got {self.mode!r}")
        if self.signal not in ("activity", "synaptic"):
            raise ValueError(f"signal must be activity|synaptic, got {self.signal!r}")
        if self.granularity not in ("branch", "neuron"):
            raise ValueError(
                f"granularity must be branch|neuron, got {self.granularity!r}"
            )
        if self.cap is not None and not 0.0 < self.cap <= 1.0:
            raise ValueError(f"cap must be in (0, 1], got {self.cap}")


def _units(model) -> dict[str, torch.nn.Module]:
    """The two latent-code units, keyed by stream, dendritic or not."""
    out = {}
    for stream, attr in (("x_sparse", "neuron_unit_x"), ("y_sparse", "neuron_unit_y")):
        u = getattr(model, attr, None)
        if u is not None:
            out[stream] = u
    return out


class Consolidation:
    """Accumulates importance across tasks and masks gradients with it.

    One instance per run. ``estimate_and_merge`` is called at each task
    boundary; ``mask_grads`` is called every step between ``backward`` and
    ``optimizer.step``.
    """

    def __init__(self, spec: ConsolidationSpec, model, device):
        self.spec = spec
        self.device = device
        self.units = _units(model)
        # k = 1 for a non-dendritic unit, so the branch axis is always present
        # and the two granularities share one code path.
        self.k = {s: int(getattr(u, "k", 1)) for s, u in self.units.items()}
        self.N = {s: int(u.N) if hasattr(u, "N") else None for s, u in self.units.items()}
        self.n_head = {
            s: int(u.n_head) if hasattr(u, "n_head") else None
            for s, u in self.units.items()
        }
        #: stream -> (nh, k, N) running importance, normalised to [0, 1]
        self.omega: dict[str, torch.Tensor] = {}
        #: parameter name -> weights at the last task boundary (penalty target)
        self._anchors: dict[str, torch.Tensor] = {}
        self.history: list[dict] = []

    # ---------------------------------------------------------------- estimate

    @torch.no_grad()
    def _activity(self, model, task, device, rng) -> dict[str, torch.Tensor]:
        """Mean |activation| per (head, branch, neuron) on this task's data."""
        acc: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        grabbed: dict[str, list] = {s: [] for s in self.units}
        handles = []
        for stream, u in self.units.items():
            handles.append(
                u.register_forward_pre_hook(
                    lambda mod, inp, s=stream: grabbed[s].append(inp[0].detach())
                )
            )
        try:
            was_training = model.training
            model.eval()
            batches = task.eval_batches()[: self.spec.batches]
            for b in batches:
                model(b.to(device).x, None)
            model.train(was_training)
        finally:
            for h in handles:
                h.remove()

        for stream, u in self.units.items():
            for x in grabbed[stream]:
                if hasattr(u, "branch_activations"):
                    a = u.branch_activations(x)          # B, nh, T, k, N
                else:
                    # Default unit: relu(x @ weight), one "branch".
                    a = F.relu(x @ u_weight(model, stream)).unsqueeze(-2)
                m = a.abs().mean(dim=(0, 2))             # nh, k, N
                acc[stream] = m if stream not in acc else acc[stream] + m
                counts[stream] = counts.get(stream, 0) + 1
        return {s: v / max(counts[s], 1) for s, v in acc.items()}

    @torch.no_grad()
    def _synaptic(self, model, task, device, rng) -> dict[str, torch.Tensor]:
        """Row norms of BDH's linear-attention state S = sum_t KR[t]^T (x) V[t].

        Upstream never materialises S -- ``tril(-1)`` forces attention through a
        T x T token matrix -- so it is recomputed here from the same inputs the
        attention sees. S is (nh, N, D) per head; ``|S|`` row norms give one
        number per neuron, which is then broadcast over the branch axis because
        attention runs on the post-soma code.
        """
        acc: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        grabbed: dict[str, list] = {s: [] for s in self.units}
        handles = []
        for stream, u in self.units.items():
            handles.append(
                u.register_forward_pre_hook(
                    lambda mod, inp, s=stream: grabbed[s].append(inp[0].detach())
                )
            )
        try:
            was_training = model.training
            model.eval()
            for b in task.eval_batches()[: self.spec.batches]:
                model(b.to(device).x, None)
            model.train(was_training)
        finally:
            for h in handles:
                h.remove()

        for stream, u in self.units.items():
            for x in grabbed[stream]:
                code = u(x)                              # B, nh, T, N (post-soma)
                # V in upstream attention is the residual stream x, shape
                # B, 1, T, D -- broadcast over heads.
                v = x if x.shape[1] == 1 else x.mean(dim=1, keepdim=True)
                # S[h] = sum_t code[h, t]^T v[t]  ->  nh, N, D
                S = torch.einsum("bhtn,bqtd->hnd", code, v)
                row = S.abs().mean(dim=-1)               # nh, N
                m = row.unsqueeze(1).expand(-1, self.k[stream], -1).contiguous()
                acc[stream] = m if stream not in acc else acc[stream] + m
                counts[stream] = counts.get(stream, 0) + 1
        return {s: v / max(counts[s], 1) for s, v in acc.items()}

    def estimate_and_merge(self, model, task, device, rng, *, task_index: int) -> dict:
        """Estimate omega for the task just finished and fold it into Omega."""
        if not self.spec.enabled:
            return {}
        est = (
            self._activity(model, task, device, rng)
            if self.spec.signal == "activity"
            else self._synaptic(model, task, device, rng)
        )
        rec = {"task_index": task_index, "task": task.name, "per_stream": {}}
        for stream, m in est.items():
            if self.spec.granularity == "neuron":
                # Collapse the branch axis, then broadcast back: what a network
                # without branch structure is restricted to.
                m = m.mean(dim=1, keepdim=True).expand_as(m).contiguous()
            hat = _normalise(m)
            if self.spec.cap is not None:
                hat = _cap(hat, self.spec.cap)
            prev = self.omega.get(stream)
            merged = hat if prev is None else torch.maximum(prev * self.spec.decay, hat)
            self.omega[stream] = merged
            rec["per_stream"][stream] = {
                "protected_frac": float((merged > 0.5).float().mean()),
                "omega_mean": float(merged.mean()),
                "omega_max": float(merged.max()),
                "dead_frac": float((merged < 1e-6).float().mean()),
            }
        # Re-anchor: the penalty pulls back to where the weights are NOW, at
        # the end of this task, not to some earlier task's values.
        self.anchor(model)
        rec["mode"] = self.spec.mode
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------- mask grads

    def aligned(self, model):
        """Yield ``(name, param, imp)`` with ``imp`` broadcastable onto ``param``.

        One place that knows how an ``(nh, k, N)`` importance maps onto each
        parameter's layout, so the penalty and the mask cannot disagree about
        it. Raises rather than skipping: a silent skip is what made the plain
        model a no-op the first time round.
        """
        for stream, u in self.units.items():
            om = self.omega.get(stream)
            if om is None:
                continue
            own = getattr(u, "weight", None)
            param = u_weight(model, stream) if own is None else own
            if param is None:
                continue
            nh, k, N = om.shape
            if param.shape[-1] == k * N and k > 1:      # dendritic (nh, D, k*N)
                imp = om.reshape(nh, 1, k * N)
            elif param.shape[-1] == N:                  # (nh, D, N), k == 1 too
                imp = om.mean(dim=1).reshape(nh, 1, N)
            else:
                raise RuntimeError(
                    f"{stream}: cannot align omega {tuple(om.shape)} with "
                    f"parameter {tuple(param.shape)}"
                )
            yield stream, param, imp

        if self.spec.protect_decoder:
            # The soma emits one value per neuron, so the decoder has no branch
            # axis. Shielding it matters: freeze the features but leave the
            # readout free and the neuron -> output mapping drifts anyway.
            core = getattr(model, "core", model)
            dec = getattr(core, "decoder", None)
            om = self.omega.get("x_sparse")
            if dec is not None and om is not None:
                nh, k, N = om.shape
                if dec.shape[0] == nh * N:
                    yield "decoder", dec, om.mean(dim=1).reshape(nh * N, 1)

    def anchor(self, model) -> None:
        """Snapshot the current weights as the point the penalty pulls back to."""
        self._anchors = {
            name: param.detach().clone() for name, param, _ in self.aligned(model)
        }

    def penalty(self, model) -> torch.Tensor | None:
        """``(lam/2) * sum_i Omega_i (theta_i - theta*_i)^2``, or None.

        Added to the loss BEFORE backward, which is what makes it survive an
        adaptive optimizer: it contributes an actual restoring gradient rather
        than merely rescaling the one the data produced.
        """
        if not self.spec.enabled or self.spec.mode != "penalty" or not self.omega:
            return None
        anchors = getattr(self, "_anchors", None)
        if not anchors:
            return None
        total = None
        for name, param, imp in self.aligned(model):
            a = anchors.get(name)
            if a is None:
                continue
            term = (imp * (param - a).pow(2)).sum()
            total = term if total is None else total + term
        return None if total is None else 0.5 * self.spec.lam * total

    @torch.no_grad()
    def mask_grads(self, model) -> None:
        """Enforce the importance on the gradients, for the non-penalty modes.

        Called between ``loss.backward()`` and ``optimizer.step()``, and a no-op
        until the first task boundary so task 0 trains unimpeded.

        ``mode="penalty"`` does its work in the loss instead, so this returns
        immediately for it.
        """
        if not self.spec.enabled or not self.omega:
            return
        if self.spec.mode == "penalty":
            return
        lam = self.spec.lam
        for _name, param, imp in self.aligned(model):
            if param.grad is None:
                continue
            if self.spec.mode == "mask":
                # Hard zero. Survives Adam: zero stays zero through the moment
                # estimates, whereas a constant rescale cancels out.
                param.grad.mul_((imp <= self.spec.threshold).to(param.grad.dtype))
            else:  # "scale" -- kept to document a measured negative result
                param.grad.mul_(1.0 / (1.0 + lam * imp))

    def summary(self) -> dict:
        return {
            "spec": dataclasses.asdict(self.spec),
            "per_task": self.history,
            "final": {
                s: {
                    "protected_frac": float((o > 0.5).float().mean()),
                    "omega_mean": float(o.mean()),
                }
                for s, o in self.omega.items()
            },
        }


# ------------------------------------------------------------------- helpers


def u_weight(model, stream: str):
    """The host projection a non-dendritic unit borrows."""
    core = getattr(model, "core", model)
    return core.encoder if stream == "x_sparse" else core.encoder_v


def _normalise(m: torch.Tensor) -> torch.Tensor:
    """Per head, map importance to [0, 1] by its own max.

    Per head rather than globally because head activation scales differ, and a
    global max would let one loud head decide every other head is unimportant.
    """
    flat = m.reshape(m.shape[0], -1)
    hi = flat.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    return (flat / hi).reshape(m.shape)


def _cap(hat: torch.Tensor, frac: float) -> torch.Tensor:
    """Keep only the top ``frac`` of units per head; zero the rest.

    Without this, importance accumulates monotonically and after a few tasks
    nearly everything is protected and nothing new can be learned -- the
    standard failure mode of PackNet / HAT-style masking.
    """
    nh = hat.shape[0]
    flat = hat.reshape(nh, -1)
    n_keep = max(1, int(round(frac * flat.shape[1])))
    thresh = flat.topk(n_keep, dim=1).values[:, -1:].contiguous()
    return (flat * (flat >= thresh)).reshape(hat.shape)


def build_consolidation(cfg: dict | None, model, device) -> Consolidation | None:
    """From a config block; ``None`` when the run does not use consolidation."""
    if not cfg or not cfg.get("enabled", False):
        return None
    spec = ConsolidationSpec(
        **{k: v for k, v in cfg.items() if k in {f.name for f in dataclasses.fields(ConsolidationSpec)}}
    )
    return Consolidation(spec, model, device)
