"""Sparsity and neuron-overlap diagnostics (Step 3, BDH-specific part).

The hypothesis under test -- *not* assumed -- is that BDH's sparse, positive,
high-dimensional latent code gives different tasks more separated neuron sets
than a dense Transformer, and therefore less interference.

To test it we need numbers that mean the same thing for both architectures.
For BDH the natural unit is a latent neuron: ``x_sparse``/``y_sparse`` have
shape ``(B, nh, T, N)`` and are non-negative by construction, so "active"
means ``> 0``. For the GPT baseline the closest analogue is the post-GELU MLP
hidden state ``(B, T, 4*n_embd)``, which is dense and signed. We therefore
define activity as ``> tol`` for both and report the resulting density; the
fact that GPT's density comes out near 50% while BDH's is low is itself part of
the result, not a bug in the probe.

Three families of measurement:

``activation_stats``  per-layer density (% positive) and mean activation.
``top_k_sets``        the identity of the most-active neurons per layer, which
                      is what "this task uses these neurons" means.
``overlap``           Jaccard similarity of two tasks' top-k sets, plus a
                      cosine similarity on the full mean-activation profile,
                      which is less sensitive to the arbitrary choice of k.
``repurposing``       whether neurons that were highly active for an earlier
                      task are still active later, or have been taken over.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch


def _as_neuron_matrix(act: torch.Tensor) -> torch.Tensor:
    """Flatten an activation tensor to ``(tokens, neurons)``.

    BDH: ``(B, nh, T, N)`` -> ``(B*T, nh*N)``, keeping head-major neuron
    identity so neuron index ``h*N + j`` is stable across calls.
    GPT:  ``(B, T, H)``    -> ``(B*T, H)``.
    """
    if act.dim() == 4:
        B, nh, T, N = act.shape
        return act.permute(0, 2, 1, 3).reshape(B * T, nh * N)
    if act.dim() == 3:
        B, T, H = act.shape
        return act.reshape(B * T, H)
    raise ValueError(f"unexpected activation shape {tuple(act.shape)}")


@dataclasses.dataclass
class LayerProfile:
    """Accumulated activation statistics for one layer of one model on one task."""

    n_neurons: int
    n_tokens: int = 0
    active_count: np.ndarray | None = None  # per-neuron count of active tokens
    act_sum: np.ndarray | None = None  # per-neuron sum of activation
    active_total: int = 0  # total active (neuron, token) pairs

    def update(self, act: torch.Tensor, tol: float = 0.0) -> None:
        m = _as_neuron_matrix(act).detach().float().cpu()
        if self.active_count is None:
            self.active_count = np.zeros(m.shape[1], dtype=np.int64)
            self.act_sum = np.zeros(m.shape[1], dtype=np.float64)
        alive = (m > tol)
        self.active_count += alive.sum(0).numpy().astype(np.int64)
        self.act_sum += m.sum(0).double().numpy()
        self.active_total += int(alive.sum().item())
        self.n_tokens += m.shape[0]

    # -- derived -----------------------------------------------------------
    @property
    def density(self) -> float:
        """Fraction of (neuron, token) pairs that are active. Sparsity = 1 - this."""
        denom = self.n_tokens * self.n_neurons
        return self.active_total / denom if denom else float("nan")

    @property
    def mean_activation(self) -> np.ndarray:
        """Per-neuron mean activation over tokens -- the task's 'profile'."""
        return self.act_sum / max(self.n_tokens, 1)

    @property
    def usage_rate(self) -> np.ndarray:
        """Per-neuron fraction of tokens on which it was active."""
        return self.active_count / max(self.n_tokens, 1)

    @property
    def dead_fraction(self) -> float:
        """Neurons never active on this task."""
        return float((self.active_count == 0).mean())

    def top_k(self, k: int) -> np.ndarray:
        """Indices of the k neurons with the highest mean activation."""
        k = min(k, self.n_neurons)
        return np.argsort(-self.mean_activation)[:k]

    def to_dict(self, k: int) -> dict:
        return {
            "n_neurons": int(self.n_neurons),
            "n_tokens": int(self.n_tokens),
            "density": self.density,
            "sparsity": 1.0 - self.density,
            "dead_fraction": self.dead_fraction,
            "mean_activation_overall": float(self.mean_activation.mean()),
            "top_k": self.top_k(k).tolist(),
        }


@dataclasses.dataclass
class ModelProfile:
    """One model's activation profile on one task, across all layers/streams.

    ``streams`` is ``{"x_sparse": [...per layer], "y_sparse": [...]}`` for BDH
    and ``{"mlp_hidden": [...]}`` for GPT.
    """

    streams: dict[str, list[LayerProfile]] = dataclasses.field(default_factory=dict)

    def _ensure(self, name: str, acts: list[torch.Tensor]) -> None:
        if name not in self.streams:
            self.streams[name] = [
                LayerProfile(n_neurons=_as_neuron_matrix(a).shape[1]) for a in acts
            ]

    def update(self, activations, tol: float = 0.0) -> None:
        """Accumulate one batch. Accepts BDHActivations or GPTActivations."""
        for name, acts in _streams_of(activations):
            self._ensure(name, acts)
            for layer, a in zip(self.streams[name], acts):
                layer.update(a, tol=tol)

    def to_dict(self, k: int = 256) -> dict:
        return {
            name: {
                "per_layer": [lp.to_dict(k) for lp in layers],
                "mean_density": float(np.mean([lp.density for lp in layers])),
                "mean_sparsity": float(1.0 - np.mean([lp.density for lp in layers])),
            }
            for name, layers in self.streams.items()
        }


def _streams_of(activations):
    """Normalise the two activation dataclasses into (name, list) pairs."""
    if hasattr(activations, "x_sparse"):
        out = [("x_sparse", activations.x_sparse), ("y_sparse", activations.y_sparse)]
        # The product is what reaches the decoder, and it is much sparser than
        # either factor. Included only when present so older callers are
        # unaffected.
        if getattr(activations, "xy_sparse", None):
            out.append(("xy_sparse", activations.xy_sparse))
        return out
    if hasattr(activations, "hidden"):
        return [("mlp_hidden", activations.hidden)]
    raise TypeError(f"unrecognised activations object {type(activations)!r}")


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


@torch.no_grad()
def collect_profile(
    model,
    batches,
    device,
    *,
    tol: float = 0.0,
    max_batches: int | None = None,
) -> ModelProfile:
    """Run the model over ``batches`` and accumulate an activation profile.

    Runs in eval mode so dropout does not distort the density estimate; the
    caller's train/eval state is restored.
    """
    was_training = model.training
    model.eval()
    prof = ModelProfile()
    try:
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            b = batch.to(device)
            _, _, acts = model(b.x, None, return_activations=True)
            prof.update(acts, tol=tol)
    finally:
        model.train(was_training)
    return prof


# --------------------------------------------------------------------------
# Comparisons between tasks
# --------------------------------------------------------------------------


def jaccard(a, b) -> float:
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def overlap(
    prof_a: ModelProfile,
    prof_b: ModelProfile,
    *,
    k: int = 256,
    k_frac: float | None = None,
) -> dict:
    """Per-stream, per-layer overlap between two tasks' neuron usage.

    ``k_frac`` (e.g. 0.05) sets k as a fraction of the neuron count, which
    makes BDH and GPT comparable despite very different widths; it overrides
    ``k`` when given.
    """
    out: dict[str, dict] = {}
    for name, layers_a in prof_a.streams.items():
        layers_b = prof_b.streams.get(name)
        if layers_b is None:
            continue
        jac, cos = [], []
        for la, lb in zip(layers_a, layers_b):
            kk = max(1, int(round(k_frac * la.n_neurons))) if k_frac else k
            jac.append(jaccard(la.top_k(kk), lb.top_k(kk)))
            cos.append(cosine(la.mean_activation, lb.mean_activation))
        out[name] = {
            "k": (max(1, int(round(k_frac * layers_a[0].n_neurons))) if k_frac else k),
            "k_frac": k_frac,
            "jaccard_per_layer": jac,
            "jaccard_mean": float(np.nanmean(jac)),
            "cosine_per_layer": cos,
            "cosine_mean": float(np.nanmean(cos)),
        }
    return out


def repurposing(
    prof_before: ModelProfile,
    prof_after: ModelProfile,
    *,
    k_frac: float = 0.05,
) -> dict:
    """Do the neurons that served an earlier task get taken over later?

    ``prof_before``/``prof_after`` are profiles of *the same task's data* under
    the model at two points in the sequence (right after learning that task,
    and at the end). For the earlier task's top-k neurons we report:

    * ``retained``  -- fraction still in the top-k at the end
    * ``activity_ratio`` -- their mean activation at the end / at the time
    * ``drift`` -- 1 - cosine between the two mean-activation profiles

    A high ``retained`` with a big drop in ``activity_ratio`` means the neurons
    were silenced; a low ``retained`` means they were reassigned.
    """
    out: dict[str, dict] = {}
    for name, before in prof_before.streams.items():
        after = prof_after.streams.get(name)
        if after is None:
            continue
        retained, ratio, drift = [], [], []
        for lb, la in zip(before, after):
            kk = max(1, int(round(k_frac * lb.n_neurons)))
            top = lb.top_k(kk)
            retained.append(jaccard(top, la.top_k(kk)))
            m_before = lb.mean_activation[top].mean()
            m_after = la.mean_activation[top].mean()
            ratio.append(float(m_after / m_before) if m_before > 0 else float("nan"))
            drift.append(1.0 - cosine(lb.mean_activation, la.mean_activation))
        out[name] = {
            "k_frac": k_frac,
            "retained_per_layer": retained,
            "retained_mean": float(np.nanmean(retained)),
            "activity_ratio_per_layer": ratio,
            "activity_ratio_mean": float(np.nanmean(ratio)),
            "drift_per_layer": drift,
            "drift_mean": float(np.nanmean(drift)),
        }
    return out


def overlap_matrix(profiles: dict[str, ModelProfile], *, k_frac: float = 0.05) -> dict:
    """All-pairs overlap between tasks, ready to plot as a heatmap."""
    names = list(profiles)
    streams = set()
    for p in profiles.values():
        streams |= set(p.streams)
    result: dict[str, dict] = {}
    for s in sorted(streams):
        J = [[float("nan")] * len(names) for _ in names]
        C = [[float("nan")] * len(names) for _ in names]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                o = overlap(profiles[a], profiles[b], k_frac=k_frac).get(s)
                if o:
                    J[i][j] = o["jaccard_mean"]
                    C[i][j] = o["cosine_mean"]
        result[s] = {"task_names": names, "jaccard": J, "cosine": C, "k_frac": k_frac}
    return result


# --------------------------------------------------------------------------
# Dendritic branch diagnostics
# --------------------------------------------------------------------------


@dataclasses.dataclass
class BranchProfile:
    """Per-neuron branch usage for one task, one stream, one layer.

    ``win_counts[j, c]`` counts tokens where branch ``c`` was the most active
    branch of neuron ``j``; ``act_sum[j, c]`` accumulates branch activation.
    Everything else is derived from those two.
    """

    n_neurons: int
    k: int
    win_counts: np.ndarray | None = None
    act_sum: np.ndarray | None = None
    n_tokens: int = 0
    alive_pairs: int = 0  # (token, neuron) pairs where at least one branch fired

    def update(self, branches: torch.Tensor) -> None:
        """``branches``: (B, nh, T, k, N), non-negative."""
        B, nh, T, k, N = branches.shape
        # (tokens, k, neurons) with neuron index h*N + j, matching _as_neuron_matrix
        m = branches.permute(0, 2, 3, 1, 4).reshape(B * T, k, nh * N).float().cpu()
        if self.win_counts is None:
            self.win_counts = np.zeros((nh * N, k), dtype=np.int64)
            self.act_sum = np.zeros((nh * N, k), dtype=np.float64)
        winner = m.argmax(dim=1)  # (tokens, neurons)
        any_on = (m.amax(dim=1) > 0)
        for c in range(k):
            sel = (winner == c) & any_on
            self.win_counts[:, c] += sel.sum(0).numpy().astype(np.int64)
        self.act_sum += m.sum(0).T.double().numpy()
        self.n_tokens += B * T
        self.alive_pairs += int(any_on.sum().item())

    @property
    def usage_share(self) -> np.ndarray:
        """Fraction of wins taken by each branch, pooled over neurons."""
        tot = self.win_counts.sum()
        return self.win_counts.sum(0) / tot if tot else np.full(self.k, np.nan)

    @property
    def dominant_branch(self) -> np.ndarray:
        """Per neuron, the branch that wins most often."""
        return self.win_counts.argmax(axis=1)

    @property
    def branch_entropy(self) -> np.ndarray:
        """Per-neuron entropy (nats) over its branch-win distribution.

        0 means the neuron always uses one branch; log(k) means it spreads
        evenly. This is the "are branches specialising at all?" number.
        """
        p = self.win_counts / np.maximum(self.win_counts.sum(1, keepdims=True), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
        return e

    @property
    def dead_branch_fraction(self) -> float:
        """Fraction of (neuron, branch) pairs that never won a single token."""
        return float((self.win_counts == 0).mean())

    def to_dict(self) -> dict:
        ent = self.branch_entropy
        return {
            "n_neurons": int(self.n_neurons),
            "k": int(self.k),
            "usage_share": self.usage_share.tolist(),
            "usage_share_max": float(self.usage_share.max()),
            "entropy_mean": float(np.mean(ent)),
            "entropy_max_possible": float(np.log(self.k)),
            "entropy_normalised": float(np.mean(ent) / np.log(self.k)) if self.k > 1 else 0.0,
            "single_branch_neuron_frac": float((ent < 1e-6).mean()),
            "dead_branch_fraction": self.dead_branch_fraction,
            # (token, neuron) pairs with any branch firing, over all such pairs.
            "alive_frac": self.alive_pairs / max(self.n_tokens * self.n_neurons, 1),
        }


def _dendritic_units(model) -> dict[str, object]:
    """The parameter-owning neuron units of a wrapper, by stream name."""
    out = {}
    for attr, stream in (("neuron_unit_x", "x_sparse"), ("neuron_unit_y", "y_sparse")):
        u = getattr(model, attr, None)
        if u is not None and getattr(u, "owns_parameters", False) and hasattr(u, "capture"):
            out[stream] = u
    return out


@torch.no_grad()
def collect_branch_profile(model, batches, device, *, max_batches: int | None = 2) -> dict:
    """Branch-usage profile per stream and layer.

    Capture is enabled only for this call: the branch tensor is ``k`` times the
    size of the layer output, so it is not something to hold during training.
    """
    units = _dendritic_units(model)
    if not units:
        return {}

    was_training = model.training
    model.eval()
    profiles: dict[str, list[BranchProfile]] = {}
    try:
        for u in units.values():
            u.capture = True
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            b = batch.to(device)
            # One forward per layer is not possible; the units are reused across
            # layers, so last_branches holds the FINAL layer's activations. We
            # therefore report per-stream (all layers pooled through reuse),
            # which is the right granularity for weight-shared BDH anyway.
            model(b.x, None)
            for stream, u in units.items():
                a = u.last_branches
                if a is None:
                    continue
                if stream not in profiles:
                    B, nh, T, k, N = a.shape
                    profiles[stream] = [BranchProfile(n_neurons=nh * N, k=k)]
                profiles[stream][0].update(a)
                u.last_branches = None
    finally:
        for u in units.values():
            u.capture = False
            u.last_branches = None
        model.train(was_training)

    return {s: p[0] for s, p in profiles.items()}


def branch_overlap(prof_a: dict, prof_b: dict) -> dict:
    """Do two tasks drive the *same* branch of the same neuron?

    This is the question the dendritic hypothesis turns on. If branches
    specialise by task, the dominant branch of a given neuron should differ
    between tasks, and ``same_branch_frac`` will be low. If the neuron always
    routes to the same branch regardless of input, it will be near 1 and the
    branches are not doing task-specific work.

    ``chance`` (1/k) is a *weak* reference: even at random initialisation a
    neuron has a preferred branch, because the branch weight norms differ, so
    an untrained model already scores ~0.65 at k=4. The meaningful comparison
    is a trained model against its own random-init value, which is why
    ``scripts/run_cl.py`` records the branch profile before training as well.
    """
    out = {}
    for stream, a in prof_a.items():
        b = prof_b.get(stream)
        if b is None:
            continue
        da, db = a.dominant_branch, b.dominant_branch
        # Only neurons that actually fired for both tasks are comparable.
        live = (a.win_counts.sum(1) > 0) & (b.win_counts.sum(1) > 0)
        n_live = int(live.sum())
        same = float((da[live] == db[live]).mean()) if n_live else float("nan")
        out[stream] = {
            "same_branch_frac": same,
            "chance": 1.0 / a.k,
            "excess_over_chance": same - 1.0 / a.k if n_live else float("nan"),
            "n_live_neurons": n_live,
            "n_neurons": int(a.n_neurons),
        }
    return out
