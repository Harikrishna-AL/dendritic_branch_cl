"""Parameter accounting for BDH and its baselines (Step 7).

Two jobs:

1. Give an *analytic* parameter formula for each architecture, so future
   dendritic variants can be sized on paper before anything is built.
2. Search for a GPT config whose parameter count matches a given BDH config
   within a tolerance, which is what the Step 3 comparison needs.

BDH's analytic count::

    encoder    : n_head * n_embd * N        = mult * n_embd^2
    encoder_v  : n_head * n_embd * N        = mult * n_embd^2
    decoder    : (n_head * N) * n_embd      = mult * n_embd^2
    embed      : vocab * n_embd
    lm_head    : n_embd * vocab
    ln         : 0  (elementwise_affine=False)
    ------------------------------------------------------------
    total      = 3 * mult * n_embd^2 + 2 * vocab * n_embd
               = 3 * n * d + 2 * vocab * d,  with n = n_head*N = mult*d

``n_layer`` does not appear: upstream shares the three weight matrices across
all levels of the layer loop.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from ..config import BDHSpec, GPTSpec


# --------------------------------------------------------------------------
# Analytic counts
# --------------------------------------------------------------------------


def bdh_param_count(spec: BDHSpec) -> dict[str, int]:
    d = spec.n_embd
    n = spec.n_neurons  # n_head * N == mult * d
    parts = {
        "encoder": spec.n_head * d * spec.N,
        "encoder_v": spec.n_head * d * spec.N,
        "decoder": n * d,
        "embed": spec.vocab_size * d,
        "lm_head": d * spec.vocab_size,
    }
    parts["total"] = sum(parts.values())
    return parts


def gpt_param_count(spec: GPTSpec) -> dict[str, int]:
    """nanoGPT-style blocks with RoPE (no wpe) and no weight tying.

    Per block: qkv (3d^2) + attn out (d^2) + mlp (4d^2 up, 4d^2 down) = 12 d^2,
    plus two LayerNorm weight vectors (2d) and, if bias=True, the bias terms.
    """
    d = spec.n_embd
    per_block = 12 * d * d
    per_block += 2 * d  # 2x LayerNorm weight
    if spec.bias:
        # qkv bias 3d, attn proj d, mlp up 4d, mlp down d, 2x LN bias 2d
        per_block += 3 * d + d + 4 * d + d + 2 * d
    parts = {
        "blocks": spec.n_layer * per_block,
        "embed": spec.vocab_size * d,
        "ln_f": d + (d if spec.bias else 0),
        "lm_head": d * spec.vocab_size,
    }
    parts["total"] = sum(parts.values())
    return parts


def count_module(module) -> dict[str, int]:
    """Empirical count from an instantiated nn.Module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "per_tensor": {n: p.numel() for n, p in module.named_parameters()},
    }


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match_gpt_to_bdh(
    bdh_spec: BDHSpec,
    *,
    tol: float = 0.10,
    n_layer_choices: Iterable[int] = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32),
    n_embd_choices: Iterable[int] = (
        64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024,
    ),
    head_dim: int = 64,
    prefer: str = "aspect",
) -> tuple[GPTSpec, dict]:
    """Find a GPTSpec within ``tol`` of ``bdh_spec``'s parameter count.

    ``prefer`` picks among the candidates that fit:
      * ``"aspect"``  -- the most GPT-2-like depth:width ratio (d / n_layer ~ 64)
      * ``"exact"``   -- the smallest relative parameter error
      * ``"width"``   -- the candidate whose n_embd equals BDH's n_embd, if any
    """
    target = bdh_param_count(bdh_spec)["total"]
    candidates = []
    for n_layer in n_layer_choices:
        for d in n_embd_choices:
            n_head = max(1, d // head_dim)
            if d % n_head:
                continue
            spec = GPTSpec(
                n_layer=n_layer,
                n_embd=d,
                n_head=n_head,
                dropout=bdh_spec.dropout,
                vocab_size=bdh_spec.vocab_size,
            )
            total = gpt_param_count(spec)["total"]
            rel = (total - target) / target
            if abs(rel) <= tol:
                candidates.append((spec, total, rel))

    if not candidates:
        raise RuntimeError(
            f"no GPT config within {tol:.0%} of BDH's {target:,} parameters"
        )

    if prefer == "exact":
        key = lambda c: abs(c[2])
    elif prefer == "width":
        key = lambda c: (c[0].n_embd != bdh_spec.n_embd, abs(c[2]))
    else:  # "aspect"
        key = lambda c: (abs(c[0].n_embd / c[0].n_layer - 64.0), abs(c[2]))

    spec, total, rel = min(candidates, key=key)
    report = {
        "bdh_total": target,
        "gpt_total": total,
        "rel_error": rel,
        "within_tol": abs(rel) <= tol,
        "tol": tol,
        "prefer": prefer,
        "n_candidates": len(candidates),
    }
    return spec, report


def compare(bdh_spec: BDHSpec, gpt_spec: GPTSpec) -> dict:
    b = bdh_param_count(bdh_spec)
    g = gpt_param_count(gpt_spec)
    rel = (g["total"] - b["total"]) / b["total"]
    return {
        "bdh": {"spec": dataclasses.asdict(bdh_spec), "params": b},
        "gpt": {"spec": dataclasses.asdict(gpt_spec), "params": g},
        "rel_error_gpt_vs_bdh": rel,
        "matched_within_10pct": abs(rel) <= 0.10,
    }


# --------------------------------------------------------------------------
# Forward-looking: budget for a dendritic variant (Step 7, no implementation)
# --------------------------------------------------------------------------


def dendritic_budget(spec: BDHSpec, n_branches: int) -> dict:
    """What a k-branch dendritic BDH would cost, and how to pay for it.

    The intended shape (see ``neuron_units.DendriticNeuronUnit``) gives every
    latent neuron ``k`` branches with their own input weights, so the two
    *encoder* matrices grow by a factor of ``k`` while ``decoder`` -- which
    reads the soma output, one value per neuron -- does not.

    To stay parameter-matched with baseline BDH you shrink the neuron count:
    with ``n' = n * 3 / (2k + 1)`` the totals agree. This function reports
    that equal-parameter neuron count and the multiplier it implies for
    ``mlp_internal_dim_multiplier``.
    """
    base = bdh_param_count(spec)
    d, n = spec.n_embd, spec.n_neurons
    dendritic_backbone = (2 * n_branches + 1) * n * d
    matched_n = int(round(3 * n / (2 * n_branches + 1)))
    return {
        "n_branches": n_branches,
        "baseline_total": base["total"],
        "naive_dendritic_total": dendritic_backbone + 2 * spec.vocab_size * d,
        "naive_blowup": dendritic_backbone / (3 * n * d),
        "param_matched_n_neurons": matched_n,
        "param_matched_mult": matched_n / d,
        "note": (
            "encoder and encoder_v scale with k (per-branch input weights); "
            "decoder reads one soma output per neuron and does not."
        ),
    }
