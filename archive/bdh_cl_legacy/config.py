"""Config presets and YAML loading for the BDH continual-learning harness.

Three model-size presets (Step 0.4):

  tiny  -- CPU / laptop smoke tests
  small -- the upstream repo default (BDHConfig())
  base  -- larger; only usable if the hardware allows

A note on BDH parameter counts. Upstream ``bdh.py`` allocates ``encoder``,
``encoder_v`` and ``decoder`` *once* and reuses them at every level of the
layer loop, so ``n_layer`` does not change BDH's parameter count -- only its
depth of computation. See :mod:`bdh_cl.models.param_accounting`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent


@dataclasses.dataclass
class BDHSpec:
    """Mirror of upstream ``bdh.BDHConfig`` plus our own extras."""

    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    # Ours: which NeuronUnit to use (Step 7). "default" == upstream maths.
    neuron_unit: str = "default"
    # Extra kwargs for a parameter-owning unit, e.g.
    # {"n_branches": 4, "soma": "max"} for the dendritic unit.
    neuron_unit_kwargs: dict | None = None
    # Per-path OVERRIDES, merged over neuron_unit_kwargs for that path only.
    # BDH's output is the product x_sparse * y_sparse, so a constraint applied
    # to both paths compounds: two 5%-dense codes give a 0.25%-dense
    # conjunction. These let a constraint be applied to ONE path, which is the
    # control that separates "sparsity hurts" from "sparsity compounds through
    # the multiplicative gate".
    neuron_unit_kwargs_x: dict | None = None
    neuron_unit_kwargs_y: dict | None = None
    # Slow synaptic state that survives across sequences (milestone 1 of the
    # persistent-memory direction). None or {"enabled": False} leaves the model
    # exactly as upstream.
    persistent_memory: dict | None = None

    @property
    def N(self) -> int:
        """Per-head latent width, as upstream computes it."""
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head

    @property
    def n_neurons(self) -> int:
        """Total latent neurons n = n_head * N."""
        return self.n_head * self.N


@dataclasses.dataclass
class GPTSpec:
    """nanoGPT-style baseline. RoPE instead of learned position embeddings so
    that the parameter count does not depend on block_size."""

    n_layer: int = 8
    n_embd: int = 128
    n_head: int = 4
    dropout: float = 0.1
    vocab_size: int = 256
    bias: bool = False


@dataclasses.dataclass
class TrainSpec:
    block_size: int = 256
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    steps_per_task: int = 400
    warmup_steps: int = 50
    eval_every: int = 100
    eval_batches: int = 20
    log_every: int = 50
    # Reset AdamW state between tasks (default per the brief); the
    # "keep state" variant flips this.
    reset_optimizer_between_tasks: bool = True
    seed: int = 0


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "bdh": dict(n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=32),
        # 8 layers x d=128 matches BDH-tiny's 1.64M within ~0.1%.
        "gpt": dict(n_layer=8, n_embd=128, n_head=4),
        "train": dict(block_size=256, batch_size=16, steps_per_task=400),
    },
    "small": {
        # The upstream default.
        "bdh": dict(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128),
        # d=512 x 8 layers ~= 25.4M vs BDH 25.3M (+0.4%).
        "gpt": dict(n_layer=8, n_embd=512, n_head=8),
        "train": dict(block_size=256, batch_size=8, steps_per_task=1000),
    },
    "base": {
        "bdh": dict(n_layer=6, n_embd=512, n_head=8, mlp_internal_dim_multiplier=128),
        # d=896 x 10 layers ~= 96.8M vs BDH 100.9M (-4.1%).
        "gpt": dict(n_layer=10, n_embd=896, n_head=14),
        "train": dict(block_size=256, batch_size=4, steps_per_task=2000),
    },
}


def preset(name: str) -> tuple[BDHSpec, GPTSpec, TrainSpec]:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    p = PRESETS[name]
    return (
        BDHSpec(**p["bdh"]),
        GPTSpec(**p["gpt"]),
        TrainSpec(**p["train"]),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML: a preset name plus per-section overrides."""
    cfg = load_yaml(path)
    bdh_spec, gpt_spec, train_spec = preset(cfg.get("preset", "tiny"))
    for spec, key in ((bdh_spec, "bdh"), (gpt_spec, "gpt"), (train_spec, "train")):
        for k, v in (cfg.get(key) or {}).items():
            if not hasattr(spec, k):
                raise KeyError(f"{key}: unknown field {k!r}")
            setattr(spec, k, v)
    cfg["bdh_spec"] = bdh_spec
    cfg["gpt_spec"] = gpt_spec
    cfg["train_spec"] = train_spec

    # Optional dendritic variant: starts from the BDH preset, then applies the
    # `dendritic:` overrides (typically a smaller mlp multiplier so that the
    # k-branch encoders stay parameter-matched to baseline).
    import copy

    def _den(overrides: dict):
        d = copy.deepcopy(bdh_spec)
        for key, val in overrides.items():
            if not hasattr(d, key):
                raise KeyError(f"dendritic: unknown field {key!r}")
            setattr(d, key, val)
        if "neuron_unit" not in overrides:
            d.neuron_unit = "dendritic"
        return d

    if cfg.get("dendritic"):
        cfg["dendritic_spec"] = _den(cfg["dendritic"])
    # Several variants in one sweep, e.g. {"gated": {...}, "max": {...}},
    # each exposed as model "bdh_den_<name>".
    if cfg.get("narrow"):
        n = copy.deepcopy(bdh_spec)
        for key, val in cfg["narrow"].items():
            if not hasattr(n, key):
                raise KeyError(f"narrow: unknown field {key!r}")
            setattr(n, key, val)
        cfg["narrow_spec"] = n
    if cfg.get("dendritic_variants"):
        cfg["dendritic_specs"] = {
            name: _den(ov) for name, ov in cfg["dendritic_variants"].items()
        }

    # Persistent-memory variants, exposed as model "bdh_mem_<name>". Built off
    # the baseline spec rather than the dendritic one: milestone 1 deliberately
    # has no dendrites, so that persistent memory is measured on its own before
    # anything is layered on top of it.
    def _mem(overrides: dict):
        m = copy.deepcopy(bdh_spec)
        for key, val in overrides.items():
            if not hasattr(m, key):
                raise KeyError(f"memory: unknown field {key!r}")
            setattr(m, key, val)
        if not (m.persistent_memory or {}).get("enabled"):
            raise ValueError("memory variant must set persistent_memory.enabled")
        return m

    if cfg.get("memory_variants"):
        cfg["memory_specs"] = {
            name: _mem(ov) for name, ov in cfg["memory_variants"].items()
        }

    # THE control for the dendritic claim: same parameters and same neuron count,
    # spent on nonlinear DEPTH instead of per-neuron branch grouping. Exposed as
    # model "bdh_deep".
    if cfg.get("deep"):
        dp = copy.deepcopy(bdh_spec)
        for key, val in cfg["deep"].items():
            if not hasattr(dp, key):
                raise KeyError(f"deep: unknown field {key!r}")
            setattr(dp, key, val)
        dp.neuron_unit = "deep"
        cfg["deep_spec"] = dp
    return cfg


def as_dict(obj: Any) -> Any:
    """Recursively make dataclasses/paths JSON-serialisable."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: as_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
