"""Optional Weights & Biases logging.

Two hard rules, both learned the hard way in this project:

1. **Logging must never kill a run.** Every call is wrapped; if wandb is
   missing, unauthenticated, offline, or throws mid-run, we degrade to a no-op
   and training continues. A dead logger must not cost hours of GPU time.
2. **Logging must not add GPU syncs.** Calling ``.item()`` on a device tensor
   every step is what hung an earlier run on MPS. Callers pass values that are
   already on the host, sampled at log boundaries.

Disabled by default; enabled per-experiment from YAML::

    wandb:
      enabled: true
      project: bdh-cl
      group: exp3_original_config
"""

from __future__ import annotations

import os
from typing import Any

_RUN = None
_ENABLED = False
_FAILED = False


def _safe(fn):
    """Never let a logging failure propagate into training."""
    def wrapper(*a, **kw):
        global _FAILED
        if not _ENABLED or _FAILED:
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - deliberately broad
            _FAILED = True
            print(f"  [wandb] disabled after error: {type(e).__name__}: {e}", flush=True)
            return None
    return wrapper


def init(cfg: dict | None, *, name: str, config: dict, tags=None) -> bool:
    """Start a run. Returns whether logging is active."""
    global _RUN, _ENABLED, _FAILED
    _RUN, _ENABLED, _FAILED = None, False, False
    cfg = cfg or {}
    if not cfg.get("enabled"):
        return False
    try:
        import wandb
    except ImportError:
        print("  [wandb] not installed; logging disabled", flush=True)
        return False
    kwargs = dict(
            project=cfg.get("project", "bdh-cl"),
            entity=cfg.get("entity"),
            group=cfg.get("group"),
            job_type=cfg.get("job_type"),
            name=name,
            config=config,
            tags=tags or [],
            mode=cfg.get("mode", os.environ.get("WANDB_MODE", "online")),
            reinit=True,
    )
    try:
        _RUN = wandb.init(**{k: v for k, v in kwargs.items() if v is not None})
        _ENABLED = True
        print(f"  [wandb] logging to {_RUN.url}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [wandb] init failed ({type(e).__name__}: {e}); continuing without", flush=True)
        return False


@_safe
def log(data: dict[str, Any], step: int | None = None) -> None:
    _RUN.log(data, step=step)


@_safe
def summary(data: dict[str, Any]) -> None:
    for k, v in data.items():
        _RUN.summary[k] = v


@_safe
def log_matrix(name: str, matrix, task_names: list[str], step: int | None = None) -> None:
    """Log the accuracy matrix as a table, so it renders live in the UI."""
    import wandb

    cols = ["after_task"] + list(task_names)
    table = wandb.Table(columns=cols)
    for i, row in enumerate(matrix):
        label = task_names[i] if i < len(task_names) else str(i)
        table.add_data(label, *[None if v != v else float(v) for v in row])
    _RUN.log({name: table}, step=step)


@_safe
def log_image(name: str, path, step: int | None = None) -> None:
    import wandb

    _RUN.log({name: wandb.Image(str(path))}, step=step)


@_safe
def finish() -> None:
    global _RUN, _ENABLED
    _RUN.finish()
    _RUN, _ENABLED = None, False


def active() -> bool:
    return _ENABLED and not _FAILED
