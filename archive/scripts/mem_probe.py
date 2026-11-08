"""Peak-memory probe: one forward+backward at a real config, nothing else.

Baseline BDH already needs ~37 GiB of activation memory at upstream's
B=32 / T=512, and a dendritic unit holds a branch tensor k times the size of
the latent it pools. So a matched dendritic model may not fit even though its
PARAMETER count is matched. Two minutes here beats discovering it three hours
into an allocation, and the number tells us which soma is affordable: `gated`
additionally materialises the softmax weights, doubling the branch tensor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.config import BDHSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.param_accounting import count_module  # noqa: E402
from bdh_cl.scripts.sparsity_original import (  # noqa: E402
    DENDRITIC_BRANCHES, DENDRITIC_MULT, UPSTREAM_BDH,
)


def build(model: str):
    if model == "bdh":
        return BDHWrapper(BDHSpec(**UPSTREAM_BDH))
    soma = model.rsplit("_", 1)[1]
    return BDHWrapper(BDHSpec(
        **{**UPSTREAM_BDH, "mlp_internal_dim_multiplier": DENDRITIC_MULT},
        neuron_unit="dendritic",
        neuron_unit_kwargs={"n_branches": DENDRITIC_BRANCHES, "soma": soma},
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bdh,bdh_den_sum,bdh_den_max,bdh_den_gated")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=512)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev.type == "cuda" else 0
    print(f"device={dev} total={total:.1f} GiB  B={args.batch_size} T={args.block_size}")

    for name in args.models.split(","):
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            m = build(name).to(dev)
            x = torch.randint(0, 256, (args.batch_size, args.block_size), device=dev)
            _logits, loss = m(x, x)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
            n = count_module(m)["total"]
            print(f"  {name:16s} params={n:>12,}  peak={peak:6.1f} GiB  OK")
            del m, x, loss
        except torch.cuda.OutOfMemoryError:
            print(f"  {name:16s} OOM at B={args.batch_size} T={args.block_size}")
        except Exception as e:                      # noqa: BLE001
            print(f"  {name:16s} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
