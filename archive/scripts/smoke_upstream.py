"""Step 0.2: run the repo's own training loop, confirm loss decreases, record tokens/sec.

The upstream ``train.py`` is used *as a module*, not edited: its
``BDH_CONFIG``, ``get_batch``, ``LEARNING_RATE`` and ``WEIGHT_DECAY`` are the
ones exercised here. Only module-level hyperparameters are rebound from the
outside.

Two documented deviations, both forced by the hardware (16 GiB unified memory,
no CUDA) and both recorded in NOTES.md:

1. ``BATCH_SIZE``/``BLOCK_SIZE`` are reduced. Upstream's defaults
   (B=32, T=512) need ~37 GiB of activation memory for the backward pass --
   3 sparse (B, nh, T, N) tensors per layer x 6 layers at N=8192. Running it
   unchanged drove the machine 37 GB into swap and made no progress.
2. ``torch.compile`` and the float16 ``GradScaler`` are skipped. The scaler
   upstream builds is ``GradScaler(device="cpu", enabled=True)``, a CUDA-
   oriented path; we train in fp32 on MPS instead. This changes speed, not the
   maths being checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "upstream_bdh"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT.parent))

import bdh as upstream_bdh  # noqa: E402
import train as upstream_train  # noqa: E402  (main is __main__-guarded, so this is safe)

from bdh_cl.config import preset  # noqa: E402
from bdh_cl.models.bdh_wrapper import upstream_config  # noqa: E402
from bdh_cl.models.param_accounting import count_module  # noqa: E402


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)

    # Upstream's own data pipeline, on its own dataset.
    upstream_train.fetch_data()
    upstream_train.device = device
    upstream_train.BATCH_SIZE = args.batch_size
    upstream_train.BLOCK_SIZE = args.block_size

    # 'small' IS the upstream default config; the others are our presets.
    if args.preset == "small":
        cfg = upstream_train.BDH_CONFIG
    else:
        cfg = upstream_config(preset(args.preset)[0])

    torch.manual_seed(1337)
    model = upstream_bdh.BDH(cfg).to(device)
    n_params = count_module(model)["total"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=upstream_train.LEARNING_RATE,
        weight_decay=upstream_train.WEIGHT_DECAY,
    )

    print(
        f"device={device} preset={args.preset} params={n_params:,} "
        f"B={args.batch_size} T={args.block_size} steps={args.steps}",
        flush=True,
    )

    losses: list[float] = []
    curve: list[dict] = []
    model.train()
    x, y = upstream_train.get_batch("train")

    # One untimed step so lazy init / kernel warmup is not counted.
    logits, loss = model(x, y)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for step in range(args.steps):
        logits, loss = model(x, y)
        x, y = upstream_train.get_batch("train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            window = losses[-args.log_every :]
            el = time.perf_counter() - t0
            tps = (step + 1) * args.batch_size * args.block_size / el
            print(
                f"step {step:5d}/{args.steps}  loss {sum(window)/len(window):.4f}  "
                f"{tps:,.0f} tok/s",
                flush=True,
            )
            curve.append({"step": step, "loss": sum(window) / len(window), "tokens_per_sec": tps})

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    n_tokens = args.steps * args.batch_size * args.block_size
    first = sum(losses[:20]) / len(losses[:20])
    last = sum(losses[-20:]) / len(losses[-20:])

    result = {
        "step": "0.2 upstream train.py smoke",
        "preset": args.preset,
        "device": str(device),
        "dtype": "float32",
        "params": n_params,
        "config": {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "mlp_internal_dim_multiplier": cfg.mlp_internal_dim_multiplier,
            "vocab_size": cfg.vocab_size,
            "dropout": cfg.dropout,
        },
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "steps": args.steps,
        "elapsed_sec": elapsed,
        "tokens_total": n_tokens,
        "tokens_per_sec": n_tokens / elapsed,
        "steps_per_sec": args.steps / elapsed,
        "loss_first20_mean": first,
        "loss_last20_mean": last,
        "loss_decreased": last < first,
        "curve": curve,
        "deviations": [
            "BATCH_SIZE/BLOCK_SIZE reduced from upstream 32/512: defaults need ~37 GiB activations, machine has 16 GiB",
            "torch.compile skipped (slow/unsupported for this graph on MPS)",
            "float16 GradScaler skipped; fp32 on MPS",
            "grad clipping at 1.0 added (upstream has none)",
        ],
    }

    out = Path(args.out) if args.out else ROOT / "results" / f"step0_upstream_smoke_{args.preset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"\nloss {first:.4f} -> {last:.4f}  decreased={last < first}\n"
        f"{n_tokens / elapsed:,.0f} tokens/sec over {elapsed:.1f}s\nwrote {out}"
    )


if __name__ == "__main__":
    main()
