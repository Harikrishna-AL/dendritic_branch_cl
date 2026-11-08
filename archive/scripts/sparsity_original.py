"""Sparsity at the ORIGINAL upstream training config, BDH vs GPT, apples to apples.

This is the measurement that could not be made on a laptop: upstream's default
config needs ~37 GiB of activation memory for a backward pass at B=32, T=512.
On an H100 it fits.

What "apples to apples" means here, concretely -- everything below is held
identical between the two models, so the only difference is the architecture:

  data            tiny Shakespeare, raw bytes, upstream's own 90/10 split
  vocabulary      256 bytes (neither model gets a better tokenizer)
  block size      512    (upstream BLOCK_SIZE)
  batch size      32     (upstream BATCH_SIZE)
  steps           3000   (upstream MAX_ITERS)
  optimizer       AdamW, lr 1e-3, weight_decay 0.1  (upstream values)
  schedule        none, and no gradient clipping    (upstream has neither)
  precision       fp32
  parameters      matched within 0.6% (25.30M vs 25.44M)

Density is probed on a FIXED held-out batch every ``--probe-every`` steps, so
the series is comparable over time and between models, and we can see whether
BDH's code sparsifies as it trains rather than only measuring the endpoint.

"Active" means > 0 for both: BDH's ReLU latents are non-negative by
construction, GPT's post-GELU hidden state is signed and thresholded at zero.
GELU is positive exactly when its input is, so the threshold means the same
thing in both -- but they are not the same nonlinearity, and the report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl import evaluate as ev, probes as probes_mod, wblog  # noqa: E402
from bdh_cl.config import BDHSpec, GPTSpec, as_dict  # noqa: E402
from bdh_cl.data.bytes import ByteStream  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.gpt_baseline import GPTBaseline  # noqa: E402
from bdh_cl.models.param_accounting import (  # noqa: E402
    bdh_param_count, count_module, gpt_param_count,
)

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

# Upstream defaults, verbatim from train.py / bdh.py.
UPSTREAM_BDH = dict(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128,
                    dropout=0.1, vocab_size=256)

#: k=4 branches, and the multiplier that keeps a dendritic model matched to
#: upstream's 25.30M: n' = 3n/(2k+1), so 128 * 3/9 = 42.67 -> 43 (+0.78%).
DENDRITIC_BRANCHES = 4
DENDRITIC_MULT = 43
# Parameter-matched Transformer: 25,436,672 vs BDH's 25,296,896 (+0.55%).
MATCHED_GPT = dict(n_layer=8, n_embd=512, n_head=8, dropout=0.1, vocab_size=256)


def get_data(cache: Path) -> np.ndarray:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        import requests
        cache.write_bytes(requests.get(SHAKESPEARE_URL, timeout=60).content)
    return np.frombuffer(cache.read_bytes(), dtype=np.uint8)


@torch.no_grad()
def probe_density(model, batches, device) -> dict:
    prof = probes_mod.collect_profile(model, batches, device)
    out = {}
    for stream, layers in prof.streams.items():
        dens = [lp.density for lp in layers]
        out[stream] = {
            "per_layer": dens,
            "mean": float(np.mean(dens)),
            "n_neurons": int(layers[0].n_neurons),
            "active_units_per_layer": float(np.mean(dens) * layers[0].n_neurons),
            "dead_fraction": float(np.mean([lp.dead_fraction for lp in layers])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True,
        choices=["bdh", "gpt", "bdh_den_gated", "bdh_den_sum", "bdh_den_max",
                 "bdh_deep"],
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-batches", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "exp3_original"))
    ap.add_argument("--data-cache", default=str(ROOT / "data_cache" / "shakespeare.txt"))
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default="exp3_original_config")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data = get_data(Path(args.data_cache))
    stream = ByteStream(data, block_size=args.block_size, val_frac=0.1)

    if args.model == "bdh":
        spec = BDHSpec(**UPSTREAM_BDH)
        model = BDHWrapper(spec)
        analytic = bdh_param_count(spec)["total"]
    elif args.model == "bdh_deep":
        # The control for the dendritic result: same multiplier (so the same
        # neuron count N' = 2752), same parameter count, but the encoder's extra
        # capacity goes into one hidden layer with a learned projection instead
        # of k branches with fixed pooling. match_branches=4 derives the hidden
        # width that makes the two match exactly.
        spec = BDHSpec(
            **{**UPSTREAM_BDH, "mlp_internal_dim_multiplier": DENDRITIC_MULT},
            neuron_unit="deep",
            neuron_unit_kwargs={"match_branches": DENDRITIC_BRANCHES},
        )
        model = BDHWrapper(spec)
        analytic = count_module(model)["total"]
    elif args.model.startswith("bdh_den_"):
        # Dendritic BDH at the SAME upstream config. Everything is upstream's
        # -- layers, width, heads, dropout, block, batch, steps, optimizer --
        # except two things that cannot both stay fixed:
        #
        #   neuron_unit                 the architecture under test
        #   mlp_internal_dim_multiplier 128 -> DENDRITIC_MULT
        #
        # The multiplier has to move because the two encoders scale with k, so
        # holding it fixed would hand the dendritic model ~3x the parameters and
        # the comparison would measure capacity instead of architecture. 43
        # keeps the total within +0.78% of upstream's 25.30M, the same
        # discipline used for the matched GPT.
        soma = args.model.rsplit("_", 1)[1]
        spec = BDHSpec(
            **{**UPSTREAM_BDH, "mlp_internal_dim_multiplier": DENDRITIC_MULT},
            neuron_unit="dendritic",
            neuron_unit_kwargs={"n_branches": DENDRITIC_BRANCHES, "soma": soma},
        )
        model = BDHWrapper(spec)
        analytic = count_module(model)["total"]
    else:
        spec = GPTSpec(**MATCHED_GPT)
        model = GPTBaseline(spec)
        analytic = gpt_param_count(spec)["total"]
    model = model.to(device)
    n_params = count_module(model)["total"]
    assert n_params == analytic, (n_params, analytic)

    # Upstream's optimizer settings exactly: no schedule, no clipping.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # A FIXED held-out probe set, identical across steps and across models.
    probe_batches = list(stream.iter_val_blocks(args.batch_size, max_batches=args.probe_batches))

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"model={args.model} params={n_params:,} device={device} ({gpu}) "
          f"B={args.batch_size} T={args.block_size} steps={args.steps}", flush=True)

    wblog.init(
        {"enabled": bool(args.wandb_project), "project": args.wandb_project,
         "group": args.wandb_group, "job_type": "sparsity"},
        name=f"exp3-{args.model}-seed{args.seed}",
        config={"model": args.model, "params": n_params, "seed": args.seed,
                "batch_size": args.batch_size, "block_size": args.block_size,
                "steps": args.steps, "lr": args.lr, "weight_decay": args.weight_decay,
                "spec": as_dict(spec), "gpu": gpu, "config_origin": "upstream train.py defaults"},
        tags=[args.model, "exp3", "original-config"],
    )

    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    loss_buf = torch.zeros(args.steps, device=device)
    model.train()
    t0 = time.perf_counter()

    for step in range(args.steps):
        b = stream.batch("train", args.batch_size, rng).to(device)
        _, loss = model(b.x, b.y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_buf[step] = loss.detach()

        if step % args.log_every == 0 or step == args.steps - 1:
            lo = max(0, step - args.log_every + 1)
            window = float(loss_buf[lo : step + 1].mean())
            elapsed = time.perf_counter() - t0
            tok_s = (step + 1) * args.batch_size * args.block_size / elapsed
            wblog.log({"train/loss": window, "train/bits_per_byte": window / np.log(2),
                       "train/tokens_per_sec": tok_s}, step=step)
            print(f"  step {step:5d}/{args.steps} loss {window:.4f} "
                  f"bpb {window/np.log(2):.4f} {tok_s:,.0f} tok/s", flush=True)

        if step % args.probe_every == 0 or step == args.steps - 1:
            d = probe_density(model, probe_batches, device)
            model.train()
            rec = {"step": step, "density": d,
                   "loss": float(loss_buf[max(0, step - 9) : step + 1].mean())}
            history.append(rec)
            flat = {}
            for sname, v in d.items():
                flat[f"density/{sname}"] = v["mean"]
                flat[f"sparsity/{sname}"] = 1.0 - v["mean"]
                flat[f"active_units/{sname}"] = v["active_units_per_layer"]
                flat[f"dead/{sname}"] = v["dead_fraction"]
                for li, x in enumerate(v["per_layer"]):
                    flat[f"density_layer/{sname}/L{li}"] = x
            wblog.log(flat, step=step)
            msg = "  ".join(f"{k} {v['mean']*100:.1f}%" for k, v in d.items())
            print(f"    [probe] step {step:5d}  {msg}", flush=True)

    elapsed = time.perf_counter() - t0
    final = probe_density(model, probe_batches, device)

    # HELD-OUT bits-per-byte. The training loss is NOT a fair quality metric
    # here: 3000 steps x 32 x 512 is ~49M tokens over a 1.1MB corpus, i.e. ~44
    # epochs, so a 25M-parameter model largely memorises the training split.
    val_batches = list(stream.iter_val_blocks(args.batch_size, max_batches=32))
    val = ev.bits_per_byte(model, val_batches, device)

    result = {
        "experiment": "exp3_sparsity_original_config",
        "model": args.model, "params": n_params, "spec": as_dict(spec),
        "device": str(device), "gpu": gpu,
        "batch_size": args.batch_size, "block_size": args.block_size,
        "steps": args.steps, "lr": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "tokens_per_sec": args.steps * args.batch_size * args.block_size / elapsed,
        "loss_first20": float(loss_buf[:20].mean()),
        "loss_last20": float(loss_buf[-20:].mean()),
        # Explicitly named: this is TRAIN loss, heavily memorised at this
        # token/corpus ratio. Use bits_per_byte_val for quality.
        "bits_per_byte_train": float(loss_buf[-20:].mean()) / float(np.log(2)),
        "bits_per_byte_val": val["bits_per_byte"],
        "val_n_bytes": val["n_bytes"],
        "density_final": final,
        "history": history,
        "apples_to_apples": {
            "held_identical": ["data", "vocab", "block_size", "batch_size", "steps",
                               "optimizer", "lr", "weight_decay", "no_schedule",
                               "no_grad_clip", "precision_fp32", "probe_set"],
            "differs": ["architecture", "layer_count", "hidden_width"],
            "note": "parameters matched within 0.6%; BDH 25,296,896 vs GPT 25,436,672",
        },
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")

    wblog.summary({
        "final/bits_per_byte_train": result["bits_per_byte_train"],
        "final/bits_per_byte_val": result["bits_per_byte_val"],
        "final/loss": result["loss_last20"],
        "final/tokens_per_sec": result["tokens_per_sec"],
        **{f"final/density/{k}": v["mean"] for k, v in final.items()},
        **{f"final/active_units/{k}": v["active_units_per_layer"] for k, v in final.items()},
    })
    wblog.finish()

    print(f"\nloss {result['loss_first20']:.4f} -> {result['loss_last20']:.4f}  "
          f"bpb train {result['bits_per_byte_train']:.4f} | "
          f"HELD-OUT {result['bits_per_byte_val']:.4f}")
    for k, v in final.items():
        print(f"  {k}: density {v['mean']*100:.1f}%  "
              f"({v['active_units_per_layer']:.0f} of {v['n_neurons']} units/layer active)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
