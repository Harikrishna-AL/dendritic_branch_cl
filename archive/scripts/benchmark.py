"""Throughput and memory feasibility sweep, to size the real experiments.

Records, per (model, preset, batch, block): tokens/sec for a full
forward+backward+step, and the analytic activation-memory estimate. This is
what decides which presets are usable on a 16 GiB machine at all.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.config import preset  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.gpt_baseline import GPTBaseline  # noqa: E402
from bdh_cl.models.param_accounting import count_module  # noqa: E402


def bdh_activation_gib(spec, B, T) -> float:
    """3 sparse (B,nh,T,N) tensors + a (B,nh,T,T) score matrix, per layer, fp32."""
    xs = B * spec.n_head * T * spec.N * 4
    sc = B * spec.n_head * T * T * 4
    return (3 * xs + sc) * spec.n_layer / 2**30


def gpt_activation_gib(spec, B, T) -> float:
    """Dominant term: the 4*d MLP hidden state per layer (SDPA is fused)."""
    return (B * T * 4 * spec.n_embd * 4) * spec.n_layer * 2 / 2**30


def bench(model, B, T, device, steps: int, warmup: int = 2) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    model.train()

    def one():
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one()
    if device.type == "mps":
        torch.mps.synchronize()

    per_step = []
    for _ in range(steps):
        t0 = time.perf_counter()
        one()
        if device.type == "mps":
            torch.mps.synchronize()
        per_step.append(time.perf_counter() - t0)

    per_step.sort()
    med = per_step[len(per_step) // 2]
    return {
        "median_step_sec": med,
        "min_step_sec": per_step[0],
        "max_step_sec": per_step[-1],
        "tokens_per_sec": B * T / med,
        "steps_per_sec": 1.0 / med,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="tiny,small")
    ap.add_argument("--models", default="bdh,gpt")
    ap.add_argument("--batches", default="8,16")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
    args = ap.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    rows = []
    for pname in args.presets.split(","):
        bdh_spec, gpt_spec, _ = preset(pname)
        for mname in args.models.split(","):
            for B in [int(b) for b in args.batches.split(",")]:
                T = args.block
                est = (
                    bdh_activation_gib(bdh_spec, B, T)
                    if mname == "bdh"
                    else gpt_activation_gib(gpt_spec, B, T)
                )
                row = {
                    "model": mname,
                    "preset": pname,
                    "batch_size": B,
                    "block_size": T,
                    "activation_gib_est": est,
                }
                if est > 12.0:
                    row["skipped"] = "estimated activation memory > 12 GiB on a 16 GiB machine"
                    rows.append(row)
                    print(f"SKIP {mname}/{pname} B={B}: {est:.1f} GiB est", flush=True)
                    continue
                model = (BDHWrapper(bdh_spec) if mname == "bdh" else GPTBaseline(gpt_spec)).to(device)
                row["params"] = count_module(model)["total"]
                try:
                    row.update(bench(model, B, T, device, args.steps))
                    print(
                        f"{mname:4s}/{pname:6s} B={B:3d} T={T} "
                        f"{row['tokens_per_sec']:>9,.0f} tok/s  "
                        f"{row['median_step_sec']*1000:>8.1f} ms/step  est {est:.2f} GiB",
                        flush=True,
                    )
                except RuntimeError as e:
                    row["error"] = str(e)[:300]
                    print(f"FAIL {mname}/{pname} B={B}: {e}"[:200], flush=True)
                del model
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
