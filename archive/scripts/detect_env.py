"""Detect hardware / software environment and write it to results/env.json.

Step 0.1 of the BDH continual-learning harness.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _sysctl(key: str):
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_hash(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def collect() -> dict:
    device = pick_device()

    accel: dict = {"kind": device}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        accel.update(
            name=props.name,
            total_memory_bytes=props.total_memory,
            total_memory_gib=round(props.total_memory / 2**30, 2),
            capability=f"{props.major}.{props.minor}",
            device_count=torch.cuda.device_count(),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    elif device == "mps":
        # MPS has no separate VRAM: it shares the machine's unified memory.
        memsize = _sysctl("hw.memsize")
        accel.update(
            name=_sysctl("machdep.cpu.brand_string"),
            unified_memory=True,
            total_memory_bytes=int(memsize) if memsize else None,
            total_memory_gib=round(int(memsize) / 2**30, 2) if memsize else None,
            # MPS autocast/bf16 support is partial; we train in fp32 on MPS.
            bf16_supported=False,
        )
    else:
        memsize = _sysctl("hw.memsize")
        accel.update(
            name=platform.processor() or None,
            total_memory_bytes=int(memsize) if memsize else None,
            total_memory_gib=round(int(memsize) / 2**30, 2) if memsize else None,
        )

    return {
        "device": device,
        "accelerator": accel,
        "cpu": {
            "ncpu": os.cpu_count(),
            "brand": _sysctl("machdep.cpu.brand_string"),
            "arch": platform.machine(),
        },
        "software": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": {
            "bdh_d": git_hash(ROOT),
            "upstream_bdh": git_hash(ROOT.parent / "upstream_bdh"),
        },
    }


def main() -> None:
    env = collect()
    out = ROOT / "results" / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
