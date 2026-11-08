"""Nearest-class-mean readout, evaluated on checkpoints already on disk.

Why: `diag_label_bias.py` showed that the generative readout collapses onto the
most recent task's labels (free-choice accuracy 0.000 on every earlier task),
which saturates the reported metric and makes it blind to whether the
REPRESENTATION survived. Task-restricted accuracy would separate the two, but
it needs the task id at test time -- the very thing we are trying to do
without.

A prototype readout avoids both problems. Store one mean feature vector per
class, classify by nearest prototype over EVERY class seen so far, and there is
no learned output layer to become biased. It stays genuinely
class-incremental: no task id at test time.

Two protocols, and the difference between them is the measurement:

``frozen``    Each class's prototype is computed at the end of ITS OWN task,
              from that task's training data, and never touched again. This is
              what an online, exemplar-free method could actually store, and it
              is the honest number. Implemented by reading the per-task
              checkpoint that the CL runner already saves.
``refreshed`` All prototypes are recomputed with the FINAL model. This needs old
              data at the end, so it is an ORACLE, not a method. Reported only
              as an upper bound.

``refreshed - frozen`` is representation drift: how much the features moved
under the prototypes. That gap is precisely what a consolidation mechanism is
supposed to shrink, so this is the instrument for testing one.

Storage note: a prototype is a class mean, not stored examples, so ``frozen``
does not smuggle in replay. ``refreshed`` does, which is why it is labelled an
oracle.

    python bdh_cl/scripts/proto_eval.py \
        --config bdh_cl/configs/exp5_split_mnist.yaml \
        --ckpt-dir bdh_cl/results/exp5_split_mnist/ckpt \
        --models bdh_den_gated,bdh,gpt --strategies naive,replay,joint --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl.tasks import build_mnist_sequence  # noqa: E402
from bdh_cl.config import build_from_yaml  # noqa: E402
from bdh_cl.scripts.run_cl import model_builders, pick_device  # noqa: E402


@torch.no_grad()
def features_and_labels(model, batches, device):
    """Final-residual feature at the label position, plus the gold label byte.

    The label sits at ``prompt_len - 1`` in shifted coordinates, so that
    position's representation is the one conditioned on the whole image and
    nothing else -- the natural "embedding of this example".
    """
    feats, labs = [], []
    for b in batches:
        bb = b.to(device)
        if bb.prompt_len is None:
            raise SystemExit("prototype eval needs prompt_len on the batch")
        out = model(bb.x, None, return_hidden=True)
        hidden = out[2]                                   # B, T, D
        idx = (bb.prompt_len - 1).clamp(min=0)
        rows = torch.arange(hidden.shape[0], device=device)
        feats.append(hidden[rows, idx].float().cpu())
        labs.append(bb.y[rows, idx].cpu())
    return torch.cat(feats), torch.cat(labs)


def prototypes_from(feats: torch.Tensor, labs: torch.Tensor) -> dict[int, torch.Tensor]:
    """Class mean of L2-normalised features.

    Normalising before averaging makes the prototype a direction rather than a
    magnitude, so a global drift in feature norm -- which says nothing about
    class identity -- cannot move the decision.
    """
    f = F.normalize(feats, dim=-1)
    out = {}
    for c in labs.unique().tolist():
        out[int(c)] = f[labs == c].mean(dim=0)
    return out


def classify(feats: torch.Tensor, protos: dict[int, torch.Tensor]) -> torch.Tensor:
    """Nearest prototype by cosine similarity, over every class stored so far."""
    classes = sorted(protos)
    P = F.normalize(torch.stack([protos[c] for c in classes]), dim=-1)
    sims = F.normalize(feats, dim=-1) @ P.T
    return torch.tensor([classes[i] for i in sims.argmax(dim=-1).tolist()])


def load(builders, name, ckpt, device):
    model = builders[name]().to(device)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Checkpoints saved before SparseDendriticNeuronUnit gained its
    # ``coactivity`` buffer (exp18) lack it. That buffer is a training-time
    # regrow trace, never read in forward, so a zero fill is exact for eval.
    # Anything else missing is still an error.
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    bad = [k for k in missing if k.rsplit(".", 1)[-1] != "coactivity"]
    if bad or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={bad} unexpected={unexpected}")
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--models", default="bdh_den_gated,bdh,gpt")
    ap.add_argument("--strategies", default="naive,replay,joint")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--proto-batches", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = build_from_yaml(args.config)
    device = pick_device(args.device)
    builders = model_builders(cfg)
    ckpt_dir = Path(args.ckpt_dir)

    tasks = build_mnist_sequence(
        cfg["mnist_cache"],
        side=cfg.get("mnist_side", 14),
        n_eval=cfg.get("n_eval", 200),
        eval_seed=cfg.get("eval_seed", 0),
        batch_size=cfg.get("eval_batch_size", 32),
        task_tags=cfg.get("mnist_task_tags", False),
    )
    report = {"config": args.config, "runs": {}}

    for mname in args.models.split(","):
        for sname in args.strategies.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                label = f"{mname}_{sname}"
                cks = [
                    ckpt_dir / f"{label}_seed{seed}_after{i}_{t.name}.pt"
                    for i, t in enumerate(tasks)
                ]
                if not all(c.exists() for c in cks):
                    print(f"[skip] {label} seed{seed}: checkpoints missing", flush=True)
                    continue

                rng = np.random.default_rng(seed)

                # FROZEN: each class's prototype from the model as it stood at
                # the end of that class's own task. Training data, because an
                # online method would not have the held-out split.
                frozen: dict[int, torch.Tensor] = {}
                for i, t in enumerate(tasks):
                    m = load(builders, mname, cks[i], device)
                    items = t.sample_items(rng, args.proto_batches * t.batch_size)
                    batches = [
                        t.collate_items(items[j : j + t.batch_size])
                        for j in range(0, len(items), t.batch_size)
                    ]
                    f, l = features_and_labels(m, batches, device)
                    frozen.update(prototypes_from(f, l))
                    del m

                # Final model for both readouts.
                final = load(builders, mname, cks[-1], device)

                # REFRESHED (oracle): every prototype recomputed with the final
                # model, which requires old data and is therefore not a method.
                refreshed: dict[int, torch.Tensor] = {}
                for t in tasks:
                    items = t.sample_items(rng, args.proto_batches * t.batch_size)
                    batches = [
                        t.collate_items(items[j : j + t.batch_size])
                        for j in range(0, len(items), t.batch_size)
                    ]
                    f, l = features_and_labels(final, batches, device)
                    refreshed.update(prototypes_from(f, l))

                per_task = {"frozen": {}, "refreshed": {}}
                for t in tasks:
                    f, l = features_and_labels(final, t.eval_batches(), device)
                    for kind, protos in (("frozen", frozen), ("refreshed", refreshed)):
                        pred = classify(f, protos)
                        per_task[kind][t.name] = float((pred == l).float().mean())
                del final

                means = {
                    k: float(np.mean(list(v.values()))) for k, v in per_task.items()
                }
                report["runs"][f"{label}_seed{seed}"] = {
                    "per_task": per_task,
                    "mean": means,
                    "drift": means["refreshed"] - means["frozen"],
                    "n_classes": len(frozen),
                }
                print(
                    f"{label} seed{seed}: frozen={means['frozen']:.3f} "
                    f"refreshed(oracle)={means['refreshed']:.3f} "
                    f"drift={means['refreshed'] - means['frozen']:+.3f}",
                    flush=True,
                )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
