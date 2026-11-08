"""Is class-incremental collapse a FEATURE failure or a READOUT failure?

The consolidation sweep protects latent units and still leaves every earlier
task at exactly 0.00 while the newest task sits at 0.98. Two very different
explanations fit that:

  (a) features destroyed -- the latent code for old tasks was overwritten, so
      protecting the right units should fix it;
  (b) readout collapsed -- the features are intact but the model only ever
      emits the labels of the most recent task, because those are the only
      labels in the loss. Feature consolidation cannot fix this.

They are distinguished by looking at WHAT the model predicts. Under (b) a
picture of a 0 is confidently labelled 8 or 9. Prints the histogram of
predicted label bytes per task, plus the accuracy the model would get if it
were only allowed to choose between that task's own two classes -- which is
what the features alone can support.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from bdh_cl.cl.tasks import build_mnist_sequence  # noqa: E402
from bdh_cl.config import build_from_yaml  # noqa: E402
from bdh_cl.data.bytes import encode  # noqa: E402
from bdh_cl.scripts.run_cl import model_builders, pick_device  # noqa: E402


@torch.no_grad()
def label_logits(model, task, device, max_batches=4):
    """For each held-out example, the logits over the single label position."""
    rows = []
    for b in task.eval_batches()[:max_batches]:
        bb = b.to(device)
        logits, _ = model(bb.x, None)
        # the label byte is predicted at index len(prompt)-1 in shifted coords
        if bb.prompt_len is None:
            raise SystemExit("this diagnostic needs prompt_len on the batch")
        idx = (bb.prompt_len - 1).clamp(min=0)
        sel = logits[torch.arange(logits.shape[0], device=device), idx]  # B, vocab
        gold = bb.y[torch.arange(bb.x.shape[0], device=device), idx]
        rows.append((sel.float().cpu(), gold.cpu()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="bdh_den_gated")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = build_from_yaml(args.config)
    device = pick_device(args.device)
    model = model_builders(cfg)[args.model]().to(device)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    tasks = build_mnist_sequence(
        cfg["mnist_cache"],
        side=cfg.get("mnist_side", 14),
        n_eval=cfg.get("n_eval", 200),
        eval_seed=cfg.get("eval_seed", 0),
        batch_size=cfg.get("eval_batch_size", 32),
        task_tags=cfg.get("mnist_task_tags", False),
    )

    print(f"checkpoint: {Path(args.ckpt).name}  (after task {blob.get('task_index')})")
    for t in tasks:
        rows = label_logits(model, t, device)
        preds, golds = [], []
        for sel, gold in rows:
            preds += sel.argmax(-1).tolist()
            golds += gold.tolist()
        hist = Counter(chr(p) if 32 <= p < 127 else f"\\x{p:02x}" for p in preds)
        # restricted: force the choice between this task's own two digits
        digits = sorted({chr(g) for g in golds})
        cols = [ord(d) for d in digits]
        n_ok = n = 0
        for sel, gold in rows:
            sub = sel[:, cols].argmax(-1)
            n_ok += int((torch.tensor([cols[i] for i in sub.tolist()]) == gold).sum())
            n += len(gold)
        free = sum(int(p == g) for p, g in zip(preds, golds)) / max(len(golds), 1)
        print(
            f"  {t.name:10s} gold={digits}  free_choice_acc={free:.3f}  "
            f"restricted_to_own_classes={n_ok / max(n,1):.3f}"
        )
        print(f"    predicted bytes: {dict(hist.most_common(6))}")


if __name__ == "__main__":
    main()
