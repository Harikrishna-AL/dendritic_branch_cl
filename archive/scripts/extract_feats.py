"""Frozen ImageNet ResNet-18 features for CIFAR-100 (Exp20d).

The backbone is torchvision's IMAGENET1K_V1 ResNet-18, eval mode, never
trained here. CIFAR images are upsampled 32 -> 224 (bilinear) and ImageNet-
normalised; the 512-d global-average-pool output is the feature. No
augmentation: features are computed once and reused by every arm.

Caveat for any claim: ImageNet overlaps CIFAR-100's concepts, so this is the
pretrained-CL regime, not learning from scratch.

    TORCH_HOME=data_cache/torch_hub python bdh_cl/scripts/extract_feats.py \
        --data data_cache/cifar100 --out data_cache/cifar100_r18_feats.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def backbone(weights=True):
    import torchvision

    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if weights else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = torch.nn.Identity()
    return m.eval()


@torch.no_grad()
def encode(model, x_uint8, device, bs=256, size=224):
    out = []
    mean, std = MEAN.to(device), STD.to(device)
    for i in range(0, len(x_uint8), bs):
        x = torch.as_tensor(x_uint8[i : i + bs], device=device).float() / 255
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).float().cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N images per split")
    a = p.parse_args()
    from bdh_cl.data import cifar100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = dict(zip(("train", "valid", "test"), cifar100.load(a.data)[:3]))
    model = backbone().to(device)
    blob = {}
    for name, (x, y) in splits.items():
        if a.limit:
            x, y = x[: a.limit], y[: a.limit]
        blob[name] = (encode(model, x, device), torch.as_tensor(y).long())
        print(name, tuple(blob[name][0].shape), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
