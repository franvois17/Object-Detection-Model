#!/usr/bin/env python3
"""Download pretrained model weights."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import CONFIG


def download_yolo() -> None:
    from ultralytics import YOLO

    dest = CONFIG.pretrained_dir / CONFIG.yolo_model
    if dest.exists():
        print(f"YOLOv8n already at {dest}")
        return
    print("Downloading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    import shutil
    # ultralytics downloads to cwd or ~/.config; move to our dir
    downloaded = Path("yolov8n.pt")
    if downloaded.exists():
        shutil.move(str(downloaded), str(dest))
    print(f"YOLOv8n saved to {dest}")


def download_efficientnet() -> None:
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    dest = CONFIG.pretrained_dir / "efficientnet_v2_s.pt"
    if dest.exists():
        print(f"EfficientNet-V2-S already at {dest}")
        return
    print("Downloading EfficientNet-V2-S...")
    import torch
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), dest)
    print(f"EfficientNet-V2-S saved to {dest}")


def main() -> None:
    CONFIG.ensure_dirs()
    print("=== Downloading pretrained models ===\n")
    download_yolo()
    print()
    download_efficientnet()
    print("\nDone!")


if __name__ == "__main__":
    main()
