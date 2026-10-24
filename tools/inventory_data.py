import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = Path(".").resolve()

print("--- 1. Raw ASL Alphabet Directory ---")
raw_dir = PROJECT_ROOT / "data" / "raw"
for p in sorted(raw_dir.glob("*")):
    if p.is_dir():
        file_count = len(list(p.rglob("*.jpg")) + list(p.rglob("*.png")))
        print(f"  {p.relative_to(PROJECT_ROOT)}: {file_count} images")

print("\n--- 2. Raw Splits Directory ---")
splits_dir = raw_dir / "splits"
if splits_dir.exists():
    for s in sorted(splits_dir.iterdir()):
        if s.is_dir():
            classes = sorted([d.name for d in s.iterdir() if d.is_dir()])
            total_imgs = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) for d in s.iterdir() if d.is_dir())
            print(f"  Split '{s.name}': {len(classes)} classes, {total_imgs} images")

print("\n--- 3. Webcam Captures Directory ---")
wc_dir = PROJECT_ROOT / "data" / "webcam_captures"
if wc_dir.exists():
    wc_classes = sorted([d.name for d in wc_dir.iterdir() if d.is_dir()])
    total_wc = 0
    wc_per_class = {}
    for c in wc_classes:
        c_dir = wc_dir / c
        imgs = list(c_dir.glob("*.jpg"))
        wc_per_class[c] = len(imgs)
        total_wc += len(imgs)
    print(f"  Total webcam captures: {total_wc} images across {len(wc_classes)} classes")
    print(f"  Classes present: {wc_classes}")
    missing_classes = [c for c, count in wc_per_class.items() if count == 0]
    print(f"  Empty classes: {missing_classes}")
    for c, count in wc_per_class.items():
        print(f"    {c}: {count} images")

print("\n--- 4. Webcam Augmented Directory ---")
aug_dir = PROJECT_ROOT / "data" / "webcam_augmented"
if aug_dir.exists():
    for sub in ["train", "val"]:
        sub_dir = aug_dir / sub
        if sub_dir.exists():
            sub_classes = sorted([d.name for d in sub_dir.iterdir() if d.is_dir()])
            total_aug = sum(len(list(d.glob("*.jpg"))) for d in sub_dir.iterdir() if d.is_dir())
            print(f"  Augmented '{sub}': {len(sub_classes)} classes, {total_aug} images")
