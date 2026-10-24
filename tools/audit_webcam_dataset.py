#!/usr/bin/env python3
"""
Webcam Dataset Audit Script
===========================
Audits data/webcam_captures/ across all 29 ASL classes.
Reports sample counts, train/val split breakdown, temporal spacing, and data leakage risks.
"""

import json
import random
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.transforms import CLASSES

BASE_DIR = PROJECT_ROOT / "data" / "webcam_captures"


def audit_dataset():
    rng = random.Random(42)
    audit = {}

    for cls in CLASSES:
        cls_dir = BASE_DIR / cls
        if not cls_dir.exists():
            audit[cls] = {'count': 0, 'train': 0, 'val': 0, 'deltas': [], 'samples': []}
            continue
        jpgs = sorted(cls_dir.glob("*.jpg"))
        n = len(jpgs)
        if n == 0:
            audit[cls] = {'count': 0, 'train': 0, 'val': 0, 'deltas': [], 'samples': []}
            continue

        imgs_shuffled = list(jpgs)
        rng.shuffle(imgs_shuffled)
        n_val = max(1, int(n * 0.2))
        val_set = imgs_shuffled[:n_val]
        train_set = imgs_shuffled[n_val:]

        times = []
        for p in jpgs:
            try:
                times.append(datetime.strptime(p.stem, '%Y%m%d_%H%M%S_%f'))
            except Exception:
                pass
        deltas = []
        for i in range(1, len(times)):
            deltas.append(round((times[i] - times[i-1]).total_seconds(), 2))

        audit[cls] = {
            'count': n,
            'train': len(train_set),
            'val': len(val_set),
            'deltas': deltas,
            'samples': [p.name for p in jpgs],
            'train_samples': [p.name for p in train_set],
            'val_samples': [p.name for p in val_set],
        }

    print("=" * 80)
    print(f"{'Class':7s} | {'Total':5s} | {'Train':5s} | {'Val':5s} | {'Inter-frame intervals (sec)'}")
    print("-" * 80)
    for cls in CLASSES:
        d = audit[cls]
        deltas_str = ', '.join([f"{x}s" for x in d['deltas']]) if d['deltas'] else 'None (single or 0 samples)'
        print(f"{cls:7s} | {d['count']:5d} | {d['train']:5d} | {d['val']:5d} | {deltas_str}")

    tot_c = sum(d['count'] for d in audit.values())
    tot_tr = sum(d['train'] for d in audit.values())
    tot_v = sum(d['val'] for d in audit.values())
    print("-" * 80)
    print(f"{'TOTAL':7s} | {tot_c:5d} | {tot_tr:5d} | {tot_v:5d}")
    print("=" * 80)

    return audit


if __name__ == "__main__":
    audit_dataset()
