#!/usr/bin/env python3
"""
ASL Training Image Augmentation Pipeline
=========================================
Generates realistic, label-preserving augmented training images from
manually verified webcam captures.

Key Principles
--------------
1. Strict Handedness & Label Protection:
   - NO horizontal flipping (handedness and asymmetric signs change meaning).
   - Bounded small rotations (approx. ±10°).
   - Small translations (±6% frame width/height).
   - Moderate scaling (0.94x - 1.06x) without cropping fingers.
   - Mild brightness/contrast/saturation adjustments.
   - Small Gaussian sensor noise (sigma <= 3.5).
   - NO synthetic generation for classes without verified source images (del, nothing, space).
   - Explicit movement limitations documented for J and Z (trajectory-dependent signs).

2. Train / Validation Isolation:
   - Splits original webcam captures into train and validation sets BEFORE augmentation.
   - NEVER puts augmented variants of validation images into the training set.
   - Validation set contains ONLY genuine, unaugmented physical captures.

3. Complete Audit & Traceability:
   - Augmented files are saved in a separate directory (`data/webcam_augmented/train/`).
   - Every augmented image logs its source image, class label, and exact parameters
     in `data/webcam_augmented/manifest.json`.
   - Augmented samples are explicitly tagged `is_augmented: True` (never masqueraded as
     independent physical captures).
"""

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.transforms import CLASSES, CLASS_TO_IDX

# Dynamic trajectory signs that involve movement in ASL:
DYNAMIC_SIGNS = {"J", "Z"}


# ---------------------------------------------------------------------------
# Transformation Functions
# ---------------------------------------------------------------------------

def apply_geometric_transform(
    image: np.ndarray,
    angle: float,
    scale: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Applies small bounded affine transformation (rotation, scale, translation)
    using reflection border padding to prevent artificial black borders.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    transformed = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return transformed


def apply_photometric_transform(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue_delta: float,
) -> np.ndarray:
    """
    Applies mild brightness, contrast, and HSV saturation/hue adjustments.
    """
    # 1. Brightness and contrast
    # contrast centered at 128
    img_f = image.astype(np.float32)
    img_bc = np.clip(contrast * (img_f - 128.0) + 128.0 * brightness, 0.0, 255.0).astype(np.uint8)

    # 2. Saturation and Hue in HSV space
    hsv = cv2.cvtColor(img_bc, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0.0, 255.0)
    hsv = np.clip(hsv, 0.0, 255.0).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def apply_sensor_noise(
    image: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates realistic webcam CMOS sensor noise.
    """
    if noise_sigma <= 0.1:
        return image
    noise = rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return noisy


def generate_augmented_sample(
    source_bgr: np.ndarray,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    """
    Generates a single augmented variant and records exact transformation parameters.
    """
    h, w = source_bgr.shape[:2]

    # Bounded rotation within ±10 degrees
    angle = rng_py.uniform(-10.0, 10.0)

    # Moderate scale variation within 0.94 - 1.06
    scale = rng_py.uniform(0.94, 1.06)

    # Translation within ±6% of width/height
    max_dx = w * 0.06
    max_dy = h * 0.06
    dx = rng_py.uniform(-max_dx, max_dx)
    dy = rng_py.uniform(-max_dy, max_dy)

    # Photometric parameters
    brightness = rng_py.uniform(0.88, 1.12)
    contrast = rng_py.uniform(0.88, 1.12)
    saturation = rng_py.uniform(0.88, 1.12)
    hue_delta = rng_py.uniform(-3.0, 3.0)  # degrees in OpenCV 0-180 scale

    # Sensor noise sigma (0 to 3.5)
    noise_sigma = rng_py.uniform(0.0, 3.5)

    # Pipeline execution
    geom = apply_geometric_transform(source_bgr, angle, scale, dx, dy)
    photo = apply_photometric_transform(geom, brightness, contrast, saturation, hue_delta)
    result = apply_sensor_noise(photo, noise_sigma, rng_np)

    params = {
        "angle_deg": round(angle, 2),
        "scale": round(scale, 4),
        "translate_dx_px": round(dx, 2),
        "translate_dy_px": round(dy, 2),
        "brightness_factor": round(brightness, 3),
        "contrast_factor": round(contrast, 3),
        "saturation_factor": round(saturation, 3),
        "hue_delta": round(hue_delta, 2),
        "noise_sigma": round(noise_sigma, 2),
        "horizontal_flip": False,  # Explicitly disabled
    }

    return result, params


# ---------------------------------------------------------------------------
# Dataset Splitting (Pre-Augmentation)
# ---------------------------------------------------------------------------

def split_webcam_dataset(
    input_dir: Path,
    val_ratio: float = 0.25,
    single_sample_assignment: str = "train",
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Splits webcam captures into training and validation sets BEFORE augmentation.
    Groups captures separated by < 10 seconds to reduce adjacent-frame leakage.
    """
    rng = random.Random(seed)
    train_splits = {cls: [] for cls in CLASSES}
    val_splits = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = input_dir / cls
        if not cls_dir.exists():
            continue
        jpgs = sorted(cls_dir.glob("*.jpg"))
        if not jpgs:
            continue

        if len(jpgs) == 1:
            if single_sample_assignment == "train":
                train_splits[cls].append(jpgs[0])
            else:
                val_splits[cls].append(jpgs[0])
            continue

        # Multiple samples: cluster by timestamp interval (>10s = new session attempt)
        attempts = []
        current_attempt = [jpgs[0]]

        for i in range(1, len(jpgs)):
            try:
                t0 = datetime.strptime(jpgs[i - 1].stem, "%Y%m%d_%H%M%S_%f")
                t1 = datetime.strptime(jpgs[i].stem, "%Y%m%d_%H%M%S_%f")
                delta_sec = abs((t1 - t0).total_seconds())
            except Exception:
                delta_sec = 999.0

            if delta_sec < 10.0:
                current_attempt.append(jpgs[i])
            else:
                attempts.append(current_attempt)
                current_attempt = [jpgs[i]]
        attempts.append(current_attempt)

        if len(attempts) >= 2:
            # Session-aware split: hold out one entire attempt for validation
            rng.shuffle(attempts)
            n_val_attempts = max(1, int(len(attempts) * val_ratio))
            val_attempts = attempts[:n_val_attempts]
            train_attempts = attempts[n_val_attempts:]
            for att in val_attempts:
                val_splits[cls].extend(att)
            for att in train_attempts:
                train_splits[cls].extend(att)
        else:
            # Only one recording session: split frames with at least 1 held out for val
            shuffled = list(jpgs)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            val_splits[cls].extend(shuffled[:n_val])
            train_splits[cls].extend(shuffled[n_val:])

    return train_splits, val_splits


# ---------------------------------------------------------------------------
# Augmentation Execution
# ---------------------------------------------------------------------------

def run_augmentation(
    input_dir: Path,
    output_dir: Path,
    target_per_class: int = 50,
    val_ratio: float = 0.25,
    seed: int = 42,
    single_sample_assignment: str = "train",
) -> Dict:
    """
    Executes the augmentation pipeline:
    1. Partitions genuine webcam captures into train/val.
    2. Copies genuine train and val images to output structure.
    3. Augments ONLY training images up to target_per_class.
    4. Records complete manifest.json and split_info.json.
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    train_out_dir = output_dir / "train"
    val_out_dir = output_dir / "val"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)

    train_splits, val_splits = split_webcam_dataset(
        input_dir=input_dir,
        val_ratio=val_ratio,
        single_sample_assignment=single_sample_assignment,
        seed=seed,
    )

    manifest_entries = []
    summary_stats = {
        "classes_processed": 0,
        "classes_missing": 0,
        "original_train_count": 0,
        "original_val_count": 0,
        "augmented_train_count": 0,
        "total_train_count": 0,
        "per_class": {},
    }

    print("\n" + "=" * 78)
    print(f"ASL Webcam Augmentation Pipeline (Seed={seed}, Target={target_per_class}/class)")
    print("=" * 78)
    print(f"{'Class':7s} | {'Orig Train':10s} | {'Orig Val':8s} | {'Augmented':9s} | {'Final Train':11s} | {'Notes'}")
    print("-" * 78)

    for cls in CLASSES:
        cls_train_out = train_out_dir / cls
        cls_val_out = val_out_dir / cls
        cls_train_out.mkdir(parents=True, exist_ok=True)
        cls_val_out.mkdir(parents=True, exist_ok=True)

        orig_train_files = train_splits.get(cls, [])
        orig_val_files = val_splits.get(cls, [])

        n_train_orig = len(orig_train_files)
        n_val_orig = len(orig_val_files)
        summary_stats["original_train_count"] += n_train_orig
        summary_stats["original_val_count"] += n_val_orig

        # 1. Copy genuine validation images (never augmented)
        for val_img_path in orig_val_files:
            dst_val = cls_val_out / val_img_path.name
            shutil.copy2(val_img_path, dst_val)
            meta_src = val_img_path.with_suffix(".json")
            if meta_src.exists():
                shutil.copy2(meta_src, dst_val.with_suffix(".json"))

            manifest_entries.append({
                "output_file": str(dst_val.relative_to(output_dir)),
                "split": "val",
                "class_label": cls,
                "is_augmented": False,
                "is_genuine_webcam": True,
                "source_file": str(val_img_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in val_img_path.parents else val_img_path),
                "params": None,
            })

        # 2. Copy genuine training images
        for train_img_path in orig_train_files:
            dst_train = cls_train_out / train_img_path.name
            shutil.copy2(train_img_path, dst_train)
            meta_src = train_img_path.with_suffix(".json")
            if meta_src.exists():
                shutil.copy2(meta_src, dst_train.with_suffix(".json"))

            manifest_entries.append({
                "output_file": str(dst_train.relative_to(output_dir)),
                "split": "train",
                "class_label": cls,
                "is_augmented": False,
                "is_genuine_webcam": True,
                "source_file": str(train_img_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in train_img_path.parents else train_img_path),
                "params": None,
            })

        # 3. Augment training images if genuine source images exist
        augmented_count = 0
        notes = []
        if cls in DYNAMIC_SIGNS:
            notes.append("Dynamic gesture (J/Z movement caveat)")

        if n_train_orig == 0:
            notes.append("No source captures (del/nothing/space or missing)")
            summary_stats["classes_missing"] += 1
        else:
            summary_stats["classes_processed"] += 1
            needed = max(0, target_per_class - n_train_orig)
            if needed > 0:
                variants_per_image = math.ceil(needed / n_train_orig)
                aug_idx = 0
                for source_path in orig_train_files:
                    source_bgr = cv2.imread(str(source_path))
                    if source_bgr is None:
                        continue

                    for _ in range(variants_per_image):
                        if augmented_count >= needed:
                            break
                        aug_idx += 1
                        aug_img, aug_params = generate_augmented_sample(source_bgr, rng_py, rng_np)

                        aug_name = f"aug_{source_path.stem}_{aug_idx:03d}.jpg"
                        aug_dst = cls_train_out / aug_name
                        cv2.imwrite(str(aug_dst), aug_img)

                        meta_dst = aug_dst.with_suffix(".json")
                        with open(meta_dst, "w") as f_meta:
                            json.dump({
                                "source_file": str(source_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in source_path.parents else source_path),
                                "class_label": cls,
                                "is_augmented": True,
                                "is_genuine_webcam": False,
                                "augmentation_params": aug_params,
                                "created_at": datetime.now().isoformat(),
                            }, f_meta, indent=2)

                        manifest_entries.append({
                            "output_file": str(aug_dst.relative_to(output_dir)),
                            "split": "train",
                            "class_label": cls,
                            "is_augmented": True,
                            "is_genuine_webcam": False,
                            "source_file": str(source_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in source_path.parents else source_path),
                            "params": aug_params,
                        })

                        augmented_count += 1

        summary_stats["augmented_train_count"] += augmented_count
        final_train = n_train_orig + augmented_count
        summary_stats["total_train_count"] += final_train
        summary_stats["per_class"][cls] = {
            "orig_train": n_train_orig,
            "orig_val": n_val_orig,
            "augmented_train": augmented_count,
            "final_train": final_train,
            "notes": ", ".join(notes) if notes else "OK",
        }

        note_str = ", ".join(notes) if notes else "OK"
        print(f"{cls:7s} | {n_train_orig:10d} | {n_val_orig:8d} | {augmented_count:9d} | {final_train:11d} | {note_str}")

    print("-" * 78)
    print(
        f"{'TOTAL':7s} | "
        f"{summary_stats['original_train_count']:10d} | "
        f"{summary_stats['original_val_count']:8d} | "
        f"{summary_stats['augmented_train_count']:9d} | "
        f"{summary_stats['total_train_count']:11d} | "
        f"Manifest: {len(manifest_entries)} entries"
    )
    print("=" * 78)

    # Save manifest and split info
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "1.0",
            "seed": seed,
            "target_per_class": target_per_class,
            "created_at": datetime.now().isoformat(),
            "summary": summary_stats,
            "entries": manifest_entries,
        }, f, indent=2)

    split_path = output_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\n[OK] Augmentation completed.")
    print(f"  Manifest written to: {manifest_path}")
    print(f"  Split summary written to: {split_path}")
    return summary_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASL Training Image Augmentation Pipeline")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "webcam_captures",
                        help="Input directory of genuine webcam captures")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "webcam_augmented",
                        help="Output directory for partitioned and augmented dataset")
    parser.add_argument("--target-per-class", type=int, default=50,
                        help="Target training count per class after augmentation (default: 50)")
    parser.add_argument("--val-ratio", type=float, default=0.25,
                        help="Ratio of webcam captures allocated to validation (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--single-sample-assignment", choices=["train", "val"], default="train",
                        help="Partition assignment for classes with exactly 1 capture (default: train)")
    args = parser.parse_args()

    run_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_per_class=args.target_per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
        single_sample_assignment=args.single_sample_assignment,
    )


if __name__ == "__main__":
    main()
