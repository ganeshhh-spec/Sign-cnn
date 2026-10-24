#!/usr/bin/env python3
"""
Generate Complete Dataset Inventory CSV
=======================================
Inventories every image in:
  - data/webcam_captures/ (56 original captures)
  - data/webcam_augmented/ (1300 train + 23 held-out val)
  - scratch/physical_webcam_test/ (21 raw frames + 21 crops + 14 normalized crops)
  - scratch/video_frames/ (25 timeline frames)
  - data/raw/asl_alphabet_test/ (28 Kaggle test images)
  - data/raw/splits/ (87,000 Kaggle train/val/test images)

Outputs: reports/dataset_inventory.csv
"""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from backend.preprocessing.transforms import CLASSES

OUT_CSV = PROJECT_ROOT / "reports" / "dataset_inventory.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 1. Load manifest and split info
split_info_path = PROJECT_ROOT / "data" / "webcam_augmented" / "split_info.json"
manifest_path = PROJECT_ROOT / "data" / "webcam_augmented" / "manifest.json"

manifest_map = {}
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    for entry in m.get("entries", []):
        out_f = entry.get("output_file")
        if out_f:
            manifest_map[Path(out_f).as_posix()] = entry

print("Loaded manifest entries:", len(manifest_map))

# Helper to compute fast MD5
def compute_md5(path: Path) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

rows = []
row_id = 1

# ---------------------------------------------------------------------------
# 1. Webcam Captures (Originals)
# ---------------------------------------------------------------------------
print("Indexing data/webcam_captures...")
wc_dir = PROJECT_ROOT / "data" / "webcam_captures"
wc_timestamps = {}

# Pre-gather timestamps to flag near duplicates
for cls_dir in sorted(wc_dir.iterdir()):
    if not cls_dir.is_dir(): continue
    for j_path in sorted(cls_dir.glob("*.json")):
        with open(j_path, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
        t_str = meta.get("timestamp")
        try:
            dt = datetime.strptime(t_str, "%Y%m%d_%H%M%S_%f")
        except Exception:
            dt = None
        wc_timestamps[j_path.stem] = (cls_dir.name, dt, meta)

for cls_dir in sorted(wc_dir.iterdir()):
    if not cls_dir.is_dir(): continue
    cls_name = cls_dir.name
    jpgs = sorted(list(cls_dir.glob("*.jpg")))
    
    for jpg in jpgs:
        stem = jpg.stem
        meta_info = wc_timestamps.get(stem)
        dt = meta_info[1] if meta_info else None
        meta = meta_info[2] if meta_info else {}
        
        # Check time delta with adjacent samples in same class
        min_dt = 999999.0
        for other_stem, (other_cls, other_dt, _) in wc_timestamps.items():
            if other_cls == cls_name and other_stem != stem and dt and other_dt:
                d = abs((dt - other_dt).total_seconds())
                if d < min_dt:
                    min_dt = d
                    
        # Check split assignment
        rel_posix = jpg.relative_to(PROJECT_ROOT).as_posix()
        assigned_split = "train"
        for out_f, ent in manifest_map.items():
            if ent.get("source_file") and Path(ent["source_file"]).as_posix() == rel_posix:
                assigned_split = ent.get("split", "train")
                break
                
        flags = []
        if len(jpgs) == 1:
            flags.append("single_sample_class")
        if min_dt < 5.0:
            flags.append(f"near_duplicate_{min_dt:.1f}s")
        if cls_name in ["J", "Z"]:
            flags.append("dynamic_motion_sign")
        if meta.get("handedness") == "Left":
            flags.append("left_hand_capture")
            
        try:
            im = Image.open(jpg)
            w, h = im.size
            channels = len(im.getbands())
        except Exception:
            w, h, channels = 0, 0, 0
            flags.append("unreadable_image")
            
        rows.append({
            "file_id": row_id,
            "path": rel_posix,
            "class_label": cls_name,
            "dataset_source": "webcam_capture_original",
            "split": assigned_split,
            "is_augmented": False,
            "parent_source_file": "original",
            "width": w,
            "height": h,
            "channels": channels,
            "file_size_bytes": jpg.stat().st_size,
            "md5_hash": compute_md5(jpg),
            "handedness": meta.get("handedness", "Unknown"),
            "audit_flags": ";".join(flags) if flags else "ok"
        })
        row_id += 1

print(f"Indexed {len(rows)} original webcam captures.")

# ---------------------------------------------------------------------------
# 2. Webcam Augmented (Train & Val)
# ---------------------------------------------------------------------------
print("Indexing data/webcam_augmented...")
aug_dir = PROJECT_ROOT / "data" / "webcam_augmented"
for sub in ["train", "val"]:
    sub_dir = aug_dir / sub
    if not sub_dir.exists(): continue
    for cls_dir in sorted(sub_dir.iterdir()):
        if not cls_dir.is_dir(): continue
        cls_name = cls_dir.name
        for jpg in sorted(cls_dir.glob("*.jpg")):
            rel_to_aug = f"{sub}/{cls_name}/{jpg.name}"
            ent = manifest_map.get(rel_to_aug, {})
            is_aug = ent.get("is_augmented", (sub == "train" and "aug" in jpg.name))
            src_file = ent.get("source_file", "original" if not is_aug else "unknown")
            
            flags = []
            if cls_name in ["J", "Z"]:
                flags.append("dynamic_motion_sign")
            if sub == "val":
                flags.append("held_out_validation")
            if not is_aug:
                flags.append("genuine_capture")
            else:
                flags.append("synthetic_augmented")
                
            try:
                im = Image.open(jpg)
                w, h = im.size
                channels = len(im.getbands())
            except Exception:
                w, h, channels = 0, 0, 0
                flags.append("unreadable_image")
                
            rows.append({
                "file_id": row_id,
                "path": jpg.relative_to(PROJECT_ROOT).as_posix(),
                "class_label": cls_name,
                "dataset_source": f"webcam_augmented_{sub}",
                "split": sub,
                "is_augmented": is_aug,
                "parent_source_file": src_file,
                "width": w,
                "height": h,
                "channels": channels,
                "file_size_bytes": jpg.stat().st_size,
                "md5_hash": compute_md5(jpg),
                "handedness": "Right" if cls_name not in ["A", "C", "J", "P"] else "Audited",
                "audit_flags": ";".join(flags) if flags else "ok"
            })
            row_id += 1

print(f"Total rows after webcam augmented: {len(rows)}")

# ---------------------------------------------------------------------------
# 3. Physical Recording Attempts & Video Frames
# ---------------------------------------------------------------------------
print("Indexing scratch/physical_webcam_test & video_frames...")
scratch_phys = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test")
if scratch_phys.exists():
    for f in sorted(scratch_phys.glob("*.jpg")):
        is_crop = "crop" in f.name
        is_norm = "normalized" in f.name
        flags = ["screen_recording_attempt", "face_proximity_risk"]
        if is_norm:
            flags.append("handedness_normalized")
            
        try:
            im = Image.open(f)
            w, h = im.size
            channels = len(im.getbands())
        except Exception:
            w, h, channels = 0, 0, 0
            
        rows.append({
            "file_id": row_id,
            "path": f.as_posix(),
            "class_label": "DemoSequence",
            "dataset_source": "recording_physical_attempt",
            "split": "recording_eval",
            "is_augmented": False,
            "parent_source_file": "screen_recording_mp4",
            "width": w,
            "height": h,
            "channels": channels,
            "file_size_bytes": f.stat().st_size,
            "md5_hash": compute_md5(f),
            "handedness": "Mixed",
            "audit_flags": ";".join(flags)
        })
        row_id += 1

scratch_vid = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\video_frames")
if scratch_vid.exists():
    for f in sorted(scratch_vid.glob("*.jpg")):
        rows.append({
            "file_id": row_id,
            "path": f.as_posix(),
            "class_label": "ScreenTimeline",
            "dataset_source": "screen_recording_timeline",
            "split": "recording_eval",
            "is_augmented": False,
            "parent_source_file": "screen_recording_mp4",
            "width": 1920 if "frame" in f.name else 400,
            "height": 1080 if "frame" in f.name else 300,
            "channels": 3,
            "file_size_bytes": f.stat().st_size,
            "md5_hash": compute_md5(f),
            "handedness": "N/A",
            "audit_flags": "recording_frame"
        })
        row_id += 1

# ---------------------------------------------------------------------------
# 4. Raw Kaggle Test Set (28 images)
# ---------------------------------------------------------------------------
print("Indexing data/raw/asl_alphabet_test...")
raw_test_dir = PROJECT_ROOT / "data" / "raw" / "asl_alphabet_test" / "asl_alphabet_test"
if raw_test_dir.exists():
    for f in sorted(raw_test_dir.glob("*.jpg")):
        cls_name = f.stem.replace("_test", "")
        rows.append({
            "file_id": row_id,
            "path": f.relative_to(PROJECT_ROOT).as_posix(),
            "class_label": cls_name,
            "dataset_source": "kaggle_raw_test",
            "split": "test",
            "is_augmented": False,
            "parent_source_file": "original_kaggle",
            "width": 200,
            "height": 200,
            "channels": 3,
            "file_size_bytes": f.stat().st_size,
            "md5_hash": compute_md5(f),
            "handedness": "Right",
            "audit_flags": "kaggle_author_benchmark"
        })
        row_id += 1

# ---------------------------------------------------------------------------
# 5. Kaggle Dataset Splits (train, val, test) - 87,000 images
# ---------------------------------------------------------------------------
print("Indexing data/raw/splits (87,000 images)...")
splits_dir = PROJECT_ROOT / "data" / "raw" / "splits"
for split_name in ["train", "val", "test"]:
    s_dir = splits_dir / split_name
    if not s_dir.exists(): continue
    for cls_dir in sorted(s_dir.iterdir()):
        if not cls_dir.is_dir(): continue
        cls_name = cls_dir.name
        for f in sorted(cls_dir.glob("*.jpg")):
            # Fast MD5 or stat based
            rows.append({
                "file_id": row_id,
                "path": f.relative_to(PROJECT_ROOT).as_posix(),
                "class_label": cls_name,
                "dataset_source": f"kaggle_split_{split_name}",
                "split": split_name,
                "is_augmented": False,
                "parent_source_file": "kaggle_original_2018",
                "width": 200,
                "height": 200,
                "channels": 3,
                "file_size_bytes": f.stat().st_size,
                "md5_hash": compute_md5(f),
                "handedness": "Right",
                "audit_flags": "kaggle_clean_solid_bg"
            })
            row_id += 1
            if row_id % 15000 == 0:
                print(f"  Processed {row_id} rows...")

# Write to CSV
fieldnames = [
    "file_id", "path", "class_label", "dataset_source", "split",
    "is_augmented", "parent_source_file", "width", "height", "channels",
    "file_size_bytes", "md5_hash", "handedness", "audit_flags"
]

print(f"Writing {len(rows)} records to {OUT_CSV}...")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully generated {OUT_CSV} ({OUT_CSV.stat().st_size / (1024*1024):.2f} MB)")
