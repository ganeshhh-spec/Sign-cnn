import json
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Load split info
split_info_path = PROJECT_ROOT / "data" / "webcam_augmented" / "split_info.json"
with open(split_info_path, "r", encoding="utf-8") as f:
    split_info = json.load(f)

# Load manifest
manifest_path = PROJECT_ROOT / "data" / "webcam_augmented" / "manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

manifest_map = {}
for entry in manifest.get("entries", []):
    src = entry.get("source_file")
    if src and not entry.get("is_augmented"):
        manifest_map[Path(src).as_posix()] = entry.get("split")

wc_dir = PROJECT_ROOT / "data" / "webcam_captures"
classes = sorted([d.name for d in wc_dir.iterdir() if d.is_dir()])

print(f"Total classes found in webcam captures: {len(classes)}")

audit_records = []

for cls_name in classes:
    cls_dir = wc_dir / cls_name
    jpgs = sorted(list(cls_dir.glob("*.jpg")))
    jsons = sorted(list(cls_dir.glob("*.json")))
    
    timestamps = []
    for j_path in jsons:
        with open(j_path, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
        t_str = meta.get("timestamp")
        try:
            dt = datetime.strptime(t_str, "%Y%m%d_%H%M%S_%f")
        except Exception:
            dt = None
        timestamps.append((j_path.stem, dt, meta))
        
    for i, (stem, dt, meta) in enumerate(timestamps):
        img_path = cls_dir / f"{stem}.jpg"
        img = Image.open(img_path)
        w, h = img.size
        
        # Check time delta with previous or next sample in same class
        min_delta_sec = 999999.0
        if i > 0 and timestamps[i-1][1] and dt:
            d_prev = abs((dt - timestamps[i-1][1]).total_seconds())
            min_delta_sec = min(min_delta_sec, d_prev)
        if i < len(timestamps) - 1 and timestamps[i+1][1] and dt:
            d_next = abs((timestamps[i+1][1] - dt).total_seconds())
            min_delta_sec = min(min_delta_sec, d_next)
            
        # Determine assigned split
        rel_posix = img_path.relative_to(PROJECT_ROOT).as_posix()
        assigned_split = manifest_map.get(rel_posix, "unknown")
        
        # Determine audit flags
        flags = []
        if len(jpgs) == 1:
            flags.append("single_sample_class")
        if min_delta_sec < 5.0:
            flags.append(f"near_duplicate_{min_delta_sec:.1f}s")
        if cls_name in ["J", "Z"]:
            flags.append("dynamic_motion_sign")
        if meta.get("handedness") == "Left":
            flags.append("left_hand_capture")
            
        audit_records.append({
            "class": cls_name,
            "filename": img_path.name,
            "dimensions": f"{w}x{h}",
            "handedness": meta.get("handedness"),
            "bbox": meta.get("bbox"),
            "time_delta_sec": round(min_delta_sec, 2) if min_delta_sec < 999999 else None,
            "assigned_split": assigned_split,
            "flags": flags
        })

print("\n--- Summary of Webcam Capture Audit Records ---")
for r in audit_records:
    flag_str = f" [{', '.join(r['flags'])}]" if r['flags'] else ""
    dt_str = f"dt={r['time_delta_sec']}s" if r['time_delta_sec'] is not None else "isolated"
    print(f"Class {r['class']:2s} | {r['filename']} | {r['dimensions']:7s} | {r['handedness']:5s} | Split: {r['assigned_split']:5s} | {dt_str:10s}{flag_str}")
