#!/usr/bin/env python3
"""
ASL Alphabet Dataset Splitting Script

Creates reproducible, stratified train/validation/test splits (70/15/15)
with leakage prevention (exact duplicates kept in same split).
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import hashlib
from PIL import Image
import sys


EXPECTED_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

DATA_ROOT = Path(r"C:\Users\Ganesh\sign-language-cnn\data\raw")
TRAIN_DIR = DATA_ROOT / "asl_alphabet_train" / "asl_alphabet_train"
SPLIT_DIR = DATA_ROOT / "splits"
OUTPUT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\metrics")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def compute_image_hash(image_path):
    """Compute MD5 hash of image pixels for duplicate detection."""
    with Image.open(image_path) as img:
        img.load()
        return hashlib.md5(img.tobytes()).hexdigest()


def collect_all_images():
    """Collect all images with metadata from train directory."""
    images = []
    for cls in EXPECTED_CLASSES:
        cls_path = TRAIN_DIR / cls
        if not cls_path.exists():
            continue
        for img_file in cls_path.iterdir():
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                images.append({
                    "path": str(img_file),
                    "class": cls,
                    "filename": img_file.name
                })
    return images


def detect_duplicates(images):
    """Detect exact duplicates by image hash."""
    hash_to_indices = defaultdict(list)
    for idx, img_info in enumerate(images):
        try:
            h = compute_image_hash(img_info["path"])
            hash_to_indices[h].append(idx)
        except Exception:
            pass
    duplicate_groups = [indices for indices in hash_to_indices.values() if len(indices) > 1]
    return duplicate_groups


def create_stratified_splits(images, duplicate_groups):
    """Create stratified splits keeping duplicates together."""
    random.seed(RANDOM_SEED)
    
    class_to_indices = defaultdict(list)
    for idx, img in enumerate(images):
        class_to_indices[img["class"]].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    duplicate_indices_set = set()
    for group in duplicate_groups:
        for idx in group:
            duplicate_indices_set.add(idx)
    
    for cls in EXPECTED_CLASSES:
        cls_indices = class_to_indices[cls]
        if not cls_indices:
            continue
        
        non_dup = [i for i in cls_indices if i not in duplicate_indices_set]
        dup_groups_cls = [g for g in duplicate_groups if all(images[i]["class"] == cls for i in g)]
        
        random.shuffle(non_dup)
        n_cls = len(cls_indices)
        n_train = int(n_cls * TRAIN_RATIO)
        n_val = int(n_cls * VAL_RATIO)
        n_test = n_cls - n_train - n_val
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        for idx in non_dup:
            if train_count < n_train:
                train_indices.append(idx)
                train_count += 1
            elif val_count < n_val:
                val_indices.append(idx)
                val_count += 1
            else:
                test_indices.append(idx)
                test_count += 1
        
        for group in dup_groups_cls:
            if len(group) == 0:
                continue
            if train_count + len(group) <= n_train:
                train_indices.extend(group)
                train_count += len(group)
            elif val_count + len(group) <= n_val:
                val_indices.extend(group)
                val_count += len(group)
            else:
                test_indices.extend(group)
                test_count += len(group)
    
    return train_indices, val_indices, test_indices


def save_split_file(indices, images, output_path):
    """Save split as JSON with image paths and labels."""
    split_data = {
        "indices": indices,
        "images": [{"path": images[i]["path"], "class": images[i]["class"], "filename": images[i]["filename"]} for i in indices],
        "class_distribution": {}
    }
    for item in split_data["images"]:
        cls = item["class"]
        split_data["class_distribution"][cls] = split_data["class_distribution"].get(cls, 0) + 1
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)


def copy_images_to_split_dirs(train_idx, val_idx, test_idx, images):
    """Copy images to organized split directories."""
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }
    
    for split_name, indices in splits.items():
        split_path = SPLIT_DIR / split_name
        for idx in indices:
            img_info = images[idx]
            cls = img_info["class"]
            dest_dir = split_path / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / img_info["filename"]
            if not dest_file.exists():
                shutil.copy2(img_info["path"], dest_file)


def verify_splits():
    """Verify split integrity and check for leakage."""
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split_name
        if not split_path.exists():
            continue
        class_counts = {}
        for cls_dir in split_path.iterdir():
            if cls_dir.is_dir():
                files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
                class_counts[cls_dir.name] = len(files)
        splits[split_name] = class_counts
    return splits


def check_leakage(train_idx, val_idx, test_idx, images):
    """Check for exact duplicates across splits."""
    def get_hashes(indices):
        hashes = set()
        for idx in indices:
            try:
                h = compute_image_hash(images[idx]["path"])
                hashes.add(h)
            except Exception:
                pass
        return hashes
    
    train_hashes = get_hashes(train_idx)
    val_hashes = get_hashes(val_idx)
    test_hashes = get_hashes(test_idx)
    
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    return {
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "details": {
            "train_val": list(train_val_overlap)[:5],
            "train_test": list(train_test_overlap)[:5],
            "val_test": list(val_test_overlap)[:5]
        }
    }


def main():
    print("=" * 60)
    print("ASL Alphabet Dataset Splitting (70/15/15)")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    
    images = collect_all_images()
    print(f"\nTotal images collected: {len(images)}")
    
    class_counts = defaultdict(int)
    for img in images:
        class_counts[img["class"]] += 1
    print("Class distribution:")
    for cls in EXPECTED_CLASSES:
        print(f"  {cls}: {class_counts.get(cls, 0)}")
    
    print("\nDetecting duplicates...")
    duplicate_groups = detect_duplicates(images)
    print(f"Found {len(duplicate_groups)} duplicate groups")
    for group in duplicate_groups[:5]:
        print(f"  Group: {[images[i]['filename'] for i in group]}")
    
    print("\nCreating stratified splits...")
    train_idx, val_idx, test_idx = create_stratified_splits(images, duplicate_groups)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(images)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(images)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(images)*100:.1f}%)")
    
    train_dist = defaultdict(int)
    val_dist = defaultdict(int)
    test_dist = defaultdict(int)
    for idx in train_idx:
        train_dist[images[idx]["class"]] += 1
    for idx in val_idx:
        val_dist[images[idx]["class"]] += 1
    for idx in test_idx:
        test_dist[images[idx]["class"]] += 1
    
    print("\nPer-class split distribution:")
    for cls in EXPECTED_CLASSES:
        print(f"  {cls}: train={train_dist.get(cls,0)}, val={val_dist.get(cls,0)}, test={test_dist.get(cls,0)}")
    
    print("\nChecking for leakage...")
    leakage = check_leakage(train_idx, val_idx, test_idx, images)
    print(f"  Train-Val overlap: {leakage['train_val_overlap']}")
    print(f"  Train-Test overlap: {leakage['train_test_overlap']}")
    print(f"  Val-Test overlap: {leakage['val_test_overlap']}")
    
    print("\nSaving split index files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split_file(train_idx, images, OUTPUT_DIR / "train_split.json")
    save_split_file(val_idx, images, OUTPUT_DIR / "val_split.json")
    save_split_file(test_idx, images, OUTPUT_DIR / "test_split.json")
    
    print("\nCopying images to split directories...")
    copy_images_to_split_dirs(train_idx, val_idx, test_idx, images)
    
    print("\nVerifying split directories...")
    splits = verify_splits()
    for split_name, counts in splits.items():
        total = sum(counts.values())
        print(f"  {split_name}: {total} images")
        for cls, cnt in sorted(counts.items()):
            print(f"    {cls}: {cnt}")
    
    results = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "total_images": len(images),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "class_distribution": {
            "train": dict(train_dist),
            "val": dict(val_dist),
            "test": dict(test_dist)
        },
        "duplicate_groups": len(duplicate_groups),
        "leakage_check": leakage,
        "split_dir": str(SPLIT_DIR)
    }
    
    with open(OUTPUT_DIR / "dataset_splits.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSplit results saved to: {OUTPUT_DIR / 'dataset_splits.json'}")
    print(f"Split images organized in: {SPLIT_DIR}")
    
    return results


if __name__ == "__main__":
    main()