#!/usr/bin/env python3
"""
ASL Alphabet Dataset Validation Script

Validates the Kaggle ASL Alphabet dataset:
- Verifies all 29 expected classes are present
- Checks image integrity and decodability
- Reports class counts and image properties
- Identifies corrupt files and exact duplicates
"""

import os
import json
import hashlib
from pathlib import Path
from PIL import Image
from collections import Counter
import sys


EXPECTED_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

DATA_ROOT = Path(r"C:\Users\Ganesh\sign-language-cnn\data\raw")
TRAIN_DIR = DATA_ROOT / "asl_alphabet_train" / "asl_alphabet_train"
TEST_DIR = DATA_ROOT / "asl_alphabet_test" / "asl_alphabet_test"
OUTPUT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\metrics")


def validate_dataset():
    """Run comprehensive dataset validation."""
    results = {
        "dataset_source": "Kaggle ASL Alphabet (grassknoted/asl-alphabet)",
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(TEST_DIR),
        "expected_classes": EXPECTED_CLASSES,
        "validation_results": {}
    }

    print("=" * 60)
    print("ASL Alphabet Dataset Validation")
    print("=" * 60)

    if not TRAIN_DIR.exists():
        print(f"ERROR: Train directory not found: {TRAIN_DIR}")
        return results

    train_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    results["validation_results"]["train_classes_found"] = train_classes
    results["validation_results"]["train_class_count"] = len(train_classes)

    print(f"\nTrain classes found ({len(train_classes)}):")
    for c in train_classes:
        print(f"  {c}")

    missing = set(EXPECTED_CLASSES) - set(train_classes)
    extra = set(train_classes) - set(EXPECTED_CLASSES)
    if missing:
        print(f"\nWARNING: Missing expected classes: {missing}")
        results["validation_results"]["missing_classes"] = list(missing)
    if extra:
        print(f"\nINFO: Extra classes found: {extra}")
        results["validation_results"]["extra_classes"] = list(extra)

    class_counts = {}
    image_dims = {}
    image_modes = {}
    corrupt_files = []
    all_hashes = {}
    duplicates = []

    print("\nValidating train images...")
    total_images = 0
    for cls in train_classes:
        cls_path = TRAIN_DIR / cls
        files = [f for f in cls_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        class_counts[cls] = len(files)
        total_images += len(files)

        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    img.load()
                    dims = img.size
                    mode = img.mode
                    file_hash = hashlib.md5(img.tobytes()).hexdigest()

                    if dims not in image_dims:
                        image_dims[dims] = 0
                    image_dims[dims] += 1
                    if mode not in image_modes:
                        image_modes[mode] = 0
                    image_modes[mode] += 1

                    if file_hash in all_hashes:
                        duplicates.append((str(f), all_hashes[file_hash]))
                    else:
                        all_hashes[file_hash] = str(f)
            except Exception as e:
                corrupt_files.append({"file": str(f), "error": str(e)})

    results["validation_results"]["train_class_counts"] = class_counts
    results["validation_results"]["train_total_images"] = total_images
    results["validation_results"]["image_dimensions"] = {f"{k[0]}x{k[1]}": v for k, v in image_dims.items()}
    results["validation_results"]["image_modes"] = image_modes
    results["validation_results"]["corrupt_files"] = corrupt_files
    results["validation_results"]["exact_duplicates"] = len(duplicates)
    results["validation_results"]["duplicate_pairs"] = duplicates[:20] if duplicates else []

    print(f"\nTrain total images: {total_images}")
    print(f"\nClass counts:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

    print(f"\nImage dimensions found:")
    for dims, count in sorted(image_dims.items()):
        print(f"  {dims}: {count}")

    print(f"\nImage modes found:")
    for mode, count in sorted(image_modes.items()):
        print(f"  {mode}: {count}")

    print(f"\nCorrupt files: {len(corrupt_files)}")
    if corrupt_files:
        for c in corrupt_files[:10]:
            print(f"  {c['file']}: {c['error']}")

    print(f"\nExact duplicates: {len(duplicates)}")
    if duplicates:
        for d in duplicates[:10]:
            print(f"  {d[0]} == {d[1]}")

    if TEST_DIR.exists():
        test_files = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        test_classes = [f.stem.replace('_test', '') for f in test_files]
        results["validation_results"]["test_image_count"] = len(test_files)
        results["validation_results"]["test_classes"] = sorted(test_classes)
        print(f"\nTest set: {len(test_files)} images")
        print(f"Test classes: {sorted(test_classes)}")
        missing_test = set(EXPECTED_CLASSES) - set(test_classes)
        if missing_test:
            print(f"Missing from test set: {missing_test}")
            results["validation_results"]["test_missing_classes"] = list(missing_test)
    else:
        print("\nTest directory not found")
        results["validation_results"]["test_image_count"] = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "dataset_validation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nValidation results saved to: {output_file}")

    return results


if __name__ == "__main__":
    validate_dataset()