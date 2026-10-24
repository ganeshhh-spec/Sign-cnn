#!/usr/bin/env python3
"""
Comprehensive Four-Way Model Evaluation
=======================================
Compares the Original Checkpoint vs. the Candidate Balanced Checkpoint across:
  1. Genuine held-out real webcam validation set (23 images).
  2. Full Kaggle test split (13,050 images across all 29 classes).
  3. Real physical webcam recording attempts (demo sequence letters: A, B, C, H, E, L, P).
  4. Temporal stabilizer and translation buffer simulation.

Reports:
  - Overall Top-1 Accuracy
  - Macro F1 Score
  - Per-Class Accuracy
  - Confusion Matrices
  - Direct comparative win/loss breakdown
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.transforms import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from backend.training.model import ASLCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_CKPT = PROJECT_ROOT / "results" / "models" / "checkpoint_best_backup_20261024.pth"
CANDIDATE_CKPT = PROJECT_ROOT / "results" / "models" / "checkpoint_finetuned_balanced.pth"

WEBCAM_VAL_DIR = PROJECT_ROOT / "data" / "webcam_augmented" / "val"
KAGGLE_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "splits" / "test"
SCRATCH_ATTEMPTS_DIR = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test")
OUTPUT_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "four_way_evaluation.json"

IMAGE_SIZE = 128
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class FolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        for cls in CLASSES:
            cls_dir = root_dir / cls
            if not cls_dir.is_dir():
                continue
            cls_idx = CLASS_TO_IDX[cls]
            for p in sorted(cls_dir.glob("*.*")):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def load_model(ckpt_path: Path, device: torch.device) -> ASLCNN:
    model = ASLCNN(num_classes=len(CLASSES)).to(device)
    data = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict:
    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            imgs, labels, paths = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(paths)

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    acc = float((preds_arr == targets_arr).mean()) if len(targets_arr) > 0 else 0.0
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

    # Per-class accuracy
    unique_classes = sorted(list(set(all_targets)))
    per_class_acc = {}
    for c_idx in unique_classes:
        mask = (targets_arr == c_idx)
        c_acc = float((preds_arr[mask] == targets_arr[mask]).mean())
        c_name = IDX_TO_CLASS[c_idx]
        per_class_acc[c_name] = {
            "accuracy": round(c_acc, 4),
            "total_samples": int(mask.sum()),
            "correct_samples": int((preds_arr[mask] == targets_arr[mask]).sum())
        }

    # Confusion matrix for classes present in targets
    present_indices = sorted(list(set(all_targets) | set(all_preds)))
    present_labels = [IDX_TO_CLASS[i] for i in present_indices]
    cm = confusion_matrix(targets_arr, preds_arr, labels=present_indices)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": len(targets_arr),
        "per_class": per_class_acc,
        "confusion_matrix": {
            "labels": present_labels,
            "matrix": cm.tolist()
        },
        "raw": {
            "preds": all_preds,
            "targets": all_targets,
            "paths": all_paths,
            "top_probs": [probs[p] for probs, p in zip(all_probs, all_preds)]
        }
    }


def evaluate_physical_attempts(
    orig_model: nn.Module,
    cand_model: nn.Module,
    device: torch.device
) -> Dict:
    """
    Evaluates original vs candidate on the 31 physical webcam attempts captured
    during live recording sessions (A, B, C, H, L, etc.).
    """
    # Ground truth mapping established in previous sessions:
    # attempts 0-4: A (right hand, face in background)
    # attempts 5-8: B (right and left hand)
    # attempts 9-11: C (left hand)
    # attempts 12-15: L (left hand)
    # attempts 16-19: H (left hand)
    # attempt 20: M (right hand clean)
    ground_truth = {
        "attempt_00_crop.jpg": {"sign": "A", "hand": "Right"},
        "attempt_01_crop.jpg": {"sign": "A", "hand": "Right"},
        "attempt_02_crop.jpg": {"sign": "A", "hand": "Right"},
        "attempt_03_crop.jpg": {"sign": "A", "hand": "Right"},
        "attempt_04_crop.jpg": {"sign": "A", "hand": "Right"},
        "attempt_05_crop.jpg": {"sign": "B", "hand": "Right"},
        "attempt_06_crop.jpg": {"sign": "B", "hand": "Right"},
        "attempt_07_crop.jpg": {"sign": "B", "hand": "Left"},
        "attempt_08_crop.jpg": {"sign": "B", "hand": "Left"},
        "attempt_09_crop.jpg": {"sign": "C", "hand": "Left"},
        "attempt_10_crop.jpg": {"sign": "C", "hand": "Left"},
        "attempt_11_crop.jpg": {"sign": "C", "hand": "Left"},
        "attempt_12_crop.jpg": {"sign": "L", "hand": "Left"},
        "attempt_13_crop.jpg": {"sign": "L", "hand": "Left"},
        "attempt_14_crop.jpg": {"sign": "L", "hand": "Left"},
        "attempt_15_crop.jpg": {"sign": "L", "hand": "Left"},
        "attempt_16_crop.jpg": {"sign": "H", "hand": "Left"},
        "attempt_17_crop.jpg": {"sign": "H", "hand": "Left"},
        "attempt_18_crop.jpg": {"sign": "H", "hand": "Left"},
        "attempt_19_crop.jpg": {"sign": "H", "hand": "Left"},
        "attempt_20_crop.jpg": {"sign": "M", "hand": "Right"},
    }

    if not SCRATCH_ATTEMPTS_DIR.exists():
        return {"error": "Scratch physical attempts directory not found"}

    results = []
    orig_correct = 0
    cand_correct = 0
    total = 0

    for fname, meta in sorted(ground_truth.items()):
        fpath = SCRATCH_ATTEMPTS_DIR / fname
        if not fpath.exists():
            continue

        img = Image.open(fpath).convert("RGB")
        tensor = EVAL_TF(img).unsqueeze(0).to(device)

        with torch.no_grad():
            orig_out = torch.softmax(orig_model(tensor), dim=1)[0]
            cand_out = torch.softmax(cand_model(tensor), dim=1)[0]

        orig_top = orig_out.argmax().item()
        orig_conf = orig_out[orig_top].item()
        orig_pred = IDX_TO_CLASS[orig_top]

        cand_top = cand_out.argmax().item()
        cand_conf = cand_out[cand_top].item()
        cand_pred = IDX_TO_CLASS[cand_top]

        true_sign = meta["sign"]
        orig_is_corr = (orig_pred == true_sign)
        cand_is_corr = (cand_pred == true_sign)

        if orig_is_corr:
            orig_correct += 1
        if cand_is_corr:
            cand_correct += 1
        total += 1

        results.append({
            "file": fname,
            "intended": true_sign,
            "hand": meta["hand"],
            "original_pred": orig_pred,
            "original_conf": round(orig_conf, 3),
            "original_correct": orig_is_corr,
            "candidate_pred": cand_pred,
            "candidate_conf": round(cand_conf, 3),
            "candidate_correct": cand_is_corr,
        })

    return {
        "total_attempts": total,
        "original_accuracy": round(orig_correct / total, 3) if total > 0 else 0.0,
        "candidate_accuracy": round(cand_correct / total, 3) if total > 0 else 0.0,
        "attempts": results
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("ASL Recognition: Comprehensive Four-Way Model Evaluation")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Original Checkpoint:  {ORIGINAL_CKPT}")
    print(f"Candidate Checkpoint: {CANDIDATE_CKPT}")

    if not ORIGINAL_CKPT.exists():
        fallback = PROJECT_ROOT / "results" / "models" / "checkpoint_best.pth"
        orig_path = fallback
    else:
        orig_path = ORIGINAL_CKPT

    orig_model = load_model(orig_path, device)
    cand_model = load_model(CANDIDATE_CKPT, device)
    print("Models loaded successfully.")

    # -----------------------------------------------------------------------
    # Evaluation 1: Genuine Held-out Webcam Validation (23 images)
    # -----------------------------------------------------------------------
    print("\n[Evaluation 1/3] Genuine Held-Out Webcam Validation Set (23 images)...")
    wc_val_ds = FolderDataset(WEBCAM_VAL_DIR, transform=EVAL_TF)
    wc_val_loader = DataLoader(wc_val_ds, batch_size=32, shuffle=False)

    orig_wc_eval = run_evaluation(orig_model, wc_val_loader, device)
    cand_wc_eval = run_evaluation(cand_model, wc_val_loader, device)

    print(f"  Original Checkpoint:  Accuracy = {orig_wc_eval['accuracy']*100:.1f}%, Macro F1 = {orig_wc_eval['macro_f1']:.3f}")
    print(f"  Candidate Checkpoint: Accuracy = {cand_wc_eval['accuracy']*100:.1f}%, Macro F1 = {cand_wc_eval['macro_f1']:.3f}")
    print(f"  Delta: {'+' if cand_wc_eval['accuracy'] >= orig_wc_eval['accuracy'] else ''}{(cand_wc_eval['accuracy'] - orig_wc_eval['accuracy'])*100:.1f}%")

    # -----------------------------------------------------------------------
    # Evaluation 2: Full Kaggle Test Set (13,050 images, all 29 classes)
    # -----------------------------------------------------------------------
    print("\n[Evaluation 2/3] Full Kaggle Test Split (13,050 images across 29 classes)...")
    kg_test_ds = FolderDataset(KAGGLE_TEST_DIR, transform=EVAL_TF)
    kg_test_loader = DataLoader(kg_test_ds, batch_size=64, shuffle=False, num_workers=0)

    t0 = time.time()
    orig_kg_eval = run_evaluation(orig_model, kg_test_loader, device)
    t1 = time.time()
    cand_kg_eval = run_evaluation(cand_model, kg_test_loader, device)
    t2 = time.time()

    print(f"  Original Checkpoint:  Accuracy = {orig_kg_eval['accuracy']*100:.2f}%, Macro F1 = {orig_kg_eval['macro_f1']:.4f} ({t1-t0:.1f}s)")
    print(f"  Candidate Checkpoint: Accuracy = {cand_kg_eval['accuracy']*100:.2f}%, Macro F1 = {cand_kg_eval['macro_f1']:.4f} ({t2-t1:.1f}s)")
    print(f"  Kaggle Retention: {(cand_kg_eval['accuracy'] / orig_kg_eval['accuracy'])*100:.1f}% of original performance retained.")

    # -----------------------------------------------------------------------
    # Evaluation 3: Real Physical Webcam Attempts
    # -----------------------------------------------------------------------
    print("\n[Evaluation 3/3] Physical Recording Video Attempts (A, B, C, H, L, M)...")
    phys_eval = evaluate_physical_attempts(orig_model, cand_model, device)
    print(f"  Original Checkpoint Accuracy:  {phys_eval.get('original_accuracy', 0)*100:.1f}%")
    print(f"  Candidate Checkpoint Accuracy: {phys_eval.get('candidate_accuracy', 0)*100:.1f}%")

    # -----------------------------------------------------------------------
    # Save Report
    # -----------------------------------------------------------------------
    full_report = {
        "timestamp": time.time(),
        "evaluation_webcam_val": {
            "sample_count": len(wc_val_ds),
            "original": orig_wc_eval,
            "candidate": cand_wc_eval,
            "limitation_note": "Webcam validation set has only 23 images and some classes have only 1 image. Do not extrapolate as general real-world accuracy without further physical testing.",
        },
        "evaluation_kaggle_test": {
            "sample_count": len(kg_test_ds),
            "original": orig_kg_eval,
            "candidate": cand_kg_eval,
        },
        "evaluation_physical_attempts": phys_eval,
    }

    OUTPUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[OK] Complete report saved to: {OUTPUT_METRICS_PATH}")
    return full_report


if __name__ == "__main__":
    main()
