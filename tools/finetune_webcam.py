#!/usr/bin/env python3
"""
Balanced ASL Fine-Tuning Script
================================
Trains ASLCNN using a balanced two-stream batch loader:
  16 samples from Webcam Augmented Training set +
  16 samples from Kaggle Training set (covering all 29 classes).

Ensures 50% webcam gradient contribution per batch while regularizing
against catastrophic forgetting of existing Kaggle representations.

Original checkpoint and backup are preserved intact.
Candidate is saved to results/models/checkpoint_finetuned_balanced.pth.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocessing.transforms import CLASSES, CLASS_TO_IDX
from backend.training.model import ASLCNN

# ---------------------------------------------------------------------------
# Default Directories and Paths
# ---------------------------------------------------------------------------
WEBCAM_AUG_DIR = PROJECT_ROOT / "data" / "webcam_augmented" / "train"
WEBCAM_VAL_DIR = PROJECT_ROOT / "data" / "webcam_augmented" / "val"
KAGGLE_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "splits" / "train"
KAGGLE_VAL_DIR = PROJECT_ROOT / "data" / "raw" / "splits" / "val"

CHECKPOINT_ORIGINAL = PROJECT_ROOT / "results" / "models" / "checkpoint_best_backup_20261024.pth"
CHECKPOINT_OUT = PROJECT_ROOT / "results" / "models" / "checkpoint_finetuned_balanced.pth"
LOG_OUT = PROJECT_ROOT / "results" / "metrics" / "finetune_balanced_log.json"

IMAGE_SIZE = 128
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
WEBCAM_TRAIN_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
    transforms.RandomRotation(8, interpolation=Image.BILINEAR, fill=(0, 0, 0)),
    transforms.RandomAffine(0, translate=(0.04, 0.04), scale=(0.96, 1.04),
                             shear=3, interpolation=Image.BILINEAR, fill=(0, 0, 0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

KAGGLE_TRAIN_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
    transforms.RandomRotation(12, interpolation=Image.BILINEAR, fill=(0, 0, 0)),
    transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.92, 1.08),
                             shear=4, interpolation=Image.BILINEAR, fill=(0, 0, 0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """Dataset wrapping a list of (image_path, class_idx) tuples."""
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_dataset_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    """Loads all image files under subdirectories matching CLASSES."""
    samples = []
    if not root_dir.exists():
        return samples
    for cls in CLASSES:
        cls_dir = root_dir / cls
        if not cls_dir.is_dir():
            continue
        cls_idx = CLASS_TO_IDX[cls]
        for p in sorted(cls_dir.glob("*.*")):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((p, cls_idx))
    return samples


def load_stratified_kaggle_samples(
    kaggle_dir: Path,
    samples_per_class: int = 50,
    seed: int = 42
) -> List[Tuple[Path, int]]:
    """Loads a balanced subset of Kaggle samples across all 29 classes."""
    rng = random.Random(seed)
    selected = []
    for cls in CLASSES:
        cls_dir = kaggle_dir / cls
        if not cls_dir.is_dir():
            continue
        cls_idx = CLASS_TO_IDX[cls]
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        rng.shuffle(images)
        for img in images[:samples_per_class]:
            selected.append((img, cls_idx))
    return selected


# ---------------------------------------------------------------------------
# Balanced Dual-Stream Sampler / Loader
# ---------------------------------------------------------------------------
class BalancedDualStreamBatchLoader:
    """
    Yields batches of size (batch_size_webcam + batch_size_kaggle).
    Guarantees exactly half the batch is from the webcam domain
    and half from the Kaggle domain.
    """
    def __init__(
        self,
        webcam_dataset: Dataset,
        kaggle_dataset: Dataset,
        batch_size_webcam: int = 16,
        batch_size_kaggle: int = 16,
        batches_per_epoch: int = 80,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.batch_size_webcam = batch_size_webcam
        self.batch_size_kaggle = batch_size_kaggle
        self.batches_per_epoch = batches_per_epoch

        self.webcam_loader = DataLoader(
            webcam_dataset,
            batch_size=batch_size_webcam,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.kaggle_loader = DataLoader(
            kaggle_dataset,
            batch_size=batch_size_kaggle,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        webcam_iter = iter(self.webcam_loader)
        kaggle_iter = iter(self.kaggle_loader)

        for _ in range(self.batches_per_epoch):
            try:
                wc_imgs, wc_labels = next(webcam_iter)
            except StopIteration:
                webcam_iter = iter(self.webcam_loader)
                wc_imgs, wc_labels = next(webcam_iter)

            try:
                kg_imgs, kg_labels = next(kaggle_iter)
            except StopIteration:
                kaggle_iter = iter(self.kaggle_loader)
                kg_imgs, kg_labels = next(kaggle_iter)

            batch_imgs = torch.cat([wc_imgs, kg_imgs], dim=0)
            batch_labels = torch.cat([wc_labels, kg_labels], dim=0)

            yield batch_imgs, batch_labels


# ---------------------------------------------------------------------------
# Training & Evaluation Helpers
# ---------------------------------------------------------------------------
def set_backbone_grad(model: ASLCNN, requires_grad: bool):
    """Freezes or unfreezes backbone conv blocks; keeps classifier trainable."""
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = requires_grad


def train_one_epoch(
    model: nn.Module,
    dual_loader: BalancedDualStreamBatchLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in dual_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0
    return avg_loss, acc, all_preds, all_targets


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def run_fine_tuning(
    epochs_a: int = 5,
    epochs_b: int = 10,
    lr_a: float = 5e-4,
    lr_b: float = 3e-5,
    kaggle_samples_per_class: int = 50,
    batches_per_epoch: int = 80,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("Balanced ASL Fine-Tuning Run")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Random seed: {seed}")
    print(f"Original Checkpoint to fine-tune from: {CHECKPOINT_ORIGINAL}")
    print(f"Target Checkpoint: {CHECKPOINT_OUT}")

    # 1. Load Data
    print("\n[1/4] Loading Datasets...")
    wc_train_samples = load_dataset_samples(WEBCAM_AUG_DIR)
    wc_val_samples = load_dataset_samples(WEBCAM_VAL_DIR)
    kg_train_samples = load_stratified_kaggle_samples(KAGGLE_TRAIN_DIR, kaggle_samples_per_class, seed)
    kg_val_samples = load_stratified_kaggle_samples(KAGGLE_VAL_DIR, samples_per_class=20, seed=seed)

    print(f"  Webcam augmented train samples: {len(wc_train_samples)}")
    print(f"  Webcam genuine val samples:    {len(wc_val_samples)}")
    print(f"  Kaggle regularizer train samples: {len(kg_train_samples)} (all 29 classes)")
    print(f"  Kaggle val samples:            {len(kg_val_samples)}")

    if not wc_train_samples:
        raise RuntimeError(f"No webcam train samples found at {WEBCAM_AUG_DIR}. Run augment_webcam_data.py first.")

    wc_train_ds = FlatImageDataset(wc_train_samples, WEBCAM_TRAIN_TF)
    kg_train_ds = FlatImageDataset(kg_train_samples, KAGGLE_TRAIN_TF)
    wc_val_ds = FlatImageDataset(wc_val_samples, EVAL_TF)
    kg_val_ds = FlatImageDataset(kg_val_samples, EVAL_TF)

    dual_loader = BalancedDualStreamBatchLoader(
        webcam_dataset=wc_train_ds,
        kaggle_dataset=kg_train_ds,
        batch_size_webcam=16,
        batch_size_kaggle=16,
        batches_per_epoch=batches_per_epoch,
        pin_memory=(device.type == "cuda"),
    )

    wc_val_loader = DataLoader(wc_val_ds, batch_size=32, shuffle=False)
    kg_val_loader = DataLoader(kg_val_ds, batch_size=32, shuffle=False)

    # 2. Model Initialization (From Original Checkpoint Backup)
    print("\n[2/4] Initializing ASLCNN from original checkpoint...")
    if not CHECKPOINT_ORIGINAL.exists():
        fallback = PROJECT_ROOT / "results" / "models" / "checkpoint_best.pth"
        src_ckpt = fallback
    else:
        src_ckpt = CHECKPOINT_ORIGINAL

    print(f"  Loading state_dict from: {src_ckpt}")
    checkpoint_data = torch.load(src_ckpt, map_location=device, weights_only=True)
    model = ASLCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    print(f"  Loaded model successfully ({sum(p.numel() for p in model.parameters()):,} parameters).")

    criterion = nn.CrossEntropyLoss()
    history = {"phase_a": [], "phase_b": []}
    best_webcam_val_acc = 0.0
    best_state_dict = None
    best_epoch_info = {}

    # Initial Zero-Shot Baseline
    init_wc_loss, init_wc_acc, _, _ = evaluate(model, wc_val_loader, criterion, device)
    init_kg_loss, init_kg_acc, _, _ = evaluate(model, kg_val_loader, criterion, device)
    print(f"\nInitial Baseline: Webcam Val Acc = {init_wc_acc*100:.1f}%, Kaggle Val Acc = {init_kg_acc*100:.1f}%")

    # 3. Phase A: Classifier Head Only
    print("\n" + "=" * 50)
    print(f"Phase A — Classifier Head Training ({epochs_a} Epochs, LR={lr_a})")
    print("=" * 50)
    set_backbone_grad(model, requires_grad=False)
    opt_a = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_a, weight_decay=1e-4)
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs_a, eta_min=1e-5)

    for ep in range(epochs_a):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, dual_loader, criterion, opt_a, device)
        wc_loss, wc_acc, _, _ = evaluate(model, wc_val_loader, criterion, device)
        kg_loss, kg_acc, _, _ = evaluate(model, kg_val_loader, criterion, device)
        sched_a.step()
        dt = time.time() - t0

        print(f"  Ep {ep+1:2d}/{epochs_a} | Train Loss={tr_loss:.3f} Acc={tr_acc*100:.1f}% | "
              f"Webcam Val Acc={wc_acc*100:.1f}% | Kaggle Val Acc={kg_acc*100:.1f}% | {dt:.1f}s")

        history["phase_a"].append({
            "epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
            "webcam_val_acc": wc_acc, "webcam_val_loss": wc_loss,
            "kaggle_val_acc": kg_acc, "kaggle_val_loss": kg_loss,
        })

        if wc_acc >= best_webcam_val_acc and kg_acc >= 0.85:
            best_webcam_val_acc = wc_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch_info = {"phase": "A", "epoch": ep + 1, "wc_acc": wc_acc, "kg_acc": kg_acc}

    # 4. Phase B: Full Network Fine-Tuning
    print("\n" + "=" * 50)
    print(f"Phase B — Full Network Fine-Tuning ({epochs_b} Epochs, LR={lr_b})")
    print("=" * 50)
    set_backbone_grad(model, requires_grad=True)
    opt_b = optim.Adam(model.parameters(), lr=lr_b, weight_decay=1e-4)
    sched_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=epochs_b, eta_min=5e-6)

    for ep in range(epochs_b):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, dual_loader, criterion, opt_b, device)
        wc_loss, wc_acc, _, _ = evaluate(model, wc_val_loader, criterion, device)
        kg_loss, kg_acc, _, _ = evaluate(model, kg_val_loader, criterion, device)
        sched_b.step()
        dt = time.time() - t0

        print(f"  Ep {ep+1:2d}/{epochs_b} | Train Loss={tr_loss:.3f} Acc={tr_acc*100:.1f}% | "
              f"Webcam Val Acc={wc_acc*100:.1f}% | Kaggle Val Acc={kg_acc*100:.1f}% | {dt:.1f}s")

        history["phase_b"].append({
            "epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
            "webcam_val_acc": wc_acc, "webcam_val_loss": wc_loss,
            "kaggle_val_acc": kg_acc, "kaggle_val_loss": kg_loss,
        })

        if wc_acc >= best_webcam_val_acc and kg_acc >= 0.85:
            best_webcam_val_acc = wc_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch_info = {"phase": "B", "epoch": ep + 1, "wc_acc": wc_acc, "kg_acc": kg_acc}

    # Save Candidate Checkpoint
    if best_state_dict is None:
        best_state_dict = model.state_dict()
        best_epoch_info = {"phase": "final", "epoch": epochs_b, "wc_acc": wc_acc, "kg_acc": kg_acc}

    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state_dict,
        "selection_info": best_epoch_info,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "sampler": "BalancedDualStreamBatchLoader (16 webcam + 16 kaggle)",
        "seed": seed,
        "created_at": time.time(),
    }, CHECKPOINT_OUT)
    print(f"\n[OK] Candidate Checkpoint saved to: {CHECKPOINT_OUT}")
    print(f"  Selected model from {best_epoch_info}")

    # Also record training log
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "w") as f:
        json.dump({
            "seed": seed,
            "sampler_config": {
                "batch_size_webcam": 16,
                "batch_size_kaggle": 16,
                "batches_per_epoch": batches_per_epoch,
                "total_batch_size": 32,
            },
            "initial_baseline": {"webcam_val_acc": init_wc_acc, "kaggle_val_acc": init_kg_acc},
            "best_epoch_info": best_epoch_info,
            "history": history,
        }, f, indent=2)
    print(f"[OK] Training log saved to: {LOG_OUT}")

    return CHECKPOINT_OUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced ASL Fine-Tuning")
    parser.add_argument("--epochs-a", type=int, default=5)
    parser.add_argument("--epochs-b", type=int, default=10)
    parser.add_argument("--lr-a", type=float, default=5e-4)
    parser.add_argument("--lr-b", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_fine_tuning(
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        lr_a=args.lr_a,
        lr_b=args.lr_b,
        seed=args.seed,
    )
