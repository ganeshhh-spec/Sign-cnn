#!/usr/bin/env python3
"""
Training Pipeline for ASL Alphabet CNN

Includes:
- Training and validation loops
- Mixed precision (AMP)
- Checkpoint saving (best + latest)
- Early stopping
- Logging and metrics tracking
- Reproducibility controls
"""

import os
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, r'C:\Users\Ganesh\sign-language-cnn')
from backend.preprocessing.transforms import get_train_transforms, get_val_test_transforms, CLASSES, CLASS_TO_IDX
from backend.preprocessing.verify_dataset import ASLDataset
from backend.training.model import ASLCNN, count_parameters


# Configuration
SPLIT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\data\raw\splits")
CHECKPOINT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\models")
LOG_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\metrics")
PLOT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\plots\training")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
PATIENCE = 5
MIN_DELTA = 1e-4
NUM_WORKERS = 0  # Windows multiprocessing safety
PIN_MEMORY = True
USE_AMP = True  # Automatic Mixed Precision

# Reproducibility
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Create train, validation, and test DataLoaders."""
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    train_dataset = ASLDataset(SPLIT_DIR / "train", transform=train_transform)
    val_dataset = ASLDataset(SPLIT_DIR / "val", transform=val_transform)
    test_dataset = ASLDataset(SPLIT_DIR / "test", transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=False if num_workers == 0 else True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=False if num_workers == 0 else True
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def compute_accuracy(outputs, targets):
    """Compute top-1 accuracy."""
    _, preds = outputs.max(1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with AMP."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if USE_AMP and scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        running_total += images.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*running_correct/running_total:.2f}%")
    
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, scaler=None):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if USE_AMP and scaler is not None:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            running_total += images.size(0)
            
            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())
    
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    torch.save(state, latest_path)
    
    # Save best separately
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pth"
        torch.save(state, best_path)
        print(f"  New best model saved: {best_path}")


def load_checkpoint(path, model, optimizer=None, scaler=None, device='cuda', weights_only=False):
    """Load checkpoint. Use weights_only=False for full resume (optimizer, scaler, metadata)."""
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def plot_training_curves(history, output_dir):
    """Plot training and validation curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")


def main():
    print("=" * 70)
    print("ASL CNN Training Pipeline")
    print("=" * 70)
    
    # Setup
    set_seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders()
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    
    # Model
    model = ASLCNN(num_classes=len(CLASSES)).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Optimizer, loss, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Patience: {PATIENCE}")
    print(f"  AMP: {USE_AMP and scaler is not None}")
    print(f"  Workers: {NUM_WORKERS}")
    
    # Resume from latest checkpoint if exists
    latest_path = CHECKPOINT_DIR / "checkpoint_latest.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    if latest_path.exists():
        print(f"\nResuming from {latest_path}")
        checkpoint = load_checkpoint(latest_path, model, optimizer, scaler, device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        history = checkpoint.get('history', history)
        print(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Quick smoke test
    print("\nRunning smoke test...")
    model.train()
    test_batch = next(iter(train_loader))
    images, labels = test_batch[0].to(device), test_batch[1].to(device)
    optimizer.zero_grad(set_to_none=True)
    
    if USE_AMP and scaler:
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"  Forward pass: OK | Loss: {loss.item():.4f} | Output shape: {outputs.shape}")
    print(f"  Backward pass: OK | Params updated")
    optimizer.zero_grad()
    if device.type == 'cuda':
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    model.train()
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, scaler)
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Train: Loss={train_loss:.4f} Acc={train_acc*100:.2f}%")
        print(f"Val:   Loss={val_loss:.4f} Acc={val_acc*100:.2f}%")
        print(f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Checkpoint
        is_best = val_loss < best_val_loss - MIN_DELTA
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'history': history,
            'hyperparameters': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'max_epochs': MAX_EPOCHS,
                'patience': PATIENCE,
                'use_amp': USE_AMP,
                'num_workers': NUM_WORKERS,
                'seed': RANDOM_SEED
            },
            'class_mapping': CLASS_TO_IDX,
            'classes': CLASSES
        }, is_best, CHECKPOINT_DIR)
        
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered (no improvement for {PATIENCE} epochs)")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
    print(f"{'='*70}")
    
    # Plot curves
    plot_training_curves(history, PLOT_DIR)
    
    # Save final history
    history_path = LOG_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Load best model for final evaluation
    best_path = CHECKPOINT_DIR / "checkpoint_best.pth"
    if best_path.exists():
        print(f"\nLoading best model from {best_path}")
        checkpoint = load_checkpoint(best_path, model, device=device, weights_only=True)
        print(f"Best model epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, device, test_loader, history


if __name__ == "__main__":
    main()