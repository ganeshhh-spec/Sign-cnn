#!/usr/bin/env python3
"""
Evaluation Script for ASL CNN

Evaluates the best model on the held-out test split:
- Test accuracy
- Per-class precision, recall, F1
- 29x29 confusion matrix
- Misclassified examples for error analysis
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.preprocessing.transforms import get_val_test_transforms, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from backend.preprocessing.verify_dataset import ASLDataset
from backend.training.model import ASLCNN, count_parameters


SPLIT_DIR = PROJECT_ROOT / "data" / "raw" / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "models"
LOG_DIR = PROJECT_ROOT / "results" / "metrics"
PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "training"

BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = True


def compute_per_class_metrics(conf_matrix):
    """Compute precision, recall, F1 for each class from confusion matrix."""
    num_classes = conf_matrix.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[CLASSES[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support)
        }
    
    return metrics


def plot_confusion_matrix(conf_matrix, output_path, class_names=None):
    """Plot and save confusion matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if class_names is None:
        class_names = CLASSES
    
    # Normalize for visualization
    conf_norm = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
    conf_norm = np.nan_to_num(conf_norm)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(conf_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def save_misclassified_examples(model, test_loader, device, output_dir, max_per_class=5):
    """Save misclassified images for error analysis."""
    import shutil
    from PIL import Image
    
    model.eval()
    misclassified = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            preds = outputs.argmax(1)
            
            # Find misclassified
            mask = preds != labels
            if mask.any():
                for idx in mask.nonzero().squeeze(1):
                    true_label = labels[idx].item()
                    pred_label = preds[idx].item()
                    true_class = CLASSES[true_label]
                    pred_class = CLASSES[pred_label]
                    
                    if len(misclassified[true_class]) < max_per_class:
                        # Save image
                        img_tensor = images[idx].cpu()
                        # Denormalize
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = img_tensor.clamp(0, 1)
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        
                        misclassified[true_class].append({
                            'true': true_class,
                            'pred': pred_class,
                            'image': img_np
                        })
    
    # Save misclassified images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    for true_class, items in misclassified.items():
        class_dir = output_dir / f"true_{true_class}"
        class_dir.mkdir(exist_ok=True)
        summary[true_class] = []
        
        for i, item in enumerate(items):
            img = Image.fromarray(item['image'])
            filename = f"pred_{item['pred']}_idx_{i}.png"
            img.save(class_dir / filename)
            summary[true_class].append({
                'true': item['true'],
                'pred': item['pred'],
                'file': filename
            })
    
    # Save summary JSON
    with open(output_dir / "misclassified_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    total_misclassified = sum(len(v) for v in misclassified.values())
    print(f"Misclassified examples saved: {total_misclassified} images")
    print(f"Saved to: {output_dir}")
    
    return summary


def evaluate_model(model, test_loader, device):
    """Run full evaluation on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            running_total += images.size(0)
            
            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())
    
    test_loss = running_loss / running_total
    test_acc = running_correct / running_total
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Confusion matrix
    num_classes = len(CLASSES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels.numpy(), all_preds.numpy()):
        conf_matrix[t, p] += 1
    
    return test_loss, test_acc, conf_matrix, all_preds.numpy(), all_labels.numpy()


def main():
    print("=" * 70)
    print("ASL CNN Model Evaluation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load best checkpoint
    best_path = CHECKPOINT_DIR / "checkpoint_best.pth"
    if not best_path.exists():
        print(f"ERROR: Best checkpoint not found at {best_path}")
        return
    
    # Use weights_only=True since we only need model weights for inference
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    # Create model and load weights
    model = ASLCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Load class mapping from checkpoint (should match)
    saved_classes = checkpoint.get('classes', CLASSES)
    saved_mapping = checkpoint.get('class_mapping', CLASS_TO_IDX)
    
    # Verify class mapping consistency
    if saved_classes != CLASSES or saved_mapping != CLASS_TO_IDX:
        print("WARNING: Class mapping mismatch! Using checkpoint mapping.")
        print(f"  Checkpoint classes: {saved_classes}")
        print(f"  Current classes: {CLASSES}")
    
    # Test data
    val_transform = get_val_test_transforms()
    test_dataset = ASLDataset(SPLIT_DIR / "test", transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    print(f"\nTest set: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Run evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, conf_matrix, preds, labels = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Per-class metrics
    metrics = compute_per_class_metrics(conf_matrix)
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 50)
    
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    
    for cls in CLASSES:
        m = metrics[cls]
        print(f"{cls:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>8d}")
        macro_precision += m['precision']
        macro_recall += m['recall']
        macro_f1 += m['f1']
    
    num_classes = len(CLASSES)
    print("-" * 50)
    print(f"{'MACRO':<10} {macro_precision/num_classes:>10.4f} {macro_recall/num_classes:>10.4f} {macro_f1/num_classes:>10.4f}")
    
    # Overall accuracy from confusion matrix
    total_correct = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()
    print(f"\nOverall Accuracy (from CM): {total_correct}/{total_samples} = {total_correct/total_samples*100:.2f}%")
    
    # Find frequently confused pairs
    print(f"\nMost Confused Pairs (off-diagonal > 5):")
    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 5:
                confused.append((conf_matrix[i, j], CLASSES[i], CLASSES[j]))
    
    confused.sort(reverse=True)
    for count, true_cls, pred_cls in confused[:15]:
        print(f"  {true_cls} -> {pred_cls}: {count}")
    
    # Save confusion matrix plot
    plot_confusion_matrix(conf_matrix, PLOT_DIR / "confusion_matrix.png")
    
    # Save misclassified examples
    misclassified_dir = PLOT_DIR / "misclassified"
    save_misclassified_examples(model, test_loader, device, misclassified_dir)
    
    # Save all results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'per_class_metrics': metrics,
        'macro_precision': macro_precision / num_classes,
        'macro_recall': macro_recall / num_classes,
        'macro_f1': macro_f1 / num_classes,
        'confusion_matrix': conf_matrix.tolist(),
        'confused_pairs': [{'count': int(c), 'true': t, 'pred': p} for c, t, p in confused],
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_val_loss': checkpoint.get('best_val_loss'),
        'model_parameters': count_parameters(model)
    }
    
    results_path = LOG_DIR / "test_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_path}")
    
    # Save class mapping for deployment
    deployment_info = {
        'class_to_idx': CLASS_TO_IDX,
        'idx_to_class': IDX_TO_CLASS,
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'input_size': [3, 128, 128],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'checkpoint_path': str(best_path),
        'model_architecture': 'ASLCNN'
    }
    
    deploy_path = CHECKPOINT_DIR / "deployment_info.json"
    with open(deploy_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    print(f"Deployment info saved to: {deploy_path}")
    
    return results


if __name__ == "__main__":
    main()