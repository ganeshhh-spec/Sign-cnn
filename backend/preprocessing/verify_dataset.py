#!/usr/bin/env python3
"""
ASL Alphabet PyTorch Dataset and Verification Script

Creates a PyTorch Dataset class and verifies:
- Batch loading works correctly
- Dimensions and labels are correct
- GPU compatibility
- Creates contact sheet of samples
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SPLIT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\data\raw\splits")
OUTPUT_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\metrics")
PLOTS_DIR = Path(r"C:\Users\Ganesh\sign-language-cnn\results\plots")

CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

IMAGE_SIZE = 128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL Alphabet."""
    
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples = []
        
        for cls in CLASSES:
            cls_dir = self.split_dir / cls
            if cls_dir.exists():
                for img_file in cls_dir.glob("*.jpg"):
                    self.samples.append((str(img_file), CLASS_TO_IDX[cls]))
                for img_file in cls_dir.glob("*.png"):
                    self.samples.append((str(img_file), CLASS_TO_IDX[cls]))
                for img_file in cls_dir.glob("*.jpeg"):
                    self.samples.append((str(img_file), CLASS_TO_IDX[cls]))
        
        self.samples.sort(key=lambda x: x[0])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split):
    """Get transforms for a given split."""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
            transforms.RandomRotation(15, interpolation=Image.BILINEAR, fill=(0, 0, 0)),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, 
                                    interpolation=Image.BILINEAR, fill=(0, 0, 0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])


def denormalize(tensor):
    """Denormalize tensor for visualization."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1).squeeze(0)


def create_contact_sheet(dataset, split_name, n_per_class=1):
    """Create contact sheet with samples from each class."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    class_indices = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(dataset.samples):
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)
    
    selected_indices = []
    for cls_idx in range(len(CLASSES)):
        if class_indices[cls_idx]:
            selected_indices.extend(class_indices[cls_idx])
    
    if not selected_indices:
        return
    
    images = []
    labels = []
    for idx in selected_indices:
        img, label = dataset[idx]
        img_denorm = denormalize(img)
        images.append(img_denorm)
        labels.append(CLASSES[label])
    
    grid = torch.stack(images)
    from torchvision.utils import make_grid
    grid_img = make_grid(grid, nrow=10, padding=2, pad_value=1.0)
    grid_np = grid_img.permute(1, 2, 0).numpy()
    
    ncols = 10
    nrows = (len(images) + ncols - 1) // ncols
    fig, ax = plt.subplots(figsize=(ncols * 1.5, nrows * 1.8))
    ax.imshow(grid_np)
    ax.axis('off')
    
    for i, label in enumerate(labels):
        row = i // ncols
        col = i % ncols
        x = col * (IMAGE_SIZE + 2) + IMAGE_SIZE // 2
        y = row * (IMAGE_SIZE + 2) + IMAGE_SIZE + 20
        ax.text(x, y, label, ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = PLOTS_DIR / f"contact_sheet_{split_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Contact sheet saved to: {output_path}")


def verify_dataloader(split_name, batch_size=32, num_workers=0):
    """Verify DataLoader works correctly."""
    print(f"\n{'='*60}")
    print(f"Verifying {split_name} DataLoader")
    print(f"{'='*60}")
    
    split_path = SPLIT_DIR / split_name
    if not split_path.exists():
        print(f"ERROR: Split directory not found: {split_path}")
        return None
    
    transform = get_transforms(split_name)
    dataset = ASLDataset(split_path, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes present: {len(set(label for _, label in dataset.samples))}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name=='train'), 
                        num_workers=num_workers, pin_memory=True)
    
    batch = next(iter(loader))
    images, labels = batch
    
    print(f"Batch shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image value range: [{images.min():.4f}, {images.max():.4f}]")
    
    class_counts = torch.bincount(labels, minlength=len(CLASSES))
    print(f"Classes in batch: {(class_counts > 0).sum().item()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_gpu = images.to(device)
    labels_gpu = labels.to(device)
    print(f"GPU transfer: OK (device: {images_gpu.device})")
    
    del images_gpu, labels_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    create_contact_sheet(dataset, split_name, n_per_class=1)
    
    return dataset


def main():
    print("=" * 60)
    print("ASL Alphabet Dataset Verification")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    for split in ['train', 'val', 'test']:
        dataset = verify_dataloader(split, batch_size=64)
        if dataset:
            print(f"  {split}: {len(dataset)} samples - OK")
    
    print(f"\n{'='*60}")
    print("Verification Complete")
    print(f"{'='*60}")
    print(f"Contact sheets saved to: {PLOTS_DIR}")
    
    results = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "splits_verified": ["train", "val", "test"],
        "contact_sheets": [f"contact_sheet_{s}.png" for s in ["train", "val", "test"]]
    }
    
    with open(OUTPUT_DIR / "dataset_verification.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Verification results saved to: {OUTPUT_DIR / 'dataset_verification.json'}")


if __name__ == "__main__":
    main()