#!/usr/bin/env python3
"""
ASL Alphabet Preprocessing Transforms

Defines preprocessing and augmentation pipelines for training and inference.
Compatible with PyTorch DataLoader and torchvision transforms.
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import math


# Configuration
IMAGE_SIZE = 128
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Classes (must match dataset)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


def get_train_transforms(image_size=IMAGE_SIZE, augment_strength='moderate'):
    """
    Get training transforms with augmentation.
    
    Args:
        image_size: Target image size (square)
        augment_strength: 'light', 'moderate', or 'strong'
    
    Returns:
        Composed transforms
    """
    if augment_strength == 'light':
        rotation = 10
        translate = 0.05
        scale = (0.95, 1.05)
        brightness = 0.1
        contrast = 0.1
        saturation = 0.1
    elif augment_strength == 'moderate':
        rotation = 15
        translate = 0.1
        scale = (0.9, 1.1)
        brightness = 0.2
        contrast = 0.2
        saturation = 0.2
    elif augment_strength == 'strong':
        rotation = 25
        translate = 0.15
        scale = (0.8, 1.2)
        brightness = 0.3
        contrast = 0.3
        saturation = 0.3
    else:
        raise ValueError(f"Unknown augment_strength: {augment_strength}")
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.RandomRotation(
            degrees=rotation,
            interpolation=Image.BILINEAR,
            fill=(0, 0, 0)
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(translate, translate),
            scale=scale,
            shear=5,
            interpolation=Image.BILINEAR,
            fill=(0, 0, 0)
        ),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=0.05
        ),
        transforms.RandomHorizontalFlip(p=0.0),  # Disabled - flipping changes sign meaning
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_val_test_transforms(image_size=IMAGE_SIZE):
    """
    Get validation/test transforms (deterministic, no augmentation).
    
    Args:
        image_size: Target image size (square)
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_inference_transforms(image_size=IMAGE_SIZE):
    """
    Get transforms for webcam inference (same as val/test).
    
    Args:
        image_size: Target image size (square)
    
    Returns:
        Composed transforms
    """
    return get_val_test_transforms(image_size)


class RandomResizedCropWithCoords:
    """Random resized crop that returns crop coordinates for debugging."""
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, Image.BILINEAR)


def denormalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    tensor = tensor * std + mean
    return tensor.clamp(0, 1).squeeze(0)


def create_contact_sheet(dataset, indices, transforms, output_path, ncols=10, image_size=IMAGE_SIZE):
    """
    Create a contact sheet of sample images from dataset.
    
    Args:
        dataset: PyTorch dataset with __getitem__ returning (image, label)
        indices: List of indices to visualize
        transforms: Transform to apply (should include ToTensor but not Normalize for display)
        output_path: Path to save contact sheet
        ncols: Number of columns in grid
        image_size: Size of each image in grid
    """
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        if isinstance(img, torch.Tensor):
            if img.max() > 1.0:
                img = img / 255.0
        images.append(img)
        labels.append(IDX_TO_CLASS.get(label, str(label)))
    
    if images:
        grid = make_grid(images, nrow=ncols, padding=2, pad_value=1.0)
        grid_np = grid.permute(1, 2, 0).numpy()
        
        fig, ax = plt.subplots(figsize=(ncols * 1.5, (len(images) // ncols + 1) * 1.5))
        ax.imshow(grid_np)
        ax.axis('off')
        
        for i, label in enumerate(labels):
            row = i // ncols
            col = i % ncols
            x = col * (image_size + 2) + image_size // 2
            y = row * (image_size + 2) + image_size + 15
            ax.text(x, y, label, ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Contact sheet saved to: {output_path}")


def print_transform_config():
    """Print the current transform configuration."""
    print("=" * 60)
    print("Preprocessing Configuration")
    print("=" * 60)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Input format: RGB")
    print(f"Normalization mean: {NORMALIZE_MEAN}")
    print(f"Normalization std: {NORMALIZE_STD}")
    print(f"\nClass mapping ({len(CLASSES)} classes):")
    for cls, idx in CLASS_TO_IDX.items():
        print(f"  {idx:2d}: {cls}")
    print(f"\nTrain augmentation (moderate):")
    print(f"  RandomRotation: ±15°")
    print(f"  RandomAffine: translate±10%, scale 0.9-1.1, shear±5°")
    print(f"  ColorJitter: brightness±0.2, contrast±0.2, saturation±0.2, hue±0.05")
    print(f"  HorizontalFlip: DISABLED (changes sign meaning)")
    print(f"\nVal/Test transforms:")
    print(f"  Resize to {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  ToTensor + Normalize")
    print(f"  No random augmentation")


if __name__ == "__main__":
    print_transform_config()
    
    train_tf = get_train_transforms()
    val_tf = get_val_test_transforms()
    infer_tf = get_inference_transforms()
    
    print(f"\nTrain transforms: {train_tf}")
    print(f"\nVal/Test transforms: {val_tf}")