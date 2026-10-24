#!/usr/bin/env python3
"""
Custom CNN for ASL Alphabet Recognition

Architecture: Compact CNN with 4 convolutional blocks
- Input: 3x128x128 (RGB)
- Output: 29 logits (A-Z, del, nothing, space)
- ~1.2M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU -> MaxPool"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 pool_size=2, pool_stride=2, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size, pool_stride)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ASLCNN(nn.Module):
    """
    Custom CNN for ASL Alphabet Classification.
    
    Architecture:
    - Block 1: 3 -> 32 channels, 128x128 -> 64x64
    - Block 2: 32 -> 64 channels, 64x64 -> 32x32
    - Block 3: 64 -> 128 channels, 32x32 -> 16x16
    - Block 4: 128 -> 256 channels, 16x16 -> 8x8
    - Global Average Pooling: 256x8x8 -> 256
    - Classifier: 256 -> 128 -> 29
    """
    
    def __init__(self, num_classes=29, dropout_conv=0.1, dropout_fc=0.3):
        super().__init__()
        
        # Convolutional backbone
        self.block1 = ConvBlock(3, 32, dropout=dropout_conv)
        self.block2 = ConvBlock(32, 64, dropout=dropout_conv)
        self.block3 = ConvBlock(64, 128, dropout=dropout_conv)
        self.block4 = ConvBlock(128, 256, dropout=dropout_conv)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional blocks
        x = self.block1(x)  # [B, 32, 64, 64]
        x = self.block2(x)  # [B, 64, 32, 32]
        x = self.block3(x)  # [B, 128, 16, 16]
        x = self.block4(x)  # [B, 256, 8, 8]
        
        # Global average pooling
        x = self.gap(x)     # [B, 256, 1, 1]
        x = x.flatten(1)    # [B, 256]
        
        # Classification head
        x = self.classifier(x)  # [B, 29]
        
        return x
    
    def get_features(self, x):
        """Extract features before classification head (for analysis)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return x.flatten(1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(3, 128, 128)):
    """Print model architecture summary."""
    print("=" * 70)
    print("ASL CNN Model Summary")
    print("=" * 70)
    
    # Calculate output shapes
    x = torch.randn(1, *input_size)
    print(f"Input shape: {tuple(x.shape)}")
    
    x = model.block1(x)
    print(f"After Block1 (Conv32+Pool): {tuple(x.shape)}")
    
    x = model.block2(x)
    print(f"After Block2 (Conv64+Pool): {tuple(x.shape)}")
    
    x = model.block3(x)
    print(f"After Block3 (Conv128+Pool): {tuple(x.shape)}")
    
    x = model.block4(x)
    print(f"After Block4 (Conv256+Pool): {tuple(x.shape)}")
    
    x = model.gap(x)
    print(f"After GlobalAvgPool: {tuple(x.shape)}")
    
    x = x.flatten(1)
    x = model.classifier(x)
    print(f"Output (logits): {tuple(x.shape)}")
    
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024**2:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    model = ASLCNN(num_classes=29)
    print_model_summary(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    print(f"\nTest forward pass: input {x.shape} -> output {out.shape}")
    assert out.shape == (2, 29), "Output shape mismatch!"
    print("Forward pass successful")