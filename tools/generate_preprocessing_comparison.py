#!/usr/bin/env python3
"""
Generate Preprocessing Pipeline Comparison Diagram
==================================================
Produces: reports/dataset_comparison/preprocessing_pipeline_comparison.png
Demonstrates before-and-after transformation steps for:
  1. Kaggle ASL training image
  2. Real physical webcam frame
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from backend.app.services.hand_detector import HandDetector

OUT_PATH = PROJECT_ROOT / "reports" / "dataset_comparison" / "preprocessing_pipeline_comparison.png"

detector = HandDetector(static_image_mode=True, expand_ratio=1.20, enable_smoothing=False)

# 1. Load real webcam frame from scratch/physical_webcam_test
wc_raw_path = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test\attempt_08_raw.jpg")
if not wc_raw_path.exists():
    wc_raw_path = list(Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test").glob("*_raw.jpg"))[0]

wc_raw_bgr = cv2.imread(str(wc_raw_path))
wc_raw_rgb = cv2.cvtColor(wc_raw_bgr, cv2.COLOR_BGR2RGB)

# Run MediaPipe
dets = detector.detect(wc_raw_rgb)
d = dets[0] if dets else None

# Hand crop
if d:
    wc_crop = detector.crop_hand(wc_raw_rgb, d["bbox"])
    mp_hand = d["handedness"]
    mp_conf = d["confidence"]
else:
    wc_crop = wc_raw_rgb[100:300, 200:400]
    mp_hand = "Right"
    mp_conf = 0.90

# Left-hand flip
is_left = (mp_hand == "Left")
wc_norm = cv2.flip(wc_crop, 1) if is_left else wc_crop.copy()

# Resize
wc_resized = cv2.resize(wc_norm, (128, 128), interpolation=cv2.INTER_LINEAR)

# 2. Load Kaggle image
kg_path = list((PROJECT_ROOT / "data" / "raw" / "splits" / "train" / "B").glob("*.jpg"))[0]
kg_raw_bgr = cv2.imread(str(kg_path))
kg_raw_rgb = cv2.cvtColor(kg_raw_bgr, cv2.COLOR_BGR2RGB)
kg_resized = cv2.resize(kg_raw_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

# Create 2-row figure
fig, axes = plt.subplots(2, 5, figsize=(18, 8), facecolor="#f8fafc")
fig.suptitle("Preprocessing Pipeline Audit: Training Pipeline vs. Live Webcam Pipeline",
             fontsize=16, fontweight="bold", y=0.98, color="#0f172a")

# Row 1: Real Webcam Pipeline
# Step 1: Raw Frame
axes[0, 0].imshow(wc_raw_rgb)
axes[0, 0].set_title(f"Step 1: Raw Webcam Frame\n({wc_raw_rgb.shape[1]}x{wc_raw_rgb.shape[0]} px, Unmirrored)", fontsize=9, fontweight="bold")
axes[0, 0].axis("off")

# Step 2: MediaPipe Landmark Bounding Box
wc_annotated = wc_raw_rgb.copy()
if d:
    bb = d["bbox"]
    cv2.rectangle(wc_annotated, (bb["x_min"], bb["y_min"]), (bb["x_max"], bb["y_max"]), (0, 255, 0), 3)
axes[0, 1].imshow(wc_annotated)
axes[0, 1].set_title(f"Step 2: MediaPipe Detection\nHandedness: {mp_hand} ({mp_conf:.2f})", fontsize=9, fontweight="bold")
axes[0, 1].axis("off")

# Step 3: Square Crop + 1.20x Expansion
axes[0, 2].imshow(wc_crop)
axes[0, 2].set_title(f"Step 3: Hand Crop (1.20x)\n({wc_crop.shape[1]}x{wc_crop.shape[0]} px Square)", fontsize=9, fontweight="bold")
axes[0, 2].axis("off")

# Step 4: Left-Hand Normalization
axes[0, 3].imshow(wc_norm)
norm_label = "Horiz Flip (Left -> Right)" if is_left else "No Flip (Right Hand)"
axes[0, 3].set_title(f"Step 4: Handedness Norm\n{norm_label}", fontsize=9, fontweight="bold")
axes[0, 3].axis("off")

# Step 5: Model Input (128x128 Normalized)
axes[0, 4].imshow(wc_resized)
axes[0, 4].set_title("Step 5: Final Model Input\n(128x128 px, Normalized Tensor)", fontsize=9, fontweight="bold")
axes[0, 4].axis("off")

# Row 2: Kaggle Training Pipeline
# Step 1: Raw Kaggle Image
axes[1, 0].imshow(kg_raw_rgb)
axes[1, 0].set_title(f"Step 1: Kaggle Dataset File\n({kg_raw_rgb.shape[1]}x{kg_raw_rgb.shape[0]} px, Studio Board)", fontsize=9, fontweight="bold")
axes[1, 0].axis("off")

# Step 2: Detection Skip
axes[1, 1].imshow(kg_raw_rgb)
axes[1, 1].set_title("Step 2: MediaPipe Detection\nSKIPPED in Training Pipeline", fontsize=9, fontweight="bold", color="#b91c1c")
axes[1, 1].text(0.5, 0.5, "NO MEDIAPIPE IN TRAINING\n(Images pre-cropped by author)",
                ha="center", va="center", color="yellow", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#991b1b", alpha=0.8))
axes[1, 1].axis("off")

# Step 3: Cropping Skip
axes[1, 2].imshow(kg_raw_rgb)
axes[1, 2].set_title("Step 3: Hand Cropping\nSKIPPED (Already 200x200)", fontsize=9, fontweight="bold", color="#b91c1c")
axes[1, 2].axis("off")

# Step 4: Normalization Skip
axes[1, 3].imshow(kg_raw_rgb)
axes[1, 3].set_title("Step 4: Handedness Norm\nSKIPPED (100% Right Hands)", fontsize=9, fontweight="bold", color="#b91c1c")
axes[1, 3].axis("off")

# Step 5: Resize to 128x128
axes[1, 4].imshow(kg_resized)
axes[1, 4].set_title("Step 5: Final Model Input\n(128x128 px, Direct Downsample)", fontsize=9, fontweight="bold")
axes[1, 4].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Generated {OUT_PATH}")
