import cv2
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(".").resolve()

# 1. Measure Kaggle images (sample of 100 images across classes)
kg_samples = list((PROJECT_ROOT / "data" / "raw" / "splits" / "train").glob("*/*.jpg"))[:200]
wc_samples = list((PROJECT_ROOT / "data" / "webcam_captures").glob("*/*.jpg"))
rec_crops = list(Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test").glob("*_crop.jpg"))

def analyze_image_batch(paths, name):
    widths = []
    heights = []
    mean_intensities = []
    std_intensities = []
    skin_fractions = []
    
    for p in paths:
        img = cv2.imread(str(p))
        if img is None: continue
        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        mean_intensities.append(img.mean())
        std_intensities.append(img.std())
        
        # Simple skin color detector in YCrCb space:
        # Y in [0, 255], Cr in [133, 173], Cb in [77, 127]
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_fractions.append(np.mean(mask > 0))
        
    print(f"\n--- {name} (N={len(paths)}) ---")
    print(f"Dimensions (W x H): min={min(widths)}x{min(heights)}, max={max(widths)}x{max(heights)}, mean={np.mean(widths):.1f}x{np.mean(heights):.1f}")
    print(f"Mean Pixel Intensity: {np.mean(mean_intensities):.2f} (std={np.mean(std_intensities):.2f})")
    print(f"Skin Pixel Fraction (Hand Coverage): {np.mean(skin_fractions)*100:.1f}% (min={min(skin_fractions)*100:.1f}%, max={max(skin_fractions)*100:.1f}%)")

analyze_image_batch(kg_samples, "Kaggle ASL Training Split")
analyze_image_batch(wc_samples, "Real Webcam Captures (56 images)")
analyze_image_batch(rec_crops, "Screen Recording Physical Attempts (21 crops)")
