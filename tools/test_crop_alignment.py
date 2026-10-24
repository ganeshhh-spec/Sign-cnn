import cv2
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from backend.app.services.hand_detector import HandDetector
from backend.app.services.model_loader import initialize_model, get_model_loader
from backend.preprocessing.transforms import CLASSES

initialize_model()
loader = get_model_loader()
detector = HandDetector(static_image_mode=True, expand_ratio=1.20, enable_smoothing=False)

# Let's test on the 21 physical webcam test attempts:
raw_dir = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\physical_webcam_test")
attempt_files = sorted(list(raw_dir.glob("attempt_*_raw.jpg")))

# Ground truth mapping from test_physical_webcam_pipeline.py
ground_truth = {
    0: ("A", "Right"), 1: ("A", "Right"), 2: ("A", "Right"), 3: ("A", "Right"), 4: ("A", "Right"),
    5: ("B", "Right"), 6: ("B", "Right"), 7: ("B", "Left"),  8: ("B", "Left"),
    9: ("C", "Left"),  10: ("C", "Left"), 11: ("C", "Left"),
    12: ("L", "Left"), 13: ("L", "Left"), 14: ("L", "Left"), 15: ("L", "Left"),
    16: ("H", "Left"), 17: ("H", "Left"), 18: ("H", "Left"), 19: ("H", "Left"),
    20: ("M", "Right")
}

print(f"Testing {len(ground_truth)} physical attempts with varied crop expansion & vertical shifts...")

for exp_ratio in [1.10, 1.20, 1.30, 1.40]:
    detector.expand_ratio = exp_ratio
    correct = 0
    total = 0
    for idx, (intended, hand) in ground_truth.items():
        raw_f = raw_dir / f"attempt_{idx:02d}_raw.jpg"
        if not raw_f.exists(): continue
        img = cv2.imread(str(raw_f))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector.detect(rgb)
        if not dets: continue
        d = dets[0]
        crop = detector.crop_hand(rgb, d["bbox"])
        is_left = (d["handedness"] == "Left")
        t, _ = detector.preprocess_for_inference(crop, flip_horizontal=is_left)
        p, _ = loader.predict(t)
        pred = p[0][0]["class_label"]
        conf = p[0][0]["confidence"]
        if pred == intended:
            correct += 1
        total += 1
    print(f"Expansion {exp_ratio:.2f}: Detected {total}/{len(ground_truth)}, Correct = {correct}/{total} ({correct/total*100:.1f}%)")
