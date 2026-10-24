import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from backend.app.services.hand_detector import HandDetector

detector = HandDetector(static_image_mode=True, enable_smoothing=False)

sample_classes = ['A', 'B', 'C', 'H', 'L', 'Y', 'del', 'nothing', 'space']
for c in sample_classes:
    img_paths = list((Path('data/raw/splits/train') / c).glob('*.jpg'))
    if not img_paths:
        continue
    first_img = img_paths[0]
    img = cv2.imread(str(first_img))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector.detect(rgb)
    if dets:
        d = dets[0]
        bb = d['bbox']
        w = bb['x_max'] - bb['x_min']
        h = bb['y_max'] - bb['y_min']
        area_pct = (w * h) / (img.shape[0] * img.shape[1]) * 100
        print(f"Kaggle {c:7s}: Hand detected! Handedness={d['handedness']} ({d['confidence']:.2f}) | BBox size: {w}x{h} in {img.shape[1]}x{img.shape[0]} ({area_pct:.1f}% area)")
    else:
        print(f"Kaggle {c:7s}: NO HAND DETECTED")
