import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from backend.app.services.hand_detector import HandDetector

detector = HandDetector(static_image_mode=True, enable_smoothing=False)

sample_classes = ['A', 'B', 'C', 'H', 'L', 'Y']
for c in sample_classes:
    img_paths = list((Path('data/raw/splits/train') / c).glob('*.jpg'))
    if not img_paths: continue
    img = cv2.imread(str(img_paths[0]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Direct 200x200
    d1 = detector.detect(rgb)
    
    # 2. Padded into 480x640 frame (simulating a camera view)
    padded = np.zeros((480, 640, 3), dtype=np.uint8)
    # place in center
    padded[140:340, 220:420] = rgb
    d2 = detector.detect(padded)
    
    # 3. Resized to 400x400
    resized = cv2.resize(rgb, (400, 400))
    d3 = detector.detect(resized)
    
    print(f"Class {c}: Direct 200x200: {len(d1)} hands | Padded in 480x640: {len(d2)} hands | Resized 400x400: {len(d3)} hands")
    if d2:
        print(f"   -> Padded detected: Hand={d2[0]['handedness']}, conf={d2[0]['confidence']:.2f}, bbox={d2[0]['bbox']}")
