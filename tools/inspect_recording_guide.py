import cv2
import numpy as np
from pathlib import Path

frames_dir = Path(r"C:\Users\Ganesh\.gemini\antigravity\brain\b56a98ce-c54a-49b3-b97b-9f38cd01ccbc\scratch\video_frames")

# Inspect frames at 10s, 20s, 30s
for sec in [6, 12, 18, 24, 30, 36]:
    fpath = frames_dir / f"frame_{sec:02d}s.jpg"
    if not fpath.exists(): continue
    img = cv2.imread(str(fpath))
    h, w, c = img.shape
    
    # In 960x1080 screen recordings, camera view is roughly y in [80, 520], x in [40, 920]
    cam_crop = img[80:520, 40:920]
    
    # Let's detect green pixels (the guide box border or active bounding box)
    # Green in HSV: H in [35, 85], S > 100, V > 100
    hsv = cv2.cvtColor(cam_crop, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    # Yellow/amber (guide box inactive border): H in [15, 35]
    mask_amber = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    # Cyan/blue (hand bbox border): H in [85, 105] or [100, 130]
    mask_cyan = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([125, 255, 255]))
    
    cnts_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_a, _ = cv2.findContours(mask_amber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_c, _ = cv2.findContours(mask_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Time {sec:02d}s (Cam size {cam_crop.shape[1]}x{cam_crop.shape[0]}):")
    for cnt in cnts_g:
        if cv2.contourArea(cnt) > 200:
            x, y, bw, bh = cv2.boundingRect(cnt)
            print(f"   GREEN box at x={x} ({x/cam_crop.shape[1]*100:.1f}%), y={y}, w={bw}, h={bh}")
    for cnt in cnts_a:
        if cv2.contourArea(cnt) > 200:
            x, y, bw, bh = cv2.boundingRect(cnt)
            print(f"   AMBER box at x={x} ({x/cam_crop.shape[1]*100:.1f}%), y={y}, w={bw}, h={bh}")
    for cnt in cnts_c:
        if cv2.contourArea(cnt) > 200:
            x, y, bw, bh = cv2.boundingRect(cnt)
            print(f"   CYAN box at x={x} ({x/cam_crop.shape[1]*100:.1f}%), y={y}, w={bw}, h={bh}")
