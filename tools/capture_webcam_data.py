#!/usr/bin/env python3
"""
Webcam Data Capture Tool for ASL Fine-tuning
=============================================
Captures labelled hand crops through the EXACT production pipeline:
  Camera → MediaPipe HandDetector → square crop → handedness normalization

Controls
--------
  A-Z           → label and save crop for that letter (including Q)
  BACKSPACE / D → delete / nothing class
  SPACE         → space class
  ESC           → quit

Output
------
  data/webcam_captures/<LETTER>/<timestamp>.jpg   — raw crop (pre-resize)
  data/webcam_captures/<LETTER>/<timestamp>.json  — metadata
  data/webcam_captures/session_log.jsonl          — per-capture JSONL log

Signs to capture for demo: A, B, C, H, E, L, P
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.app.services.hand_detector import HandDetector
    HAND_DETECTOR_OK = True
except Exception as e:
    print(f"[WARN] Could not import HandDetector: {e}")
    HAND_DETECTOR_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "data" / "webcam_captures"
DEMO_SIGNS = list("ABCHELP")          # Priority letters
SIGNING_ZONE_COLOR = (0, 220, 0)      # Green
GUIDE_COLOR = (0, 160, 255)           # Orange
CAPTURE_COOLDOWN = 0.4               # Seconds between auto-duplicate captures
EXPAND_RATIO = 1.20                  # Must match hand_detector.py default

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Signing zone: right third of frame, vertically centred
ZONE_X1 = int(FRAME_WIDTH  * 0.30)
ZONE_X2 = int(FRAME_WIDTH  * 0.98)
ZONE_Y1 = int(FRAME_HEIGHT * 0.05)
ZONE_Y2 = int(FRAME_HEIGHT * 0.95)

# ---------------------------------------------------------------------------
# Keyboard mapping and configuration
# ---------------------------------------------------------------------------
QUIT_KEY_CODE = 27  # ESC (key code 27)

KEY_MAP = {}
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    KEY_MAP[ord(c)] = c
for c in "abcdefghijklmnopqrstuvwxyz":
    KEY_MAP[ord(c)] = c.upper()
KEY_MAP[8] = "del"     # BACKSPACE
KEY_MAP[127] = "del"   # DEL
KEY_MAP[32] = "space"  # SPACE

INSTRUCTION_TEXT = "Press A–Z to save a sample for that class. Press ESC to quit."
WINDOW_TITLE = "ASL Capture — Press A-Z to save a sample for that class. Press ESC to quit."


def is_quit_key(key: int) -> bool:
    """
    Check whether key code is the quit key.
    Exits ONLY on ESC (key code 27). Alphabet keys like 'Q' are reserved for class labels.
    """
    return (key & 0xFF) == QUIT_KEY_CODE


def get_label_for_key(key: int) -> Optional[str]:
    """
    Map keyboard key code to class label.
    Normalizes keyboard input with chr(key).upper() so lowercase and uppercase
    key presses work consistently.
    Reserves all alphabet keys A-Z exclusively for capturing corresponding class labels.
    """
    key_code = key & 0xFF

    # Special non-alphabetic keys
    if key_code in (8, 127):
        return "del"
    if key_code == 32:
        return "space"

    # Normalize alphabetic keys with chr(key).upper()
    try:
        char = chr(key_code).upper()
        if "A" <= char <= "Z":
            return char
    except (ValueError, OverflowError):
        pass

    return KEY_MAP.get(key_code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_crop(crop_bgr, label, handedness, flip_applied, bbox, frame_shape, session_log_path, output_dir=None):
    """Save a labelled crop and its metadata."""
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    label_dir = target_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = label_dir / f"{ts}.jpg"
    meta_path = label_dir / f"{ts}.json"

    cv2.imwrite(str(img_path), crop_bgr)

    meta = {
        "timestamp":        ts,
        "label":            label,
        "handedness":       handedness,
        "flip_applied":     flip_applied,
        "bbox":             bbox,
        "frame_shape":      list(frame_shape[:2]),
        "crop_shape":       list(crop_bgr.shape[:2]),
        "image_path":       str(img_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in img_path.parents else img_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Append to session log
    session_log = Path(session_log_path)
    session_log.parent.mkdir(parents=True, exist_ok=True)
    with open(session_log, "a") as f:
        f.write(json.dumps(meta) + "\n")

    return img_path


def draw_signing_guide(frame):
    """Draw transparent signing zone overlay and instructions."""
    overlay = frame.copy()

    # Semi-transparent zone rectangle
    cv2.rectangle(overlay, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2),
                  SIGNING_ZONE_COLOR, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Zone border
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2),
                  SIGNING_ZONE_COLOR, 2)

    # Label
    cv2.putText(frame, "SIGNING ZONE", (ZONE_X1 + 4, ZONE_Y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, SIGNING_ZONE_COLOR, 1)


def draw_hud(frame, counts, last_label, hand_detected, handedness, flip_applied):
    """Draw HUD showing counts, last capture, and hand status."""
    # Background panel
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)

    # Priority sign counts
    x = 6
    for letter in DEMO_SIGNS:
        cnt = counts.get(letter, 0)
        color = (0, 220, 0) if cnt >= 20 else ((0, 180, 255) if cnt >= 10 else (180, 180, 180))
        cv2.putText(frame, f"{letter}:{cnt}", (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        x += 56

    # Last label
    cv2.putText(frame, f"Last: {last_label or '--'}", (6, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Hand status
    if hand_detected:
        norm_txt = f"FLIP ({handedness})" if flip_applied else f"OK ({handedness})"
        cv2.putText(frame, f"Hand: {norm_txt}", (6, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)
    else:
        cv2.putText(frame, "Hand: NOT DETECTED", (6, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1)

    # Key hint: instructions in the capture window
    hint_text = INSTRUCTION_TEXT.replace("–", "-")
    cv2.putText(frame, hint_text,
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)


def count_existing():
    """Count already-captured samples per class."""
    counts = {}
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                counts[d.name] = len(list(d.glob("*.jpg")))
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_log = OUTPUT_DIR / "session_log.jsonl"

    print("=" * 60)
    print("ASL Webcam Data Capture Tool")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Priority signs: {DEMO_SIGNS}")
    print()
    print("INSTRUCTIONS:")
    print("  1. Keep your hand INSIDE the green signing zone.")
    print("  2. Hold the sign UPRIGHT (fingers pointing up for most signs).")
    print("  3. Keep face OUTSIDE the crop box shown in the preview.")
    print(f"  4. {INSTRUCTION_TEXT}")
    print("  5. Aim for 20+ samples per letter with slight position variation.")
    print()

    # Initialise hand detector
    detector = None
    if HAND_DETECTOR_OK:
        try:
            detector = HandDetector(
                static_image_mode=True,
                expand_ratio=EXPAND_RATIO,
                min_detection_confidence=0.6
            )
            print("[OK] HandDetector initialised")
        except Exception as e:
            print(f"[WARN] HandDetector failed: {e}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    counts    = count_existing()
    last_label = None
    last_save  = 0.0

    print(f"\n[INFO] Camera open. {INSTRUCTION_TEXT}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.05)
            continue

        # Mirror preview ONLY (do not mirror what is sent to detector)
        preview = cv2.flip(frame, 1)

        # Draw signing zone on preview
        draw_signing_guide(preview)

        hand_detected = False
        handedness    = "Unknown"
        flip_applied  = False
        current_crop  = None
        current_bbox  = None

        if detector is not None:
            try:
                detections = detector.detect(frame)   # unmirrored frame
                if detections:
                    det = detections[0]
                    hand_detected = True
                    handedness    = det["handedness"]
                    flip_applied  = (handedness == "Left")
                    bbox          = det["bbox"]
                    current_bbox  = bbox

                    # Draw bounding box on MIRRORED preview
                    fw = frame.shape[1]
                    mx1 = fw - bbox["x_max"]
                    mx2 = fw - bbox["x_min"]
                    color = (0, 255, 80) if det["crop_fully_within_frame"] else (0, 80, 255)
                    cv2.rectangle(preview,
                                  (mx1, bbox["y_min"]),
                                  (mx2, bbox["y_max"]),
                                  color, 2)
                    cv2.putText(preview, f"{handedness} {'[FLIP]' if flip_applied else ''}",
                                (mx1, bbox["y_min"] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Prepare crop (using unmirrored frame)
                    crop = detector.crop_hand(frame, bbox)
                    if crop is not None and crop.size > 0:
                        if flip_applied:
                            crop = cv2.flip(crop, 1)
                        current_crop = crop

                        # Show small crop thumbnail in corner
                        thumb = cv2.resize(crop, (128, 128))
                        preview[FRAME_HEIGHT - 134:FRAME_HEIGHT - 6,
                                FRAME_WIDTH  - 134:FRAME_WIDTH  - 6] = thumb
                        cv2.rectangle(preview,
                                      (FRAME_WIDTH - 136, FRAME_HEIGHT - 136),
                                      (FRAME_WIDTH - 4,   FRAME_HEIGHT - 4),
                                      (80, 80, 200), 1)
                        cv2.putText(preview, "CNN INPUT",
                                    (FRAME_WIDTH - 132, FRAME_HEIGHT - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
            except Exception as e:
                cv2.putText(preview, f"Detector err: {str(e)[:40]}", (6, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        draw_hud(preview, counts, last_label, hand_detected, handedness, flip_applied)
        cv2.imshow(WINDOW_TITLE, preview)

        raw_key = cv2.waitKey(1)
        if raw_key == -1:
            continue
        key = raw_key & 0xFF

        # Separate quit check - exit only on ESC (key code 27)
        if is_quit_key(key):
            print("\n[INFO] ESC pressed. Exiting capture tool...")
            break

        label = get_label_for_key(key)
        if label is None:
            continue

        now = time.time()
        if now - last_save < CAPTURE_COOLDOWN:
            continue   # debounce

        if current_crop is None:
            print(f"[SKIP] No valid crop for '{label}' — reposition hand")
            continue

        img_path = save_crop(
            current_crop, label,
            handedness, flip_applied,
            current_bbox or {}, frame.shape,
            session_log
        )
        counts[label] = counts.get(label, 0) + 1
        last_label = label
        last_save  = now
        print(f"[SAVED] {label} → {img_path.name}  (total {counts[label]})")

        # Flash green border to confirm save
        flash = preview.copy()
        cv2.rectangle(flash, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (0, 255, 0), 8)
        cv2.imshow(WINDOW_TITLE, flash)
        cv2.waitKey(120)

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Session Summary ===")
    total = 0
    for letter in sorted(counts):
        n = counts[letter]
        total += n
        flag = " ✓" if n >= 20 else f" (need {max(0, 20-n)} more)"
        print(f"  {letter}: {n:3d}{flag}")
    print(f"  TOTAL: {total}")
    print(f"\nData saved to: {OUTPUT_DIR}")
    print("Next step: run tools/finetune_webcam.py")


if __name__ == "__main__":
    main()
