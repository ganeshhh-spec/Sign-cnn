# Empirical Evaluation of Landmark-Guided Orientation & Scale Normalization (OSN)

**Project:** `D:\5th sem codebase\DL pro\sign-language-cnn`  
**Date:** October 25, 2026  
**Status:** Implemented as Configurable Feature behind Flag (`enable_osn=False` by default)  
**Visual Artifacts:** [`reports/osn_comparison/`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/osn_comparison/)

---

## 1. Executive Summary & Objective Recommendation

### Final Recommendation: **REVISE & KEEP DISABLED BY DEFAULT (`enable_osn=False`)**

1. **Synthetic Validation Passes (100%):**  
   Under controlled synthetic rotations of known angles ($\pm 15^\circ, \pm 30^\circ, \pm 45^\circ$), OSN accurately detects the wrist-to-MCP vector and reorients the hand to vertical with residual palm-axis tilt $< 5^\circ$.
2. **Real Physical Webcam Impact: Baseline 12.5% vs. OSN 0.0%:**  
   On physical webcam recordings, OSN reorients the hand crop and rescales to ~48% coverage, but **does not overcome the fundamental background and skin-texture domain gap**. In fact, for letters with inherent horizontal or downward orientation (such as `Q`, `P`, `H`, `G`), forced vertical normalization violates standard ASL grammar and degrades recognition.
3. **Architecture & Safety Integrity (100% Passing):**  
   - 92 backend tests pass cleanly (`test_api.py`, `test_augment_webcam.py`, `test_capture_webcam.py`, `test_osn.py`).
   - Degenerate distance, missing landmarks, boundary crops, and edge padding all fall back gracefully to the production bounding-box crop.
   - Frontend production build compiles cleanly.
   - Zero changes made to model checkpoints.

---

## 2. Test 1: Synthetic Known Rotations Validation

We tested OSN on controlled synthetic rotations applied to textbook Kaggle gestures:

| Synthetic Tilt Angle | MediaPipe Detected | OSN Measured Angle | Residual Axis Tilt | Validation Status |
| :---: | :---: | :---: | :---: | :---: |
| **-45.0°** | True | +60.7° |  1.4° | **`PASS`** |
| **-30.0°** | True | +45.4° |  0.6° | **`PASS`** |
| **-15.0°** | True | +30.7° |  1.2° | **`PASS`** |
| ** +0.0°** | True | +15.9° |  1.4° | **`PASS`** |
| **+15.0°** | True |  +0.1° |  0.9° | **`PASS`** |
| **+30.0°** | True | -15.5° |  0.2° | **`PASS`** |
| **+45.0°** | True | -28.9° |  1.9° | **`PASS`** |

**Conclusion:** The mathematical rotation formula $\text{rot\_angle} = \theta + 90.0^\circ$ correctly normalizes hand orientation under clean background conditions.

---

## 3. Test 2: Controlled Side-by-Side Webcam Comparison

Evaluated on 8 independently labeled webcam gestures across physical recording sessions:

| Sample ID | Intended Sign | Hand | OSN Rotation Applied | Hand Coverage | Baseline Pred (Conf) | OSN Pred (Conf) | Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PWT_00** | **A** | Right | +45.0° | 50.0% | `S` (0.96) | `Z` (0.62) | **`SAME (INCORRECT)`** |
| **PWT_05** | **B** | Right | -18.1° | 50.0% | `K` (0.56) | `F` (0.26) | **`SAME (INCORRECT)`** |
| **PWT_07** | **B** | Left | -27.8° | 50.1% | `S` (0.83) | `N` (0.72) | **`SAME (INCORRECT)`** |
| **PWT_09** | **C** | Left | -17.4° | 50.0% | `S` (0.61) | `S` (0.50) | **`SAME (INCORRECT)`** |
| **PWT_12** | **L** | Left | -11.9° | 50.0% | `I` (0.87) | `W` (0.67) | **`SAME (INCORRECT)`** |
| **PWT_16** | **H** | Left | +36.8° | 50.0% | `H` (0.56) | `J` (0.78) | **`REGRESSED`** |
| **PWT_17** | **H** | Right | +14.8° | 50.0% | `M` (1.00) | `C` (0.52) | **`SAME (INCORRECT)`** |
| **PWT_20** | **M** | Right | +34.8° | 50.0% | `I` (0.99) | `Z` (0.63) | **`SAME (INCORRECT)`** |

### Comparative Accuracy & Metrics Summary:
- **Baseline Pipeline (Standard BBox Crop):** **1/8 (12.5%)**
- **OSN Pipeline (Landmark-Guided OSN Crop):** **0/8 (0.0%)**
- **Mean Confidence (Baseline):** 0.797
- **Mean Confidence (OSN):** 0.587

---

## 4. Key Empirical Insights: Why OSN Alone Does Not Fix Live Recognition

1. **ASL Hand Posture Is Non-Uniform in Orientation:**
   - Letters `A`, `B`, `C`, `D`, `E`, `F`, `K`, `L`, `R`, `U`, `V`, `W`, `X`, `Y` are upright.
   - Letters `G` and `H` are inherently **horizontal** (fingers point sideways).
   - Letters `P` and `Q` are inherently **downward** (wrist at top, fingers point down).
   - Forcing all gestures to point vertically upright strips away orientation features that distinguish `Q` from `G`/`H` and `P` from `K`.
2. **Background Clutter Dominates Convolutional Filters:**
   - The CNN backbone relies on $3\times 3$ convolutions without background segmentation. In residential webcam frames, sharp background edges from furniture and room clutter dominate feature activations regardless of crop orientation.
3. **The Stabilizer Guard Remains Essential:**
   - The temporal stabilizer (min 3 frames, 0.70 confidence threshold) successfully blocks ambiguous frames from reaching the text buffer.

---

## 5. Usage & Verification Commands

### Test Mode 1: Baseline Pipeline (Default, Production-Safe)
```powershell
# Executes standard bounding-box crop (enable_osn=False)
.\.venv\Scripts\python -m pytest backend/tests/test_api.py backend/tests/test_osn.py
```

### Test Mode 2: OSN Pipeline (Flag-Enabled)
```powershell
# Enable OSN via environment variable
$env:ASL_ENABLE_OSN="1"
.\.venv\Scripts\python tools/evaluate_osn.py
$env:ASL_ENABLE_OSN="0"
```

### Run Regression Suite & Frontend Build:
```powershell
# Backend regression suite (92 tests)
.\.venv\Scripts\python -m pytest backend/tests/

# Frontend production build
cd frontend; npm run build
```
