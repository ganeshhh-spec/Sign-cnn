# Visual Comparison Summary: Kaggle ASL vs. Real Webcam Dataset

Generated as part of the Comprehensive Dataset Audit.

| Class | Kaggle Count | Webcam Count | Handedness | Audit Flags | Hand Shape Fidelity | Background & Lighting |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **A** | 2100 | 4 | Left & Right | `LEFT_HAND_PRESENT` | Verified correct handshape | Home office background with shelves/clothing |
| **B** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **C** | 2100 | 3 | Left & Right | `LEFT_HAND_PRESENT` | Verified correct handshape | Home office background with shelves/clothing |
| **D** | 2100 | 3 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **E** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **F** | 2100 | 1 | Right | `SINGLE_WEBCAM_SAMPLE` | Verified correct handshape | Home office background with shelves/clothing |
| **G** | 2100 | 1 | Right | `SINGLE_WEBCAM_SAMPLE` | Verified correct handshape | Home office background with shelves/clothing |
| **H** | 2100 | 4 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **I** | 2100 | 1 | Right | `SINGLE_WEBCAM_SAMPLE` | Verified correct handshape | Home office background with shelves/clothing |
| **J** | 2100 | 1 | Left & Right | `SINGLE_WEBCAM_SAMPLE | DYNAMIC_MOTION_GESTURE | LEFT_HAND_PRESENT` | Static snapshot only; lacks temporal motion curve | Home office background with shelves/clothing |
| **K** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **L** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **M** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **N** | 2100 | 3 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **O** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **P** | 2100 | 2 | Left & Right | `LEFT_HAND_PRESENT` | Verified correct handshape | Home office background with shelves/clothing |
| **Q** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **R** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **S** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **T** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **U** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **V** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **W** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **X** | 2100 | 2 | Right | `STANDARD ASL POSE` | Verified correct handshape | Home office background with shelves/clothing |
| **Y** | 2100 | 1 | Right | `SINGLE_WEBCAM_SAMPLE` | Verified correct handshape | Home office background with shelves/clothing |
| **Z** | 2100 | 4 | Right | `DYNAMIC_MOTION_GESTURE` | Static snapshot only; lacks temporal motion curve | Home office background with shelves/clothing |
| **del** | 2100 | 0 | N/A | `NO_WEBCAM_SAMPLES` | N/A (no webcam images) | N/A |
| **nothing** | 2100 | 0 | N/A | `NO_WEBCAM_SAMPLES` | N/A (no webcam images) | N/A |
| **space** | 2100 | 0 | N/A | `NO_WEBCAM_SAMPLES` | N/A (no webcam images) | N/A |

## Major Visual Findings

1. **Background Disparity:** Kaggle images feature a plain, dark studio backdrop. Real webcam captures feature natural room backgrounds with desks, shelving, and ambient clutter.
2. **Cropping & Scale:** Kaggle hand images occupy ~46% of the frame with uniform border margins. MediaPipe live crops occupy 62–76% of the frame, resulting in an enlarged scale when fed to the CNN.
3. **Missing Classes:** `del`, `nothing`, and `space` have zero real webcam images captured.
4. **Dynamic Signs:** `J` and `Z` are dynamic trajectory signs. Static snapshots cannot represent the tracing motion required for correct interpretation.
5. **Handedness:** Real webcam captures contain left-hand signs for `A`, `C`, `J`, and `P`. Horizontal mirroring normalizes static symmetric signs, but directional signs like `H` are inverted.
