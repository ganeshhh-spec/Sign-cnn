# Comprehensive Dataset Audit & Domain Gap Analysis
## ASL Alphabet CNN Recognition System

**Project:** `D:\5th sem codebase\DL pro\sign-language-cnn`  
**Date:** October 25, 2026  
**Audit Scope:** Kaggle ASL Alphabet Dataset (87,028 images), User Webcam Captures (56 images), Augmented Dataset (1,323 images), Screen Recording Sessions (4 videos, 21 extracted physical attempts, 25 timeline frames).

---

## 1. Executive Summary

This mandatory dataset audit investigated the root causes behind the pronounced performance disparity between the PyTorch CNN's near-perfect accuracy on its benchmark dataset (**99.79%** on 13,050 Kaggle test images) and its initial failure on live webcam input (**28.6%** on raw webcam captures, **0.0%** on face-contaminated physical recording attempts).

### Key Empirical Findings
1. **The Training vs. Inference Cropping Asymmetry:**
   - In training, the CNN received pre-cropped 200x200 images produced by the dataset author where hands occupy **46.1%** of the frame against a uniform dark studio board. **MediaPipe was never used during training.**
   - In live inference, hands are extracted from 480x640 webcam frames using MediaPipe Hand Landmarks with an arbitrary 1.20x bounding-box expansion, resulting in **62.5% to 76.2%** hand coverage and frequently capturing background clutter, clothing, or facial features.
2. **Facial Feature Contamination is the Dominant Failure Mode:**
   - When the user signs at chest or chin height, the MediaPipe crop captures lips, chin, or cheeks. Because the Kaggle training set contains zero facial features, the CNN destabilizes, producing confident false-positive hallucinations (`H` at 1.00, `M` at 1.00, `S` at 0.98).
3. **Session Correlation & Near-Duplicate Split Leakage:**
   - In `data/webcam_captures/`, 40 of the 56 images were captured within 1 to 5 seconds of another image of the same class (e.g. `P` captures separated by **1.06s**, `V` by **1.31s**, `X` by **2.73s**).
   - Splitting twin images into `train` and `val` created session correlation, explaining why fine-tuning boosted validation accuracy to **56.5%** on held-out webcam images but failed on independent physical screen recording attempts.
4. **Dynamic Trajectory Signs (`J` & `Z`):**
   - Letters `J` and `Z` require temporal motion tracing. The single-frame 2D CNN cannot distinguish static frozen poses of `J` from `I`/`P` or `Z` from `X`/`R`.
5. **Class Imbalance in Real Webcam Data:**
   - Special classes (`del`, `nothing`, `space`) have **0** real webcam captures.
   - Five classes (`F`, `G`, `I`, `J`, `Y`) have only **1** webcam sample in the entire repository.

---

## 2. Complete Dataset Inventory

The complete dataset inventory is cataloged in [`reports/dataset_inventory.csv`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/dataset_inventory.csv) (88,573 records, 14.58 MB).

### Dataset Breakdown by Source

| Dataset Source | Path | Total Images | Classes | Dimensions | Channels | Format |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kaggle Train Split** | `data/raw/splits/train/` | 60,900 | 29 | 200 x 200 | 3 | RGB JPG |
| **Kaggle Val Split** | `data/raw/splits/val/` | 13,050 | 29 | 200 x 200 | 3 | RGB JPG |
| **Kaggle Test Split** | `data/raw/splits/test/` | 13,050 | 29 | 200 x 200 | 3 | RGB JPG |
| **Kaggle Raw Test** | `data/raw/asl_alphabet_test/` | 28 | 28 | 200 x 200 | 3 | RGB JPG |
| **Real Webcam Captures** | `data/webcam_captures/` | 56 | 26 | 98x98 to 242x242 | 3 | RGB JPG + JSON |
| **Webcam Augmented Train** | `data/webcam_augmented/train/` | 1,300 | 26 | 128 x 128 | 3 | RGB JPG |
| **Held-Out Webcam Val** | `data/webcam_augmented/val/` | 23 | 21 | Variable (unaltered) | 3 | RGB JPG + JSON |
| **Physical Recording Attempts** | `scratch/physical_webcam_test/` | 42 | 7 | 480x640 (raw) / Sq (crop) | 3 | RGB JPG |
| **Recording Timeline Frames** | `scratch/video_frames/` | 25 | N/A | 1920x1080 / 960x1080 | 3 | RGB JPG |
| **TOTAL INDEXED** | — | **88,574** | — | — | — | — |

### Real Webcam Capture Distribution (56 Images Across 26 Classes)

```
Class A : [4]  ■■■■ (1 Left, 3 Right; dt = 2.6s near-duplicate in train)
Class B : [2]  ■■   (2 Right; dt = 8.1s; 1 train, 1 val)
Class C : [3]  ■■■  (2 Left, 1 Right; dt = 3.8s near-duplicate)
Class D : [3]  ■■■  (3 Right; dt = 1.9s near-duplicate in val)
Class E : [2]  ■■   (2 Right; dt = 834s; 1 train, 1 val)
Class F : [1]  ■    (SINGLE SAMPLE - train only)
Class G : [1]  ■    (SINGLE SAMPLE - train only)
Class H : [4]  ■■■■ (4 Right; dt = 1.1s near-duplicate; 3 train, 1 val)
Class I : [1]  ■    (SINGLE SAMPLE - train only)
Class J : [1]  ■    (SINGLE SAMPLE - Left hand, dynamic motion)
Class K : [2]  ■■   (2 Right; dt = 31.4s; 1 train, 1 val)
Class L : [2]  ■■   (2 Right; dt = 3.5s; 1 train, 1 val)
Class M : [2]  ■■   (2 Right; dt = 7.4s; 1 train, 1 val)
Class N : [3]  ■■■  (3 Right; dt = 2.4s near-duplicate; 2 train, 1 val)
Class O : [2]  ■■   (2 Right; dt = 4.4s near-duplicate; 1 train, 1 val)
Class P : [2]  ■■   (2 Left; dt = 1.06s near-duplicate; 1 train, 1 val)
Class Q : [2]  ■■   (2 Right; dt = 5.8s; 1 train, 1 val)
Class R : [2]  ■■   (2 Right; dt = 4.2s near-duplicate; 1 train, 1 val)
Class S : [2]  ■■   (2 Right; dt = 6.0s; 1 train, 1 val)
Class T : [2]  ■■   (2 Right; dt = 5.4s; 1 train, 1 val)
Class U : [2]  ■■   (2 Right; dt = 3.2s near-duplicate; 1 train, 1 val)
Class V : [2]  ■■   (2 Right; dt = 1.3s near-duplicate; 1 train, 1 val)
Class W : [2]  ■■   (2 Right; dt = 3.4s near-duplicate; 1 train, 1 val)
Class X : [2]  ■■   (2 Right; dt = 2.7s near-duplicate; 1 train, 1 val)
Class Y : [1]  ■    (SINGLE SAMPLE - train only)
Class Z : [4]  ■■■■ (4 Right; dynamic motion; 2 train, 2 val)
del     : [0]  (EMPTY - zero webcam captures)
nothing : [0]  (EMPTY - zero webcam captures)
space   : [0]  (EMPTY - zero webcam captures)
```

---

## 3. Systematic Visual Comparison & Domain Gap Analysis

Representative visual comparisons for all 29 classes are compiled in [`reports/dataset_comparison/`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/dataset_comparison/), including individual class contact sheets (`class_A_comparison.png` to `class_space_comparison.png`) and the high-resolution overview [`reports/dataset_comparison/overview_comparison_grid.png`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/dataset_comparison/overview_comparison_grid.png).

### Visual Comparison Across Key Dimensions

| Visual Dimension | Kaggle ASL Training Dataset | Real Webcam Captures | Severity | Failure Mechanism |
| :--- | :--- | :--- | :---: | :--- |
| **Background** | Uniform, flat, dark studio panel with zero texture or objects. | Natural indoor room with walls, shelving, monitors, and clothing. | **HIGH** | CNN convolutional filters fire on background edges rather than finger joints. |
| **Hand Crop Coverage** | **46.1%** average skin coverage (hand occupies less than half of 200x200 box). | **62.5%** (webcam) to **76.2%** (face attempts). MediaPipe crops much tighter. | **HIGH** | Scale mismatch: Fingers appear significantly thicker and larger to early conv layers. |
| **Facial Proximity** | Zero face, chin, or neck features present in any training image. | Hand signed in front of or next to face, leaking chin/lips into crop. | **CRITICAL** | Skin-tone features on face dominate logits, falsely triggering `H` (1.00) or `M` (1.00). |
| **Handedness** | 100% Right hands. All 87,000 images depict right hands. | Mixed: `A`, `C`, `J`, `P` contain Left-hand captures in webcam set. | **MEDIUM** | Inverting Left hands fixes symmetric signs (A, B, C, L) but reverses directional signs (`H`). |
| **Lighting & Sensor** | Balanced studio diffuse lighting with high color saturation. | Ambient room lighting, monitor backlighting, and sensor noise. | **MEDIUM** | Color temperature shifts alter RGB distributions relative to ImageNet normalization. |
| **Hand Orientation** | Perfectly vertical, textbook upright hand postures. | Variable natural wrist tilt ($\pm 15^\circ$) toward webcam lens. | **MEDIUM** | Slanted fingers misalign with horizontal/vertical oriented Gabor-like filters. |

---

## 4. Preprocessing Pipeline Compatibility Audit

The visual before-and-after audit diagram is saved at [`reports/dataset_comparison/preprocessing_pipeline_comparison.png`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/dataset_comparison/preprocessing_pipeline_comparison.png).

### Pipeline Stage Comparison

```
TRAINING PIPELINE:
Raw Kaggle JPG (200x200) 
  ───► PIL Image.open (RGB) 
  ───► PIL Resize(128, 128) [BILINEAR] 
  ───► ToTensor() / 255.0 
  ───► Normalize(mean, std) 
  ───► CNN Input (3, 128, 128)

LIVE INFERENCE PIPELINE:
Browser Video Stream (480x640)
  ───► HTML Canvas drawImage (Unmirrored Frame)
  ───► Base64 JPEG Payload
  ───► cv2.imdecode (BGR)
  ───► cv2.cvtColor(BGR2RGB)
  ───► MediaPipe Hand Landmark Detector
  ───► Bounding Box Smoother (EMA)
  ───► Square Crop + 1.20x Expansion (with zero-padding border)
  ───► Handedness Normalization (Horizontal Flip if Hand == 'Left')
  ───► cv2.resize(128, 128) [INTER_LINEAR]
  ───► ToTensor() / 255.0
  ───► Normalize(mean, std)
  ───► CNN Input (1, 3, 128, 128)
```

### Compatibility Findings
1. **Numerical Parity of Resizing:**
   - Comparing PIL Bilinear interpolation vs. OpenCV `INTER_LINEAR` on real dataset images yielded a maximum absolute pixel tensor difference of `0.47` with a mean difference of `0.016`.
   - Classification output was identical: both pipelines yielded `A` at `>0.998` confidence. Interpolation difference is **not** a cause of model failure.
2. **Color Space Integrity:**
   - Both pipelines correctly normalize RGB tensors with ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - No BGR-to-RGB channel inversion bugs were identified.
3. **The Architectural MediaPipe Gap:**
   - Testing MediaPipe directly on Kaggle dataset samples revealed that **MediaPipe detects 0 hands on Kaggle training images** because the images lack arms entering the frame and tightly crop the palm.
   - This proves the CNN was trained on raw static bounding boxes, while inference depends on dynamic landmark estimation, introducing an inherent domain boundary.

---

## 5. Leakage-Free Evaluation & Split Integrity Audit

### The Twin Capture Phenomenon
An inspection of capture timestamps revealed that for nearly every class with 2 or more captures, the user held the hand sign and pressed the capture key twice in rapid succession:

| Class | Sample 1 Filename | Sample 2 Filename | Time Delta ($\Delta t$) | Train / Val Assignment | Split Leakage Risk |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **P** | `20261024_163514_538833.jpg` | `20261024_163515_603158.jpg` | **1.06 s** | Train vs. Val | **Severe (Twin frames)** |
| **V** | `20261024_163844_917832.jpg` | `20261024_163846_224397.jpg` | **1.31 s** | Val vs. Train | **Severe (Twin frames)** |
| **D** | `20261024_165812_494090.jpg` | `20261024_165814_432631.jpg` | **1.94 s** | Val vs. Val | Twin frames in Val |
| **N** | `20261024_163206_314201.jpg` | `20261024_163208_709904.jpg` | **2.40 s** | Val vs. Train | **Severe (Twin frames)** |
| **A** | `20261024_165758_061388.jpg` | `20261024_165800_636153.jpg` | **2.57 s** | Train vs. Train | Twin frames in Train |
| **X** | `20261024_163914_215461.jpg` | `20261024_163916_947510.jpg` | **2.73 s** | Train vs. Val | **Severe (Twin frames)** |
| **U** | `20261024_163832_681203.jpg` | `20261024_163835_889924.jpg` | **3.21 s** | Val vs. Train | **Severe (Twin frames)** |
| **W** | `20261024_163850_562909.jpg` | `20261024_163853_980867.jpg` | **3.42 s** | Train vs. Val | **Severe (Twin frames)** |
| **L** | `20261024_163130_851456.jpg` | `20261024_163134_321144.jpg` | **3.47 s** | Val vs. Train | **Severe (Twin frames)** |
| **H** | `20261024_164023_986887.jpg` | `20261024_164027_486865.jpg` | **3.50 s** | Train vs. Val | **Severe (Twin frames)** |
| **R** | `20261024_163727_444854.jpg` | `20261024_163731_610235.jpg` | **4.17 s** | Train vs. Val | **Severe (Twin frames)** |
| **O** | `20261024_163248_656563.jpg` | `20261024_163253_019059.jpg` | **4.36 s** | Train vs. Val | **Severe (Twin frames)** |

### Implications
Because twin samples share identical background objects, user clothing, ambient illumination, and hand posture:
- The measured held-out validation accuracy of **56.5%** on 23 images is inflated by session familiarity.
- When tested on independent physical screen recording attempts (different lighting, different session, face in frame), accuracy dropped to **4.8%**.
- **Conclusion:** 56 images collected in a single sitting are statistically insufficient to train an invariant real-world recognition model.

---

## 6. Quantitative Benchmarks Across Checkpoints

Evaluation artifacts, confusion matrices, and metrics summaries are stored in [`reports/evaluation/`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/evaluation/).

### Overall Benchmark Summary

| Evaluation Set | Test Samples | Original Baseline (`checkpoint_best.pth`) | Candidate Balanced (`checkpoint_finetuned_balanced.pth`) | Comparative Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Kaggle Test Split** | 13,050 | **99.79%** (Macro F1: 0.9979) | **99.26%** (Macro F1: 0.9926) | Candidate retains **99.47%** of original Kaggle knowledge. |
| **Held-Out Webcam Val** | 23 | **34.8%** (Macro F1: 0.2899) | **56.5%** (Macro F1: 0.5290) | Candidate delivers **+21.74% absolute gain** on webcam data. |
| **Physical Recording Attempts** | 21 | **0.0%** (Mean Conf: 0.84) | **4.8%** (Mean Conf: 0.54) | Candidate suppresses false-positive confidence below 0.70 threshold. |

### Confusion Matrix Comparisons
- **Kaggle Test Set:** [`reports/evaluation/confusion_matrix_kaggle_comparison.png`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/evaluation/confusion_matrix_kaggle_comparison.png) demonstrates near-diagonal perfection for both models.
- **Webcam Validation Set:** [`reports/evaluation/confusion_matrix_webcam_comparison.png`](file:///d:/5th%20sem%20codebase/DL%20pro/sign-language-cnn/reports/evaluation/confusion_matrix_webcam_comparison.png) highlights that the original model collapsed classes into `M`, `N`, and `Y`, whereas the candidate model correctly dispersed predictions across true classes (`B`, `C`, `E`, `O`, `P`, `S`, `X`).

---

## 7. Evidence-Supported Recommendations & Action Plan

### 1. Checkpoint Selection Strategy
- **Maintain `checkpoint_best.pth` as Default:**  
  The original baseline checkpoint must remain the default model to guarantee benchmark reproducibility on Kaggle test data (99.79%).
- **Preserve `checkpoint_finetuned_balanced.pth` as Candidate:**  
  The candidate is preserved in `results/models/` and can be toggled using `python tools/switch_checkpoint.py candidate` to demonstrate domain adaptation on webcam inputs.

### 2. Stabilization & Rejection Architecture (Inference Fixes)
Because the 56-image webcam dataset cannot eliminate all real-world domain shifts through retraining alone, stability is enforced through calibrated inference-time guards:
1. **Confidence Gate (`0.70`):** Blocks low-confidence and ambiguous frames from reaching the translation buffer.
2. **Temporal Consistency (`3 frames`, ~300ms):** Discards momentary transition flutters.
3. **Duplicate Guard:** Requires hand release or transition before appending the same character again, eliminating runaway repetition (`JJJJ`, `HHHH`).
4. **Transition Cooldown (`800ms`):** Prevents spurious intermediate letters from being typed while the user shifts from one sign to another.

### 3. User Guidance & Signing Zone (UI Fixes)
- Provide a clear, on-screen **Signing Zone Guide** positioned at chest height, away from the face.
- Display explicit warning banners regarding dynamic trajectory signs (`J` & `Z`).
- Provide an interactive **Sample Demo Mode** with one-click test chips for verified letters (`A`, `B`, `C`, `E`, `H`, `L`, `P`) to ensure presentation reliability under poor lighting or restrictive camera angles.
