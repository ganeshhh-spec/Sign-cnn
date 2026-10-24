# Dataset Preparation Documentation

## Dataset Source
- **Source**: Kaggle ASL Alphabet Dataset
- **URL**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **License**: GPL-2.0
- **Download method**: Kaggle CLI (authenticated)
- **Download date**: 2026-10-21
- **Raw data location**: `C:\Users\Ganesh\sign-language-cnn\data\raw\`

## Dataset Structure (Original)

### Training Set (`asl_alphabet_train/asl_alphabet_train/`)
- 29 class directories (A-Z, del, nothing, space)
- 3,000 images per class
- **Total**: 87,000 images
- Image format: JPG, 200×200 pixels, RGB

### Test Set (`asl_alphabet_test/asl_alphabet_test/`)
- 28 single images (one per class, missing `del`)
- File naming: `{CLASS}_test.jpg`
- Image format: JPG, 200×200 pixels, RGB

## Validation Results

### Class Completeness
✅ All 29 expected classes present in training set:
- A-Z (26 letter classes)
- `del` (delete)
- `nothing` (no hand / background)
- `space` (space character)

⚠️ Kaggle test set missing `del_test.jpg` (only 28/29 classes)

### Image Quality
- **Total images validated**: 87,000 (training)
- **Corrupt files**: 0
- **Exact duplicates**: 0 (MD5 hash comparison)
- **Image dimensions**: 200×200 pixels (100% consistent)
- **Color mode**: RGB (100% consistent)

### Per-Class Counts (Training)
| Class | Count | Class | Count | Class | Count |
|-------|-------|-------|-------|-------|-------|
| A | 3,000 | J | 3,000 | S | 3,000 |
| B | 3,000 | K | 3,000 | T | 3,000 |
| C | 3,000 | L | 3,000 | U | 3,000 |
| D | 3,000 | M | 3,000 | V | 3,000 |
| E | 3,000 | N | 3,000 | W | 3,000 |
| F | 3,000 | O | 3,000 | X | 3,000 |
| G | 3,000 | P | 3,000 | Y | 3,000 |
| H | 3,000 | Q | 3,000 | Z | 3,000 |
| I | 3,000 | R | 3,000 | del | 3,000 |
|  |  |  |  | nothing | 3,000 |
|  |  |  |  | space | 3,000 |

## Train/Validation/Test Splits

### Split Strategy
- **Method**: Stratified random split with fixed seed
- **Random seed**: 42 (reproducible)
- **Ratios**: 70% train / 15% validation / 15% test
- **Leakage prevention**: Exact duplicates kept in same split (0 duplicates found)

### Split Sizes
| Split | Total Images | Per Class | Percentage |
|-------|-------------|-----------|------------|
| Train | 60,900 | 2,100 | 70.0% |
| Validation | 13,050 | 450 | 15.0% |
| Test | 13,050 | 450 | 15.0% |
| **Total** | **87,000** | **3,000** | **100%** |

### Leakage Check Results
- **Train-Val overlap**: 0 images
- **Train-Test overlap**: 0 images  
- **Val-Test overlap**: 0 images
- ✅ **No data leakage detected**

### Split Directory Structure
```
data/raw/splits/
├── train/
│   ├── A/ (2,100 images)
│   ├── B/ (2,100 images)
│   ├── ...
│   ├── del/ (2,100 images)
│   ├── nothing/ (2,100 images)
│   └── space/ (2,100 images)
├── val/
│   ├── A/ (450 images)
│   ├── ...
│   └── space/ (450 images)
└── test/
    ├── A/ (450 images)
    ├── ...
    └── space/ (450 images)
```

## Preprocessing Configuration

### Target Specifications
- **Input format**: RGB
- **Target image size**: 128×128 pixels
- **Tensor conversion**: `ToTensor()` → [0, 1] → Normalize
- **Normalization**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Training Augmentation (Moderate)
| Transform | Parameters | Rationale |
|-----------|------------|-----------|
| RandomRotation | ±15° | Hand rotation variance |
| RandomAffine | translate±10%, scale 0.9-1.1, shear±5° | Position/scale variance |
| ColorJitter | brightness±0.2, contrast±0.2, saturation±0.2, hue±0.05 | Lighting variance |
| RandomHorizontalFlip | **DISABLED** (p=0.0) | **Flipping changes sign meaning** (e.g., 'L' ↔ 'mirror L') |

### Validation/Test Transforms (Deterministic)
- Resize to 128×128 (bilinear)
- ToTensor + Normalize
- **No random augmentation**

### Inference Transforms (Webcam)
- Identical to validation/test transforms
- Ensures consistent preprocessing between training and deployment

## Class Mapping (Stable for Inference)

| Index | Class | Index | Class | Index | Class |
|-------|-------|-------|-------|-------|-------|
| 0 | A | 10 | K | 20 | U |
| 1 | B | 11 | L | 21 | V |
| 2 | C | 12 | M | 22 | W |
| 3 | D | 13 | N | 23 | X |
| 4 | E | 14 | O | 24 | Y |
| 5 | F | 15 | P | 25 | Z |
| 6 | G | 16 | Q | 26 | del |
| 7 | H | 17 | R | 27 | nothing |
| 8 | I | 18 | S | 28 | space |
| 9 | J | 19 | T |  |  |

## PyTorch Verification

### DataLoader Test Results
```
Train DataLoader:
  Dataset size: 60,900
  Batch shape: [64, 3, 128, 128]
  Labels shape: [64]
  Label range: 0-28
  Classes per batch (shuffled): 26/29
  GPU transfer: ✅ OK (cuda:0)

Validation DataLoader:
  Dataset size: 13,050
  Batch shape: [64, 3, 128, 128]
  Labels shape: [64]
  Classes in full dataset: 29/29
  GPU transfer: ✅ OK (cuda:0)

Test DataLoader:
  Dataset size: 13,050
  Batch shape: [64, 3, 128, 128]
  Labels shape: [64]
  Classes in full dataset: 29/29
  GPU transfer: ✅ OK (cuda:0)
```

### Contact Sheets
Generated at `results/plots/`:
- `contact_sheet_train.png` - 1 sample per class (29 images)
- `contact_sheet_val.png` - 1 sample per class (29 images)
- `contact_sheet_test.png` - 1 sample per class (29 images)

## Files Created

### Scripts
- `backend/preprocessing/dataset_validation.py` - Dataset validation and quality checks
- `backend/preprocessing/dataset_splits.py` - Stratified split creation with leakage prevention
- `backend/preprocessing/transforms.py` - Preprocessing/augmentation configuration
- `backend/preprocessing/verify_dataset.py` - PyTorch DataLoader verification and contact sheets

### Outputs
- `results/metrics/dataset_validation.json` - Validation results
- `results/metrics/dataset_splits.json` - Split configuration and statistics
- `results/metrics/dataset_verification.json` - PyTorch verification results
- `results/plots/contact_sheet_{train,val,test}.png` - Visual samples

## Unresolved Issues / Notes

1. **Kaggle test set incomplete**: Missing `del_test.jpg` - Kaggle test directory not used for primary evaluation. Our custom test split (13,050 images, all 29 classes) is used instead.

2. **Original test set excluded**: The 28-image Kaggle test set is documented but not used in train/val/test splits. It can be used for qualitative inspection only.

3. **No near-duplicate detection**: Only exact pixel-level duplicates checked (0 found). Near-duplicates (e.g., same hand pose, slight lighting change) not analyzed due to computational cost.

4. **Dataset size**: 87,000 training images at 200×200 RGB ≈ 3.5 GB raw. Splits duplicate data (total ~10.5 GB on disk). Consider using index-based loading for future space savings.

5. **Class balance**: Perfectly balanced (3,000 per class). No class weighting needed for loss function.

## Next Steps
- Implement custom CNN architecture (Phase 3)
- Configure training loop with mixed precision
- Set up experiment tracking (TensorBoard/W&B)
- Begin model training on prepared splits