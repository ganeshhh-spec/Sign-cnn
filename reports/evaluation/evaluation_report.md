# Model Evaluation & Benchmark Report

Quantitative comparison between the Original Baseline and Candidate Balanced Fine-Tuned Checkpoints.

## Overall Benchmark Summary

| Evaluation Split | Sample Size | Original Baseline (`checkpoint_best.pth`) | Candidate Balanced (`checkpoint_finetuned_balanced.pth`) | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Kaggle Test Split** | 13,050 imgs | **99.79%** (F1: 0.9979) | **99.26%** (F1: 0.9926) | Candidate preserves **99.47%** of original Kaggle knowledge |
| **Held-Out Webcam Val** | 23 imgs | **34.8%** (F1: 0.2899) | **56.5%** (F1: 0.5290) | Candidate delivers **+21.7% absolute gain** on unseen webcam images |
| **Physical Screen Attempts** | 21 attempts | **0.0%** (Conf ~0.84) | **4.8%** (Conf ~0.54) | Candidate suppresses false positive confidence below 0.70 threshold |

## Per-Class Accuracy Breakdown (Held-Out Webcam Validation N=23)

| Class | Support | Original Accuracy | Candidate Accuracy | Result |
| :---: | :---: | :---: | :---: | :---: |
| **A** | 1 | 0% | 0% | UNALTERED |
| **B** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **C** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **D** | 2 | 0% | 0% | UNALTERED |
| **E** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **H** | 1 | 100% | 100% | UNALTERED |
| **K** | 1 | 0% | 0% | UNALTERED |
| **L** | 1 | 100% | 100% | UNALTERED |
| **M** | 1 | 100% | 0% | REGRESSED (-100%) |
| **N** | 1 | 100% | 0% | REGRESSED (-100%) |
| **O** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **P** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **Q** | 1 | 100% | 100% | UNALTERED |
| **R** | 1 | 100% | 100% | UNALTERED |
| **S** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **T** | 1 | 0% | 0% | UNALTERED |
| **U** | 1 | 100% | 100% | UNALTERED |
| **V** | 1 | 0% | 0% | UNALTERED |
| **W** | 1 | 100% | 100% | UNALTERED |
| **X** | 1 | 0% | 100% | **IMPROVED** (+100%) |
| **Z** | 2 | 0% | 0% | UNALTERED |

## Evaluation Limitations & Insights

1. **Support Constraint:** The held-out webcam validation split contains only 23 total images across 21 classes (1 or 2 images per class). While the +21.7% gain indicates genuine domain adaptation, the sample size is too small for statistical significance testing.
2. **False Confidence Suppression:** On face-contaminated physical attempts, the original model produced severe false-positive hallucinations at confidences of 0.80–1.00 (e.g. predicting `H`, `M`, or `S` when the hand was held near the chin). The candidate model reduced false confidence to 0.35–0.55, allowing the stabilizer's 0.70 threshold to reject ambiguous frames.
