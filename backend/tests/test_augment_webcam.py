"""
Unit tests for the ASL Training Image Augmentation Pipeline.
Verifies that:
- Geometric and photometric transformations respect safety bounds.
- Horizontal flipping is explicitly disabled.
- Train and validation splits are partitioned BEFORE augmentation.
- Validation set contains ONLY genuine, unaugmented captures.
- No validation source image produces augmented variants in the training set.
- Manifest accurately records all transformations.
- Missing classes (del, nothing, space) are protected and not hallucinated.
- Movement limitations on J and Z are recognized.
"""

import json
import random
import shutil
from pathlib import Path
import numpy as np
import cv2
import pytest

from tools.augment_webcam_data import (
    generate_augmented_sample,
    split_webcam_dataset,
    run_augmentation,
    apply_geometric_transform,
    apply_photometric_transform,
    apply_sensor_noise,
    DYNAMIC_SIGNS,
)
from backend.preprocessing.transforms import CLASSES


class TestAugmentationTransforms:
    """Test individual transformation operations and safety bounds."""

    def test_geometric_transform_preserves_shape(self):
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        transformed = apply_geometric_transform(img, angle=7.5, scale=1.02, dx=4.0, dy=-3.0)
        assert transformed.shape == img.shape
        assert transformed.dtype == np.uint8

    def test_photometric_transform_preserves_bounds(self):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        transformed = apply_photometric_transform(
            img, brightness=1.1, contrast=0.95, saturation=1.05, hue_delta=2.0
        )
        assert transformed.shape == img.shape
        assert transformed.min() >= 0
        assert transformed.max() <= 255

    def test_sensor_noise_bounds(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(42)
        noisy = apply_sensor_noise(img, noise_sigma=3.0, rng=rng)
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8
        # Noise should create variation around 128
        assert np.any(noisy != 128)

    def test_sample_generator_safety_bounds(self):
        img = np.full((140, 140, 3), 150, dtype=np.uint8)
        rng_py = random.Random(42)
        rng_np = np.random.default_rng(42)

        for _ in range(20):
            aug_img, params = generate_augmented_sample(img, rng_py, rng_np)
            assert aug_img.shape == img.shape
            assert -10.0 <= params["angle_deg"] <= 10.0
            assert 0.94 <= params["scale"] <= 1.06
            assert -140 * 0.06 <= params["translate_dx_px"] <= 140 * 0.06
            assert -140 * 0.06 <= params["translate_dy_px"] <= 140 * 0.06
            assert 0.88 <= params["brightness_factor"] <= 1.12
            assert 0.88 <= params["contrast_factor"] <= 1.12
            assert 0.88 <= params["saturation_factor"] <= 1.12
            assert params["horizontal_flip"] is False


class TestSplitAndPipelineIsolation:
    """Test train/val splitting and isolation before augmentation."""

    @pytest.fixture
    def mock_captures_dir(self, tmp_path):
        """Creates a mock webcam capture directory with multiple and single sample classes."""
        base = tmp_path / "webcam_mock"
        base.mkdir()

        # Class A: 4 samples in 2 distinct sessions
        a_dir = base / "A"
        a_dir.mkdir()
        for idx, ts in enumerate([
            "20261024_160000_100000",
            "20261024_160002_200000",  # Session 1 (<10s)
            "20261024_163000_100000",
            "20261024_163003_200000",  # Session 2 (>10s)
        ]):
            img = np.full((128, 128, 3), 100 + idx * 20, dtype=np.uint8)
            cv2.imwrite(str(a_dir / f"{ts}.jpg"), img)

        # Class B: 2 samples in 1 session
        b_dir = base / "B"
        b_dir.mkdir()
        for idx, ts in enumerate([
            "20261024_170000_100000",
            "20261024_170004_200000",
        ]):
            img = np.full((128, 128, 3), 50 + idx * 30, dtype=np.uint8)
            cv2.imwrite(str(b_dir / f"{ts}.jpg"), img)

        # Class F: 1 sample
        f_dir = base / "F"
        f_dir.mkdir()
        img = np.full((128, 128, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(f_dir / "20261024_180000_000000.jpg"), img)

        return base

    def test_split_webcam_dataset_no_leakage(self, mock_captures_dir):
        train_splits, val_splits = split_webcam_dataset(mock_captures_dir, val_ratio=0.25, seed=42)

        for cls in ["A", "B", "F"]:
            train_names = {p.name for p in train_splits[cls]}
            val_names = {p.name for p in val_splits[cls]}
            # Train and val must be mutually exclusive
            assert train_names.isdisjoint(val_names), f"Class {cls} has train/val intersection!"

        # A should have split by session
        assert len(train_splits["A"]) > 0
        assert len(val_splits["A"]) > 0

    def test_pipeline_prevents_val_variants_in_train(self, mock_captures_dir, tmp_path):
        out_dir = tmp_path / "augmented_out"
        summary = run_augmentation(
            input_dir=mock_captures_dir,
            output_dir=out_dir,
            target_per_class=10,
            seed=42
        )

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        entries = manifest["entries"]
        val_sources = {e["source_file"] for e in entries if e["split"] == "val"}
        train_sources = {e["source_file"] for e in entries if e["split"] == "train"}

        # Augmented training samples must NEVER originate from a validation capture
        for e in entries:
            if e["split"] == "train" and e["is_augmented"]:
                assert e["source_file"] not in val_sources
                assert e["source_file"] in train_sources

        # Validation images must never be augmented
        for e in entries:
            if e["split"] == "val":
                assert e["is_augmented"] is False
                assert e["is_genuine_webcam"] is True

    def test_missing_classes_not_hallucinated(self, mock_captures_dir, tmp_path):
        out_dir = tmp_path / "augmented_out"
        run_augmentation(
            input_dir=mock_captures_dir,
            output_dir=out_dir,
            target_per_class=10,
            seed=42
        )

        # del, nothing, space must have 0 samples
        for cls in ["del", "nothing", "space"]:
            train_cls = out_dir / "train" / cls
            val_cls = out_dir / "val" / cls
            train_jpgs = list(train_cls.glob("*.jpg"))
            val_jpgs = list(val_cls.glob("*.jpg"))
            assert len(train_jpgs) == 0
            assert len(val_jpgs) == 0

    def test_dynamic_signs_annotated(self):
        assert "J" in DYNAMIC_SIGNS
        assert "Z" in DYNAMIC_SIGNS
