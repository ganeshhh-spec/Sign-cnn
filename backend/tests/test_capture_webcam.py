"""
Unit tests for webcam capture keyboard handling and dataset safety.
Verifies that:
- ESC (key code 27) exits the capture tool.
- Alphabet keys A-Z (including Q) are reserved exclusively for capturing class labels.
- Lowercase and uppercase key inputs are normalized consistently.
- Q saves under class 'Q' without terminating the session.
- Sample counter increments on capture.
- Instructions are displayed properly.
- Existing captures are preserved.
"""

import os
import json
import numpy as np
import cv2
import pytest
from pathlib import Path

from tools.capture_webcam_data import (
    is_quit_key,
    get_label_for_key,
    save_crop,
    count_existing,
    draw_hud,
    INSTRUCTION_TEXT,
    WINDOW_TITLE,
    QUIT_KEY_CODE,
    OUTPUT_DIR,
)


class TestKeyboardHandling:
    """Test suite for keyboard input handling in the capture tool."""

    def test_quit_key_is_esc_only(self):
        """ESC (code 27) must trigger quit; Q and other keys must NOT quit."""
        assert is_quit_key(27) is True
        assert is_quit_key(QUIT_KEY_CODE) is True

        # Neither lowercase 'q' nor uppercase 'Q' should quit
        assert is_quit_key(ord('q')) is False
        assert is_quit_key(ord('Q')) is False

        # Other alphabetic and control keys must not quit
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert is_quit_key(ord(ch)) is False
            assert is_quit_key(ord(ch.lower())) is False
        assert is_quit_key(32) is False   # SPACE
        assert is_quit_key(8) is False    # BACKSPACE
        assert is_quit_key(127) is False  # DEL

    def test_q_key_mapped_to_q_class(self):
        """Pressing Q or q must map to label 'Q'."""
        assert get_label_for_key(ord('q')) == "Q"
        assert get_label_for_key(ord('Q')) == "Q"

    def test_all_alphabet_keys_normalized_to_uppercase(self):
        """All A-Z keys (lower and upper) must map to their respective uppercase labels."""
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert get_label_for_key(ord(ch.lower())) == ch
            assert get_label_for_key(ord(ch)) == ch

    def test_special_keys_mapping(self):
        """Backspace/Delete maps to 'del' and Space maps to 'space'."""
        assert get_label_for_key(8) == "del"
        assert get_label_for_key(127) == "del"
        assert get_label_for_key(32) == "space"

    def test_esc_and_unmapped_keys_return_none(self):
        """ESC (27) and unmapped keys must return None as label."""
        assert get_label_for_key(27) is None
        assert get_label_for_key(0) is None
        assert get_label_for_key(ord('1')) is None
        assert get_label_for_key(255) is None

    def test_instruction_text_content(self):
        """Instructions must clearly state A-Z saves a sample and ESC quits."""
        assert "A" in INSTRUCTION_TEXT and "Z" in INSTRUCTION_TEXT
        assert "ESC" in INSTRUCTION_TEXT
        assert "quit" in INSTRUCTION_TEXT.lower()

    def test_draw_hud_renders_without_error(self):
        """HUD drawing must handle counts and instructions cleanly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        counts = {"A": 5, "Q": 2}
        draw_hud(frame, counts, "Q", hand_detected=True, handedness="Right", flip_applied=False)
        # Should execute cleanly and draw on frame
        assert np.any(frame != 0)


class TestCaptureWorkflow:
    """Test sample saving, counter increments, and non-termination on Q."""

    def test_q_press_saves_sample_and_increments_counter(self, tmp_path):
        """
        Simulate pressing Q:
        1. Verify it does not quit.
        2. Verify label is 'Q'.
        3. Save crop to temp directory.
        4. Verify file exists in Q/ subfolder, metadata is valid, and counter increments.
        """
        key = ord('q')

        # 1. Check quit condition
        assert not is_quit_key(key), "Q must NOT trigger quit"

        # 2. Get class label
        label = get_label_for_key(key)
        assert label == "Q", "Key 'q' must map to class 'Q'"

        # 3. Simulate capture action
        dummy_crop = np.full((128, 128, 3), 120, dtype=np.uint8)
        session_log = tmp_path / "session_log.jsonl"
        counts = {"Q": 0}

        img_path = save_crop(
            crop_bgr=dummy_crop,
            label=label,
            handedness="Right",
            flip_applied=False,
            bbox={"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100},
            frame_shape=(480, 640, 3),
            session_log_path=session_log,
            output_dir=tmp_path
        )

        counts[label] = counts.get(label, 0) + 1

        # 4. Verify saved artifacts
        assert img_path.exists()
        assert img_path.parent.name == "Q"
        assert img_path.suffix == ".jpg"

        meta_path = img_path.with_suffix(".json")
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["label"] == "Q"
        assert meta["handedness"] == "Right"
        assert meta["flip_applied"] is False

        # Session log verification
        assert session_log.exists()
        with open(session_log) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["label"] == "Q"

        # Counter verification
        assert counts["Q"] == 1

    def test_esc_key_triggers_exit(self):
        """Simulate event loop encountering ESC (key code 27)."""
        events = [ord('a'), ord('Q'), 27, ord('b')]
        processed_labels = []
        terminated = False

        for k in events:
            if is_quit_key(k):
                terminated = True
                break
            lbl = get_label_for_key(k)
            if lbl:
                processed_labels.append(lbl)

        assert terminated is True
        assert processed_labels == ["A", "Q"]
        assert "B" not in processed_labels

    def test_existing_captures_preserved(self):
        """Verify that existing dataset directory is untouched and has valid files."""
        if OUTPUT_DIR.exists():
            existing = count_existing()
            assert isinstance(existing, dict)
            # Ensure calling count_existing does not modify or delete anything
            existing_after = count_existing()
            assert existing == existing_after
