"""
Tests for ASL Recognition API.
Run with: python -m pytest backend/tests/test_api.py -v
"""

import base64
import io
import sys
import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
from pydantic import ValidationError
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import (
    app, PredictRequest, health_check, get_classes, predict, translate,
    clear_translation, insert_space, delete_character, reset_translation,
    get_translation_state, get_preprocessing_config, get_model, get_detector,
    get_stabilizer_service, get_buffer,
)
from backend.app.services.model_loader import get_model_loader, initialize_model
from backend.app.services.hand_detector import HandDetector, get_hand_detector_stream, BoundingBoxSmoother
from backend.app.services.stabilizer import (
    PredictionStabilizer, TextBuffer, get_stabilizer, get_text_buffer, reset_translation_state,
    StabilizedPrediction, ActionType
)


# Initialize model before tests
initialize_model()

class DirectResponse:
    """Small response object matching the assertions used in this test module."""

    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def resolve_immediate(coroutine):
    """Resolve endpoints which contain no suspension points without an event loop."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise RuntimeError("Test endpoint unexpectedly requires an event loop")


def response_data(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


class DirectEndpointClient:
    """Exercise real endpoint functions when Windows loopback is unavailable.

    This test-only fixture replaces Starlette TestClient, whose AnyIO portal
    cannot create its asyncio self-pipe in this host (socketpair blocks in
    accept()).  It does not mock model, detector, stabilizer, or endpoint
    behavior; it only bypasses the broken local ASGI transport layer.
    """

    def request(self, method: str, url: str, **kwargs):
        try:
            if method == "GET" and url == "/api/health":
                value = resolve_immediate(health_check(get_model()))
            elif method == "GET" and url == "/api/classes":
                value = resolve_immediate(get_classes(get_model()))
            elif method == "POST" and url == "/api/predict":
                request = PredictRequest.model_validate(kwargs.get("json", {}))
                value = resolve_immediate(predict(request, get_model(), get_detector(), get_stabilizer_service(), get_buffer()))
            elif method == "POST" and url == "/api/translate":
                value = resolve_immediate(translate(get_stabilizer_service(), get_buffer()))
            elif method == "POST" and url == "/api/translate/clear":
                value = resolve_immediate(clear_translation(get_stabilizer_service(), get_buffer()))
            elif method == "POST" and url == "/api/translate/space":
                value = resolve_immediate(insert_space(get_buffer()))
            elif method == "POST" and url == "/api/translate/delete":
                value = resolve_immediate(delete_character(get_buffer()))
            elif method == "POST" and url == "/api/translate/reset":
                value = resolve_immediate(reset_translation(get_stabilizer_service(), get_buffer(), get_detector()))
            elif method == "GET" and url == "/api/translate/state":
                value = resolve_immediate(get_translation_state(get_stabilizer_service(), get_buffer()))
            elif method == "GET" and url == "/api/preprocessing":
                value = resolve_immediate(get_preprocessing_config(get_model()))
            else:
                raise AssertionError(f"Unexpected endpoint in test: {method} {url}")
            return DirectResponse(200, response_data(value))
        except ValidationError as exc:
            return DirectResponse(422, {"detail": exc.errors()})
        except HTTPException as exc:
            return DirectResponse(exc.status_code, {"detail": exc.detail})

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


# The model is initialized once for the suite, matching the previous setup.
# Requests exercise the real ASGI application without an already-running server.
client = DirectEndpointClient()


class TestModelLoading:
    """Test model loading and basic properties."""
    
    def test_model_loaded(self):
        """Test that model loads successfully."""
        loader = get_model_loader()
        # Model should be loaded on startup
        assert loader.is_loaded(), "Model should be loaded"
    
    def test_model_info(self):
        """Test model info endpoint data."""
        loader = get_model_loader()
        info = loader.get_info()
        
        assert info["loaded"] is True
        assert info["num_classes"] == 29
        assert len(info["classes"]) == 29
        assert info["input_size"] == (3, 128, 128)
        assert "mean" in info["normalization"]
        assert "std" in info["normalization"]
    
    def test_class_mapping(self):
        """Test class mapping matches expected 29 classes."""
        loader = get_model_loader()
        classes = loader.get_info()["classes"]
        
        expected = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        assert classes == expected
    
    def test_inference_forward_pass(self):
        """Test model forward pass works."""
        loader = get_model_loader()
        tensor = torch.randn(1, 3, 128, 128)
        
        predictions, logits = loader.predict(tensor)
        
        assert len(predictions) == 1
        assert len(predictions[0]) == 3  # top-3
        assert logits.shape == (1, 29)
        
        # Check prediction structure
        pred = predictions[0][0]
        assert "rank" in pred
        assert "class_index" in pred
        assert "class_label" in pred
        assert "confidence" in pred
        assert 0 <= pred["confidence"] <= 1


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint(self):
        """Test GET /api/health returns correct structure."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "num_classes" in data
        assert "gpu_available" in data
        assert data["num_classes"] == 29


class TestClassesEndpoint:
    """Test classes listing endpoint."""
    
    def test_classes_endpoint(self):
        """Test GET /api/classes returns 29 classes."""
        response = client.get("/api/classes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "classes" in data
        assert "count" in data
        assert data["count"] == 29
        assert len(data["classes"]) == 29
        
        # Check structure
        for cls in data["classes"]:
            assert "index" in cls
            assert "label" in cls
        
        # Check first and last
        assert data["classes"][0]["label"] == "A"
        assert data["classes"][28]["label"] == "space"


class TestImageValidation:
    """Test image validation and error handling."""
    
    def create_test_image(self, size=(200, 200), color=(255, 0, 0)) -> str:
        """Create a test base64 encoded image."""
        img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def test_valid_image_prediction(self):
        """Test prediction with valid image."""
        test_img = self.create_test_image()
        
        response = client.post("/api/predict", json={
            "image": test_img,
            "use_stabilizer": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "predictions" in data
        assert "hand_detected" in data
        assert "inference_time_ms" in data
        assert "total_time_ms" in data
    
    def test_invalid_base64(self):
        """Test handling of invalid base64."""
        response = client.post("/api/predict", json={
            "image": "not_valid_base64!!!",
            "use_stabilizer": False
        })
        
        assert response.status_code == 400
        data = response.json()
        # FastAPI HTTPException returns detail field
        assert "detail" in data or "error" in data
    
    def test_missing_image_field(self):
        """Test missing image field."""
        response = client.post("/api/predict", json={
            "use_stabilizer": False
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_empty_image(self):
        """Test empty image data."""
        response = client.post("/api/predict", json={
            "image": "",
            "use_stabilizer": False
        })
        
        assert response.status_code == 400


class TestHandDetection:
    """Test MediaPipe hand detection."""
    
    def test_detector_initialization(self):
        """Test hand detector can be created."""
        detector = get_hand_detector_stream()
        assert detector is not None
        assert isinstance(detector, HandDetector)
        detector.close()
    
    def test_no_hand_detected(self):
        """Test response when no hand is in frame."""
        # Create blank image (no hand)
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_img)
        test_img = base64.b64encode(buffer).decode('utf-8')
        
        response = client.post("/api/predict", json={
            "image": test_img,
            "use_stabilizer": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["hand_detected"] is False
        assert data["predictions"] == []
    
    def test_hand_bbox_format(self):
        """Test hand bounding box format and square cropping."""
        detector = get_hand_detector_stream()
        if detector is None:
            pytest.skip("Hand detector not available")
        
        # Test square crop logic directly with a non-square bbox
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = {"x_min": 100, "y_min": 50, "x_max": 200, "y_max": 300}
        crop = detector.crop_hand(frame, bbox)
        assert crop is not None
        assert crop.shape[0] == crop.shape[1], f"Expected square crop, got {crop.shape}"


class TestPredictionStabilizer:
    """Test prediction stabilization logic."""
    
    def setup_method(self):
        """Reset stabilizer before each test."""
        reset_translation_state()
    
    def test_stabilizer_initialization(self):
        """Test stabilizer creates correctly."""
        stabilizer = get_stabilizer(
            window_size=5,
            min_consistent=3,
            confidence_threshold=0.7
        )
        
        assert stabilizer.window_size == 5
        assert stabilizer.min_consistent == 3
        assert stabilizer.confidence_threshold == 0.7
    
    def test_insufficient_history(self):
        """Test stabilizer returns NONE before min_consistent predictions."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3)
        
        # Only 2 predictions - not enough
        result1 = stabilizer.process("A", 0.9)
        result2 = stabilizer.process("A", 0.9)
        
        assert result1.action == ActionType.NONE
        assert result2.action == ActionType.NONE
        assert result1.is_stable is False
    
    def test_consistent_predictions_accepted(self):
        """Test consistent predictions are accepted after min_consistent."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7)
        
        # 3 consistent predictions
        for _ in range(3):
            result = stabilizer.process("A", 0.9)
        
        assert result.action == ActionType.APPEND
        assert result.character == "A"
        assert result.is_stable is True
        assert result.confidence == 0.9
    
    def test_inconsistent_predictions_rejected(self):
        """Test inconsistent predictions are rejected."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7)
        
        # Mix of A and B
        stabilizer.process("A", 0.9)
        stabilizer.process("A", 0.9)
        result = stabilizer.process("B", 0.9)
        
        assert result.action == ActionType.NONE
    
    def test_low_confidence_rejected(self):
        """Test low confidence predictions are rejected."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7)
        
        # 3 predictions but low confidence
        for _ in range(3):
            result = stabilizer.process("A", 0.5)
        
        assert result.action == ActionType.NONE
        assert result.confidence == 0.5
    
    def test_space_handling(self):
        """Test SPACE class handling."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3)
        
        for _ in range(3):
            result = stabilizer.process("space", 0.9)
        
        assert result.action == ActionType.SPACE
        assert result.character == " "
    
    def test_delete_handling(self):
        """Test DELETE class handling."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3)
        
        for _ in range(3):
            result = stabilizer.process("del", 0.9)
        
        assert result.action == ActionType.DELETE
        assert result.character == ""
    
    def test_nothing_resets_guard(self):
        """Test NOTHING class resets duplicate guard."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, duplicate_guard=True)
        
        # Append A
        for _ in range(3):
            stabilizer.process("A", 0.9)
        
        # Now NOTHING
        for _ in range(3):
            result = stabilizer.process("nothing", 0.9)
        
        assert result.action == ActionType.NONE
        
        # Should be able to append A again
        for _ in range(3):
            result = stabilizer.process("A", 0.9)
        
        assert result.action == ActionType.APPEND
        assert result.character == "A"
    
    def test_duplicate_guard(self):
        """Test duplicate letter guard prevents repeats."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, duplicate_guard=True)
        
        # Append A
        for _ in range(3):
            result = stabilizer.process("A", 0.9)
        assert result.action == ActionType.APPEND
        
        # Try to append A again immediately
        for _ in range(3):
            result = stabilizer.process("A", 0.9)
        
        # Should be rejected by duplicate guard
        assert result.action == ActionType.NONE
    
    def test_reset(self):
        """Test stabilizer reset."""
        stabilizer = get_stabilizer()
        
        for _ in range(3):
            stabilizer.process("A", 0.9)
        
        stabilizer.reset()
        
        assert len(stabilizer.prediction_window) == 0
        assert stabilizer.last_appended is None

    def test_low_confidence_frames_never_append(self):
        """Test low-confidence frames (e.g. 0.64) never trigger append even if mixed with high confidence."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7)
        stabilizer.reset()
        
        # High, then low (0.64), then high - cannot trigger append
        r1 = stabilizer.process("H", 0.75)
        r2 = stabilizer.process("H", 0.64)  # Below 0.70 threshold!
        r3 = stabilizer.process("H", 0.75)
        
        assert r1.action == ActionType.NONE
        assert r2.action == ActionType.NONE
        assert r3.action == ActionType.NONE
        assert r3.is_stable is False

    def test_no_hand_resets_duplicate_guard(self):
        """Test that dropping hand (no-hand frames) acts as release, allowing repeated intentional letters."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, duplicate_guard=True)
        stabilizer.reset()
        
        # Append 'P'
        for _ in range(3):
            r = stabilizer.process("P", 0.9)
        assert r.action == ActionType.APPEND
        assert r.character == "P"
        
        # User drops hand (2 frames of no hand detected)
        nh1 = stabilizer.process_no_hand()
        nh2 = stabilizer.process_no_hand()
        assert nh1.action == ActionType.NONE
        assert nh2.action == ActionType.NONE
        assert stabilizer.last_appended is None  # Guard reset!
        
        # User intentionally signs 'P' again (e.g. spelling "APPLE")
        for _ in range(3):
            r2 = stabilizer.process("P", 0.9)
        assert r2.action == ActionType.APPEND
        assert r2.character == "P"

    def test_held_sign_appends_only_once(self):
        """Test holding a sign steady for 15 frames appends exactly once."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, duplicate_guard=True)
        stabilizer.reset()
        
        append_count = 0
        for _ in range(15):
            res = stabilizer.process("A", 0.85)
            if res.action == ActionType.APPEND:
                append_count += 1
                
        assert append_count == 1

    def test_transition_cooldown_prevents_rapid_wobble(self):
        """Test that rapid fluctuations without neutral release are throttled by cooldown."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, cooldown_seconds=0.8)
        stabilizer.reset()
        
        # User signs 'P'
        for _ in range(3):
            r_p = stabilizer.process("P", 0.85)
        assert r_p.action == ActionType.APPEND
        assert r_p.character == "P"
        
        # 300ms later during hand wobble, model flickers to 'H' without hand release
        for _ in range(3):
            r_h = stabilizer.process("H", 0.85)
            
        # Cooldown prevents accidental 'H' from immediately appending
        assert r_h.action == ActionType.NONE
        assert r_h.is_stable is False

    def test_consistent_wrong_class_is_still_explicitly_accepted(self):
        """Stabilization proves consistency, not semantic correctness of the CNN."""
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7, cooldown_seconds=0.0)
        stabilizer.reset()
        for _ in range(3):
            result = stabilizer.process("N", 0.95)
        assert result.action == ActionType.APPEND
        assert result.character == "N"

    def test_contributing_frame_metadata_is_retained_for_action_diagnostics(self):
        stabilizer = get_stabilizer(window_size=5, min_consistent=3, confidence_threshold=0.7, cooldown_seconds=0.0)
        stabilizer.reset()
        for frame_id in range(3):
            result = stabilizer.process("H", 0.85 + frame_id * 0.01, {
                "frame_id": frame_id,
                "hand_detected": True,
                "crop_bounds": {"x_min": 1, "y_min": 2, "x_max": 101, "y_max": 102},
            })
        assert result.action == ActionType.APPEND
        frames = stabilizer.get_state()["recent_frames"][-3:]
        assert [frame["frame_id"] for frame in frames] == [0, 1, 2]
        assert [frame["confidence"] for frame in frames] == [0.85, 0.86, 0.87]


class TestTextBuffer:
    """Test text buffer operations."""
    
    def setup_method(self):
        reset_translation_state()
    
    def test_append(self):
        """Test appending characters."""
        buffer = get_text_buffer()
        
        buffer.append("A")
        buffer.append("B")
        buffer.append("C")
        
        assert buffer.get_text() == "ABC"
    
    def test_delete(self):
        """Test delete operation."""
        buffer = get_text_buffer()
        
        buffer.append("A")
        buffer.append("B")
        buffer.delete()
        
        assert buffer.get_text() == "A"
    
    def test_space(self):
        """Test space operation."""
        buffer = get_text_buffer()
        
        buffer.append("A")
        buffer.space()
        buffer.append("B")
        
        assert buffer.get_text() == "A B"
    
    def test_clear(self):
        """Test clear operation."""
        buffer = get_text_buffer()
        
        buffer.append("A")
        buffer.append("B")
        buffer.clear()
        
        assert buffer.get_text() == ""
    
    def test_set_text(self):
        """Test setting text directly."""
        buffer = get_text_buffer()
        
        buffer.set_text("HELLO")
        
        assert buffer.get_text() == "HELLO"


class TestTranslationEndpoints:
    """Test translation-related API endpoints."""
    
    def setup_method(self):
        reset_translation_state()
    
    def test_get_translation(self):
        """Test GET /api/translate returns buffer."""
        response = client.post("/api/translate")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "text" in data
        assert "length" in data
    
    def test_clear_translation(self):
        """Test POST /api/translate/clear."""
        # Add some text first
        buffer = get_text_buffer()
        buffer.append("TEST")
        
        response = client.post("/api/translate/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify cleared
        response = client.post("/api/translate")
        data = response.json()
        assert data["text"] == ""
    
    def test_reset_stabilizer(self):
        """Test POST /api/translate/reset."""
        response = client.post("/api/translate/reset")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_get_translation_state(self):
        """Test GET /api/translate/state."""
        response = client.get("/api/translate/state")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "buffer" in data
        assert "stabilizer" in data

    def test_insert_space_endpoint(self):
        """Test POST /api/translate/space."""
        buffer = get_text_buffer()
        buffer.append("A")
        response = client.post("/api/translate/space")
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "A "
        assert data["length"] == 2

    def test_delete_char_endpoint(self):
        """Test POST /api/translate/delete."""
        buffer = get_text_buffer()
        buffer.append("A")
        buffer.append("B")
        response = client.post("/api/translate/delete")
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "A"
        assert data["length"] == 1

    def test_predict_returns_authoritative_buffer(self):
        """Test /api/predict returns authoritative text_buffer in response."""
        buffer = get_text_buffer()
        buffer.clear()
        buffer.append("HELLO")
        
        test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', test_img)
        b64 = base64.b64encode(buf).decode('utf-8')
        
        response = client.post("/api/predict", json={
            "image": b64,
            "use_stabilizer": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "text_buffer" in data
        assert data["text_buffer"] == "HELLO"


class TestPreprocessingEndpoint:
    """Test preprocessing configuration endpoint."""
    
    def test_preprocessing_config(self):
        """Test GET /api/preprocessing returns correct config."""
        response = client.get("/api/preprocessing")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "input_size" in data
        assert "normalization" in data
        assert "format" in data
        assert data["input_size"] == [128, 128]
        assert "mean" in data["normalization"]
        assert "std" in data["normalization"]
        assert data["format"] == "RGB"


class TestInferenceLatency:
    """Test and measure inference latency."""
    
    def create_test_image(self) -> str:
        """Create a simple test image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def test_inference_latency(self):
        """Test inference completes in reasonable time."""
        test_img = self.create_test_image()
        
        import time
        start = time.time()
        response = client.post("/api/predict", json={
            "image": test_img,
            "use_stabilizer": False
        })
        elapsed = time.time() - start
        
        assert response.status_code == 200
        data = response.json()
        
        # Should complete within reasonable time (allow generous margin for CI)
        assert elapsed < 5.0  # 5 seconds max
        assert data["total_time_ms"] > 0
        assert data["inference_time_ms"] >= 0
        
        print(f"Inference latency: {data['total_time_ms']:.1f}ms total, "
              f"{data['inference_time_ms']:.1f}ms inference")


class TestRobustStabilizationAndPreprocessing:
    """Tests for spurious predictions, no-hand transitions, held signs, and handedness."""

    def setup_method(self):
        reset_translation_state()

    def test_high_confidence_spurious_predictions_rejected_if_inconsistent(self):
        """High-confidence predictions (e.g. Z, N) fluctuating must not append."""
        stabilizer = get_stabilizer(min_consistent=3, confidence_threshold=0.7)
        buffer = get_text_buffer()
        buffer.clear()

        # Fluctuation between Z and N with high confidence (0.95)
        sequence = ["Z", "N", "Z", "N", "Z", "N"]
        for p in sequence:
            res = stabilizer.process(p, 0.95)
            assert res.action == ActionType.NONE
            assert not res.is_stable

        assert buffer.get_text() == ""

    def test_repeated_intentional_letters_via_no_hand_release(self):
        """Verify that signing 'Z', releasing hand (no-hand), and signing 'Z' again appends 'ZZ'."""
        stabilizer = get_stabilizer(min_consistent=3, confidence_threshold=0.7, cooldown_seconds=0.0)
        buffer = get_text_buffer()
        buffer.clear()

        # First sign 'Z' (3 frames >= 0.70)
        for _ in range(3):
            res = stabilizer.process("Z", 0.92)
        assert res.action == ActionType.APPEND
        assert res.character == "Z"
        buffer.append(res.character)
        assert buffer.get_text() == "Z"

        # Holding 'Z' does not append duplicate
        for _ in range(5):
            res = stabilizer.process("Z", 0.92)
            assert res.action == ActionType.NONE

        assert buffer.get_text() == "Z"

        # Hand leaves view (2 no-hand frames)
        stabilizer.process_no_hand()
        stabilizer.process_no_hand()

        # Present 'Z' again (3 frames >= 0.70)
        for _ in range(3):
            res = stabilizer.process("Z", 0.92)
        assert res.action == ActionType.APPEND
        assert res.character == "Z"
        buffer.append(res.character)

        assert buffer.get_text() == "ZZ"

    def test_left_hand_handedness_normalization(self):
        """Verify preprocess_for_inference with flip_horizontal normalizes orientation."""
        detector = get_hand_detector_stream()
        if detector is None:
            pytest.skip("Hand detector not available")

        # Create asymmetric crop: left half bright (255), right half dark (0)
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:, :50] = 255

        # With flip_horizontal=False: left half is bright
        tensor_no_flip, meta_no_flip = detector.preprocess_for_inference(crop, flip_horizontal=False)
        assert meta_no_flip["flipped_horizontal"] is False

        # With flip_horizontal=True: right half becomes bright
        tensor_flip, meta_flip = detector.preprocess_for_inference(crop, flip_horizontal=True)
        assert meta_flip["flipped_horizontal"] is True

        # Ensure tensors are actually horizontally inverted
        assert not torch.allclose(tensor_no_flip, tensor_flip)
        # Flip tensor_no_flip horizontally and compare
        tensor_flipped_back = torch.flip(tensor_no_flip, dims=[3])
        assert torch.allclose(tensor_flip, tensor_flipped_back)

    def test_crop_at_frame_boundary_preserves_square(self):
        """Verify cropping near or across frame boundary pads with zeros to maintain 1:1 aspect ratio."""
        detector = get_hand_detector_stream()
        if detector is None:
            pytest.skip("Hand detector not available")

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Bbox extends outside frame boundaries (y_min < 0)
        bbox = {"x_min": 100, "y_min": -50, "x_max": 250, "y_max": 100}
        crop = detector.crop_hand(frame, bbox)
        assert crop is not None
        assert crop.shape[0] == crop.shape[1], f"Expected square shape, got {crop.shape}"
        assert crop.shape[0] == 150

    def test_temporal_smoothing_damps_micro_jitter(self):
        """Verify temporal smoother dampens micro-jitter (displacement <= d_min)."""
        smoother = BoundingBoxSmoother(min_cutoff=0.5, max_cutoff=1.0, d_min=4.0, d_max=35.0)
        
        # Initial frame (cx=100.0, cy=100.0, side=150.0)
        cx0, cy0, s0, smoothed0 = smoother.smooth(100.0, 100.0, 150.0, "Right")
        assert not smoothed0
        assert cx0 == 100.0
        assert cy0 == 100.0

        # Frame with micro-jitter (+2px shift, dist = 2.82px <= d_min)
        cx1, cy1, s1, smoothed1 = smoother.smooth(102.0, 102.0, 150.0, "Right")
        assert smoothed1
        # With min_cutoff=0.5, new cx is 0.5 * 102 + 0.5 * 100 = 101.0
        assert abs(cx1 - 101.0) < 1e-4
        assert abs(cy1 - 101.0) < 1e-4

    def test_temporal_smoothing_adapts_to_rapid_motion(self):
        """Verify temporal smoother adapts alpha to 1.0 during rapid motion without lag."""
        smoother = BoundingBoxSmoother(min_cutoff=0.5, max_cutoff=1.0, d_min=4.0, d_max=35.0)
        smoother.smooth(100.0, 100.0, 150.0, "Right")

        # Fast motion: +40px shift (dist = 40.0px > d_max=35.0)
        cx1, cy1, s1, smoothed1 = smoother.smooth(140.0, 100.0, 150.0, "Right")
        assert smoothed1
        # Rapid motion applies max_cutoff = 1.0 (no lag)
        assert abs(cx1 - 140.0) < 1e-4

    def test_temporal_smoothing_reset_on_tracking_loss(self):
        """Verify smoother resets tracking history on explicit reset."""
        smoother = BoundingBoxSmoother()
        smoother.smooth(100.0, 100.0, 150.0, "Right")
        assert smoother.is_active

        smoother.reset()
        assert not smoother.is_active
        assert smoother.prev_cx is None

        # Next detection initializes anew
        cx, cy, s, smoothed = smoother.smooth(200.0, 200.0, 160.0, "Right")
        assert not smoothed
        assert cx == 200.0

    def test_temporal_smoothing_reset_on_handedness_change(self):
        """Verify smoother resets immediately when detected handedness changes."""
        smoother = BoundingBoxSmoother()
        smoother.smooth(100.0, 100.0, 150.0, "Right")

        # Handedness switches to Left -> snaps immediately without blending across hands
        cx, cy, s, smoothed = smoother.smooth(102.0, 102.0, 150.0, "Left")
        assert not smoothed
        assert cx == 102.0
        assert cy == 102.0

    def test_crop_aspect_ratio_landscape_and_portrait(self):
        """Verify square crop generation on landscape (640x480) and portrait (480x640) frames."""
        detector = get_hand_detector_stream()
        if detector is None:
            pytest.skip("Hand detector not available")

        # Landscape frame
        frame_landscape = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox_l = {"x_min": 100, "y_min": 50, "x_max": 250, "y_max": 200}
        crop_l = detector.crop_hand(frame_landscape, bbox_l)
        assert crop_l is not None
        assert crop_l.shape[0] == crop_l.shape[1], f"Expected square crop on landscape, got {crop_l.shape}"

        # Portrait frame
        frame_portrait = np.zeros((640, 480, 3), dtype=np.uint8)
        bbox_p = {"x_min": 50, "y_min": 100, "x_max": 200, "y_max": 300}
        crop_p = detector.crop_hand(frame_portrait, bbox_p)
        assert crop_p is not None
        assert crop_p.shape[0] == crop_p.shape[1], f"Expected square crop on portrait, got {crop_p.shape}"

    def test_crop_all_four_boundaries_padded_safely(self):
        """Verify crops exceeding top, bottom, left, or right boundaries pad safely with zeros."""
        detector = get_hand_detector_stream()
        if detector is None:
            pytest.skip("Hand detector not available")

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        h, w = frame.shape[:2]

        boundary_boxes = [
            {"x_min": 100, "y_min": -40, "x_max": 220, "y_max": 80},     # Top out-of-bounds
            {"x_min": -50, "y_min": 100, "x_max": 70, "y_max": 220},     # Left out-of-bounds
            {"x_min": 100, "y_min": 400, "x_max": 220, "y_max": 520},    # Bottom out-of-bounds
            {"x_min": 580, "y_min": 100, "x_max": 700, "y_max": 220},    # Right out-of-bounds
        ]

        for bbox in boundary_boxes:
            crop = detector.crop_hand(frame, bbox)
            assert crop is not None
            assert crop.shape[0] == crop.shape[1], f"Expected square crop for boundary box {bbox}, got {crop.shape}"
            assert detector.was_last_crop_padded() is True

    def test_predict_endpoint_returns_tracking_diagnostics(self):
        """Verify /api/predict response schema includes tracking_diagnostics."""
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_img)
        test_img = base64.b64encode(buffer).decode('utf-8')

        response = client.post("/api/predict", json={
            "image": test_img,
            "use_stabilizer": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "tracking_diagnostics" in data
        diag = data["tracking_diagnostics"]
        assert diag is not None
        assert "landmarks_count" in diag
        assert "tracking_confidence" in diag
        assert "is_smoothed" in diag
        assert "crop_padding_applied" in diag


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
