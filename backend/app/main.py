"""
FastAPI application for ASL Sign Language Recognition.
Provides REST API for health checks, class listing, and prediction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import base64
import io
import time
import json
import logging
import threading
import numpy as np
import cv2
import torch
from PIL import Image
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.model_loader import (
    get_model_loader, initialize_model, cleanup_model
)
from backend.app.services.hand_detector import (
    get_hand_detector, get_hand_detector_stream, HandDetector
)
from backend.app.services.stabilizer import (
    get_stabilizer, get_text_buffer, reset_translation_state,
    StabilizedPrediction, TextBuffer, PredictionStabilizer
)


app = FastAPI(
    title="ASL Sign Language Recognition API",
    description="Real-time hand gesture recognition for ASL alphabet",
    version="1.0.0"
)
logger = logging.getLogger("asl.diagnostics")

# Keep append evidence across backend restarts.  The console remains useful for
# live debugging, while this JSONL file is the authoritative history used to
# investigate physical-webcam sessions.  No frame pixels are persisted.
# This module lives in <workspace>/backend/app; keep diagnostics in the
# workspace even though legacy model lookup uses its existing root constant.
APPEND_DIAGNOSTIC_LOG = Path(__file__).resolve().parents[2] / "results" / "logs" / "append_diagnostics.jsonl"
if not any(getattr(handler, "baseFilename", None) == str(APPEND_DIAGNOSTIC_LOG) for handler in logger.handlers):
    APPEND_DIAGNOSTIC_LOG.parent.mkdir(parents=True, exist_ok=True)
    _append_diagnostic_handler = logging.FileHandler(APPEND_DIAGNOSTIC_LOG, encoding="utf-8")
    _append_diagnostic_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_append_diagnostic_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
DEBUG_CROP_DIR = APPEND_DIAGNOSTIC_LOG.parent / "debug_crops"
DEBUG_CROP_LIMIT = 12
_debug_crop_lock = threading.Lock()
_debug_crop_count = 0

# CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ClassInfo(BaseModel):
    index: int
    label: str


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    gpu_available: bool
    gpu_name: Optional[str] = None
    model_load_time_seconds: float
    checkpoint_name: Optional[str] = None
    checkpoint_type: Optional[str] = None


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image (JPEG/PNG)")
    use_stabilizer: bool = Field(True, description="Apply prediction stabilization")
    return_top_k: int = Field(3, ge=1, le=5, description="Number of top predictions to return")
    capture_debug_crop: bool = Field(False, description="Save up to 12 exact inference crops for this backend run")
    enable_osn: bool = Field(False, description="Apply landmark-guided Orientation and Scale Normalization (OSN)")


class PredictionItem(BaseModel):
    rank: int
    class_index: int
    class_label: str
    confidence: float


class StabilizedPredictionResponse(BaseModel):
    action: str  # "append", "delete", "space", "clear", "none"
    character: str
    confidence: float
    is_stable: bool
    raw_prediction: str
    raw_confidence: float
    status_reason: Optional[str] = None


class TrackingDiagnostics(BaseModel):
    landmarks_count: int = 0
    tracking_confidence: float = 0.0
    is_smoothed: bool = False
    crop_padding_applied: bool = False
    crop_fully_within_frame: bool = False
    crop_touches_frame_edge: bool = False
    crop_bounds: Optional[Dict[str, int]] = None
    frame_size: Optional[Dict[str, int]] = None
    handedness_normalized: bool = False
    osn_applied: bool = False
    osn_rotation_deg: float = 0.0
    osn_hand_coverage_pct: float = 0.0


class PredictResponse(BaseModel):
    success: bool
    predictions: List[PredictionItem]
    stabilized: Optional[StabilizedPredictionResponse] = None
    hand_detected: bool
    hand_bbox: Optional[Dict[str, int]] = None
    hand_handedness: Optional[str] = None
    tracking_diagnostics: Optional[TrackingDiagnostics] = None
    inference_time_ms: float
    preprocessing_time_ms: float
    total_time_ms: float
    text_buffer: Optional[str] = None
    debug_crop_path: Optional[str] = None


class TextBufferResponse(BaseModel):
    text: str
    length: int


class StabilizerConfigRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=0.99, description="Confidence threshold (0.1 - 0.99)")
    min_consistent: Optional[int] = Field(None, ge=1, le=10, description="Minimum consecutive consistent frames (1 - 10)")
    cooldown_seconds: Optional[float] = Field(None, ge=0.0, le=5.0, description="Cooldown between different letters (seconds)")
    duplicate_guard: Optional[bool] = Field(None, description="Prevent duplicate append while holding same gesture")


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("Starting ASL Recognition API...")
    success = initialize_model()
    if not success:
        print("WARNING: Model failed to load!")
    else:
        print("Model loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    cleanup_model()
    print("API shutdown complete")


# Dependency to get model loader
def get_model() -> object:
    loader = get_model_loader()
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return loader


# Dependency to get hand detector
def get_detector() -> Optional[HandDetector]:
    return get_hand_detector_stream()


# Dependency to get stabilizer
def get_stabilizer_service() -> PredictionStabilizer:
    return get_stabilizer()


def get_buffer() -> TextBuffer:
    return get_text_buffer()


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def save_debug_crop(
    crop: np.ndarray,
    timestamp_ns: int,
    top_predictions: List[Dict],
    detection: Dict,
    padding_applied: bool,
) -> Optional[str]:
    """Persist a bounded sample of the unchanged BGR crop supplied to preprocessing."""
    global _debug_crop_count
    with _debug_crop_lock:
        if _debug_crop_count >= DEBUG_CROP_LIMIT:
            return None
        DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"crop_{timestamp_ns}_{_debug_crop_count:02d}.jpg"
        image_path = DEBUG_CROP_DIR / filename
        if not cv2.imwrite(str(image_path), crop):
            return None
        metadata = {
            "timestamp_ns": timestamp_ns,
            "image": filename,
            "top_predictions": top_predictions,
            "handedness": detection.get("handedness"),
            "crop_bounds": detection.get("bbox"),
            "raw_crop_bounds": detection.get("raw_bbox"),
            "frame_size": detection.get("frame_size"),
            "crop_fully_within_frame": detection.get("crop_fully_within_frame"),
            "padding_applied": padding_applied,
        }
        with (DEBUG_CROP_DIR / "metadata.jsonl").open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(metadata, sort_keys=True) + "\n")
        _debug_crop_count += 1
        return str(image_path.relative_to(Path(__file__).resolve().parents[2]))


# API Routes
@app.get("/api/health", response_model=HealthResponse)
async def health_check(model=Depends(get_model)):
    """Health check endpoint."""
    info = model.get_info()
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy" if info["loaded"] else "degraded",
        model_loaded=info["loaded"],
        device=info["device"],
        num_classes=info["num_classes"],
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        model_load_time_seconds=info["load_time_seconds"],
        checkpoint_name=info.get("checkpoint_name"),
        checkpoint_type=info.get("checkpoint_type")
    )


@app.get("/api/classes", response_model=ClassesResponse)
async def get_classes(model=Depends(get_model)):
    """Get list of all 29 classes in correct order."""
    info = model.get_info()
    classes = info.get("classes", [])
    return ClassesResponse(
        classes=[ClassInfo(index=i, label=c) for i, c in enumerate(classes)],
        count=len(classes)
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    model=Depends(get_model),
    detector=Depends(get_detector),
    stabilizer=Depends(get_stabilizer_service),
    buffer=Depends(get_buffer)
):
    """Predict ASL sign from uploaded image."""
    total_start = time.time()
    
    try:
        # Decode image
        preprocess_start = time.time()
        frame = decode_base64_image(request.image)
        preprocess_time = (time.time() - preprocess_start) * 1000
        
        # Detect hands if detector available
        detections = []
        if detector is not None:
            detections = detector.detect(frame)
        
        if not detections:
            if request.use_stabilizer:
                stabilizer.process_no_hand()
            if detector is not None:
                detector.reset_tracking()
            return PredictResponse(
                success=True,
                predictions=[],
                stabilized=None,
                hand_detected=False,
                hand_bbox=None,
                hand_handedness=None,
                tracking_diagnostics=TrackingDiagnostics(
                    landmarks_count=0,
                    tracking_confidence=0.0,
                    is_smoothed=False,
                    crop_padding_applied=False
                ),
                inference_time_ms=0,
                preprocessing_time_ms=round(preprocess_time, 2),
                total_time_ms=round((time.time() - total_start) * 1000, 2),
                text_buffer=buffer.get_text()
            )
        
        # Use first detected hand (highest confidence)
        best_detection = max(detections, key=lambda d: d["confidence"])
        
        # Crop hand (supports optional landmark-guided OSN)
        hand_crop = detector.crop_hand(
            frame,
            best_detection["bbox"],
            landmarks=best_detection.get("landmarks"),
            use_osn=request.enable_osn
        )
        if hand_crop is None:
            if request.use_stabilizer:
                stabilizer.process_no_hand()
            if detector is not None:
                detector.reset_tracking()
            return PredictResponse(
                success=True,
                predictions=[],
                stabilized=None,
                hand_detected=False,
                hand_bbox=best_detection["bbox"],
                hand_handedness=best_detection["handedness"],
                tracking_diagnostics=TrackingDiagnostics(
                    landmarks_count=best_detection.get("landmarks_count", 0),
                    tracking_confidence=round(best_detection.get("confidence", 0.0), 3),
                    is_smoothed=best_detection.get("is_smoothed", False),
                    crop_padding_applied=False
                ),
                inference_time_ms=0,
                preprocessing_time_ms=round(preprocess_time, 2),
                total_time_ms=round((time.time() - total_start) * 1000, 2),
                text_buffer=buffer.get_text()
            )
        
        # Check handedness normalization:
        # The CNN was trained exclusively on right hands (100% in ASL Alphabet dataset).
        # When MediaPipe detects an anatomical Left hand, horizontally flipping the crop maps
        # it directly into the right-hand feature space, restoring accuracy across asymmetric signs.
        hand_handedness = best_detection.get("handedness")
        normalize_handedness = (hand_handedness == "Left")

        # Preprocess for inference
        preprocess_config = model.get_preprocessing_config()
        tensor, meta = detector.preprocess_for_inference(
            hand_crop,
            target_size=tuple(preprocess_config["input_size"]),
            normalization_mean=preprocess_config["normalization"]["mean"],
            normalization_std=preprocess_config["normalization"]["std"],
            flip_horizontal=normalize_handedness
        )
        
        if tensor is None:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {meta.get('error', 'Unknown error')}")
        
        # Run inference
        inference_start = time.time()
        predictions, logits = model.predict(tensor)
        inference_time = (time.time() - inference_start) * 1000
        
        # Format top predictions
        top_predictions = predictions[0][:request.return_top_k]
        pred_items = [
            PredictionItem(
                rank=p["rank"],
                class_index=p["class_index"],
                class_label=p["class_label"],
                confidence=p["confidence"]
            )
            for p in top_predictions
        ]
        debug_crop_path = None
        if request.capture_debug_crop:
            debug_crop_path = save_debug_crop(
                hand_crop,
                time.time_ns(),
                top_predictions,
                best_detection,
                detector.was_last_crop_padded() if detector is not None else False,
            )
        
        # Apply stabilization if requested
        stabilized_response = None
        if request.use_stabilizer and top_predictions:
            top_pred = top_predictions[0]
            state_before = stabilizer.get_state()
            crop_diagnostics = detector.get_last_crop_diagnostics() if detector is not None else {}
            frame_metadata = {
                "hand_detected": True,
                "handedness": best_detection.get("handedness"),
                "landmarks_count": best_detection.get("landmarks_count", 0),
                "tracking_confidence": best_detection.get("confidence", 0.0),
                "crop_bounds": best_detection.get("bbox"),
                "raw_crop_bounds": best_detection.get("raw_bbox"),
                "crop_fully_within_frame": best_detection.get("crop_fully_within_frame", False),
                "crop_touches_frame_edge": (
                    best_detection["bbox"]["x_min"] <= 0
                    or best_detection["bbox"]["y_min"] <= 0
                    or best_detection["bbox"]["x_max"] >= best_detection["frame_size"]["width"]
                    or best_detection["bbox"]["y_max"] >= best_detection["frame_size"]["height"]
                ),
                "crop_padding_applied": crop_diagnostics.get("padding_applied", False),
                "handedness_normalized": normalize_handedness,
            }
            stabilized = stabilizer.process(top_pred["class_label"], top_pred["confidence"], frame_metadata)

            if stabilized.is_stable and stabilized.action.value != "none":
                buffer_before = buffer.get_text()
                if stabilized.action.value == "append":
                    buffer.append(stabilized.character)
                elif stabilized.action.value == "space":
                    buffer.space()
                elif stabilized.action.value == "delete":
                    buffer.delete()
                elif stabilized.action.value == "clear":
                    buffer.clear()
                # One JSON object per mutation makes the exact three contributing
                # frames auditable; it intentionally includes no image data.
                logger.info(json.dumps({
                    "event": "APPEND_DIAGNOSTIC",
                    "timestamp": time.time(), "predicted_class": top_pred["class_label"],
                    "contributing_frames": list(stabilizer.frame_window)[-stabilizer.min_consistent:],
                    "stabilized_class": stabilized.raw_prediction, "stabilized_confidence": stabilized.confidence,
                    "action": stabilized.action.value, "cooldown_release_before": {
                        "last_appended": state_before["last_appended"], "last_action_time": state_before.get("last_action_time", 0),
                        "no_hand_count": state_before["no_hand_count"], "held_char_count": state_before["held_char_count"]},
                    "cooldown_release_after": {"last_appended": stabilizer.last_appended, "last_action_time": stabilizer.last_action_time,
                        "no_hand_count": stabilizer.no_hand_count, "held_char_count": stabilizer.held_char_count},
                    "buffer_before": buffer_before, "buffer_after": buffer.get_text(),
                }, default=str, sort_keys=True))
            
            stabilized_response = StabilizedPredictionResponse(
                action=stabilized.action.value,
                character=stabilized.character,
                confidence=stabilized.confidence,
                is_stable=stabilized.is_stable,
                raw_prediction=stabilized.raw_prediction,
                raw_confidence=stabilized.raw_confidence,
                status_reason=getattr(stabilized, "status_reason", "none")
            )
        elif request.use_stabilizer and not top_predictions:
            stabilized = stabilizer.process_no_hand()
            stabilized_response = StabilizedPredictionResponse(
                action=stabilized.action.value,
                character=stabilized.character,
                confidence=stabilized.confidence,
                is_stable=stabilized.is_stable,
                raw_prediction=stabilized.raw_prediction,
                raw_confidence=stabilized.raw_confidence,
                status_reason=getattr(stabilized, "status_reason", "no_hand")
            )
        
        total_time = (time.time() - total_start) * 1000
        
        osn_diag = detector.get_last_osn_diagnostics() if detector is not None else {}
        tracking_diag = TrackingDiagnostics(
            landmarks_count=best_detection.get("landmarks_count", 21),
            tracking_confidence=round(best_detection.get("confidence", 0.0), 3),
            is_smoothed=best_detection.get("is_smoothed", False),
            crop_padding_applied=detector.was_last_crop_padded() if detector is not None else False,
            crop_fully_within_frame=best_detection.get("crop_fully_within_frame", False),
            crop_touches_frame_edge=(
                best_detection["bbox"]["x_min"] <= 0
                or best_detection["bbox"]["y_min"] <= 0
                or best_detection["bbox"]["x_max"] >= best_detection["frame_size"]["width"]
                or best_detection["bbox"]["y_max"] >= best_detection["frame_size"]["height"]
            ),
            crop_bounds=best_detection.get("bbox"),
            frame_size=best_detection.get("frame_size"),
            handedness_normalized=normalize_handedness,
            osn_applied=osn_diag.get("osn_applied", False),
            osn_rotation_deg=osn_diag.get("rotation_deg", 0.0),
            osn_hand_coverage_pct=osn_diag.get("hand_coverage_pct", 0.0)
        )
        
        return PredictResponse(
            success=True,
            predictions=pred_items,
            stabilized=stabilized_response,
            hand_detected=True,
            hand_bbox=best_detection["bbox"],
            hand_handedness=best_detection["handedness"],
            tracking_diagnostics=tracking_diag,
            inference_time_ms=round(inference_time, 2),
            preprocessing_time_ms=round(preprocess_time, 2),
            total_time_ms=round(total_time, 2),
            text_buffer=buffer.get_text(),
            debug_crop_path=debug_crop_path,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/translate", response_model=TextBufferResponse)
async def translate(
    stabilizer=Depends(get_stabilizer_service),
    buffer=Depends(get_buffer)
):
    """Get current translated text buffer."""
    return TextBufferResponse(**buffer.get_state())


@app.post("/api/translate/clear")
async def clear_translation(
    stabilizer=Depends(get_stabilizer_service),
    buffer=Depends(get_buffer)
):
    """Clear translation buffer."""
    stabilizer.force_clear()
    buffer.clear()
    return {"success": True, "message": "Translation cleared"}


@app.post("/api/translate/space", response_model=TextBufferResponse)
async def insert_space(buffer=Depends(get_buffer)):
    """Add a space to the translation buffer."""
    buffer.space()
    return TextBufferResponse(**buffer.get_state())


@app.post("/api/translate/delete", response_model=TextBufferResponse)
async def delete_character(buffer=Depends(get_buffer)):
    """Delete the last character from the translation buffer."""
    buffer.delete()
    return TextBufferResponse(**buffer.get_state())


@app.post("/api/translate/reset")
async def reset_translation(
    stabilizer=Depends(get_stabilizer_service),
    buffer=Depends(get_buffer),
    detector=Depends(get_detector)
):
    """Reset translation and hand tracking state."""
    stabilizer.reset()
    if detector is not None:
        detector.reset_tracking()
    return {"success": True, "message": "Stabilizer and tracking reset"}


@app.get("/api/translate/state")
async def get_translation_state(
    stabilizer=Depends(get_stabilizer_service),
    buffer=Depends(get_buffer)
):
    """Get current translation state for debugging."""
    return {
        "buffer": buffer.get_state(),
        "stabilizer": stabilizer.get_state()
    }


@app.get("/api/preprocessing")
async def get_preprocessing_config(model=Depends(get_model)):
    """Get preprocessing configuration used by the model."""
    return model.get_preprocessing_config()


@app.get("/api/stabilizer/config")
async def get_stabilizer_config(stabilizer=Depends(get_stabilizer_service)):
    """Get current prediction stabilizer configuration and calibration rationale."""
    return stabilizer.get_config()


@app.post("/api/stabilizer/config")
async def update_stabilizer_config(
    request: StabilizerConfigRequest,
    stabilizer=Depends(get_stabilizer_service)
):
    """Update prediction stabilizer configuration."""
    stabilizer.update_config(
        confidence_threshold=request.confidence_threshold,
        min_consistent=request.min_consistent,
        cooldown_seconds=request.cooldown_seconds,
        duplicate_guard=request.duplicate_guard,
    )
    return {"success": True, "config": stabilizer.get_config()}


@app.get("/api/samples")
async def get_verified_samples():
    """
    Returns list of verified physical webcam sample gestures for reliable demo mode.
    Includes base64 images of priority demo letters (A, B, C, E, H, L, P) and others.
    """
    samples_dir = Path(__file__).resolve().parents[2] / "data" / "webcam_augmented" / "val"
    fallback_dir = Path(__file__).resolve().parents[2] / "data" / "webcam_captures"
    search_dir = samples_dir if samples_dir.exists() else fallback_dir

    demo_letters = ["A", "B", "C", "E", "H", "L", "P"]
    result = []

    if search_dir.exists():
        for letter in sorted(list(set(demo_letters + [d.name for d in search_dir.iterdir() if d.is_dir()]))):
            cls_dir = search_dir / letter
            if not cls_dir.is_dir():
                continue
            jpgs = sorted(list(cls_dir.glob("*.jpg")))
            if not jpgs:
                continue
            sample_path = jpgs[0]
            try:
                with open(sample_path, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode("utf-8")
                result.append({
                    "class_label": letter,
                    "filename": sample_path.name,
                    "is_priority_demo": (letter in demo_letters),
                    "image_base64": b64,
                })
            except Exception:
                continue

    return {"samples": result, "count": len(result)}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
