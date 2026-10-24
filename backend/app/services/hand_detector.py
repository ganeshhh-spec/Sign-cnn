"""
Hand detection and preprocessing service using MediaPipe 1.x Tasks API.
Locates hand, crops region, and applies training-compatible preprocessing.
"""

import cv2
import numpy as np
import torch
import math
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class BoundingBoxSmoother:
    """
    Adaptive temporal smoothing for hand bounding box coordinates.
    Filters micro-jitter from webcam sensor noise while adapting dynamically to fast motion.
    
    Attributes:
        min_cutoff: Smoothing factor when hand is stationary (0.0=frozen, 1.0=raw).
        max_cutoff: Smoothing factor when hand is moving rapidly.
        d_min: Pixel displacement threshold below which maximum smoothing is applied.
        d_max: Pixel displacement threshold above which minimal smoothing is applied.
        max_jump_distance: Maximum plausible displacement before resetting tracking.
    """
    def __init__(
        self,
        min_cutoff: float = 0.5,
        max_cutoff: float = 1.0,
        d_min: float = 4.0,
        d_max: float = 35.0,
        max_jump_distance: float = 180.0
    ):
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.d_min = d_min
        self.d_max = d_max
        self.max_jump_distance = max_jump_distance
        
        self.prev_cx: Optional[float] = None
        self.prev_cy: Optional[float] = None
        self.prev_side: Optional[float] = None
        self.prev_handedness: Optional[str] = None
        self.is_active: bool = False

    def reset(self):
        """Reset smoother state on tracking loss, scene change, or manual reset."""
        self.prev_cx = None
        self.prev_cy = None
        self.prev_side = None
        self.prev_handedness = None
        self.is_active = False

    def smooth(
        self,
        raw_cx: float,
        raw_cy: float,
        raw_side: float,
        handedness: Optional[str] = None
    ) -> Tuple[float, float, float, bool]:
        """
        Apply adaptive temporal smoothing to raw box parameters.
        
        Returns:
            Tuple of (smoothed_cx, smoothed_cy, smoothed_side, is_smoothed)
        """
        # First frame or handedness switch: initialize directly without smoothing
        if self.prev_cx is None or self.prev_handedness != handedness:
            self.prev_cx = float(raw_cx)
            self.prev_cy = float(raw_cy)
            self.prev_side = float(raw_side)
            self.prev_handedness = handedness
            self.is_active = True
            return float(raw_cx), float(raw_cy), float(raw_side), False

        # Measure displacement
        dx = raw_cx - self.prev_cx
        dy = raw_cy - self.prev_cy
        dist = (dx * dx + dy * dy) ** 0.5

        # Discontinuous jump (different hand or reacquisition) -> snap immediately
        if dist > self.max_jump_distance:
            self.prev_cx = float(raw_cx)
            self.prev_cy = float(raw_cy)
            self.prev_side = float(raw_side)
            self.prev_handedness = handedness
            self.is_active = True
            return float(raw_cx), float(raw_cy), float(raw_side), False

        # Adaptive alpha based on speed
        if dist <= self.d_min:
            alpha = self.min_cutoff
        elif dist >= self.d_max:
            alpha = self.max_cutoff
        else:
            t = (dist - self.d_min) / (self.d_max - self.d_min)
            alpha = self.min_cutoff + t * (self.max_cutoff - self.min_cutoff)

        smoothed_cx = alpha * raw_cx + (1.0 - alpha) * self.prev_cx
        smoothed_cy = alpha * raw_cy + (1.0 - alpha) * self.prev_cy
        smoothed_side = alpha * raw_side + (1.0 - alpha) * self.prev_side

        self.prev_cx = smoothed_cx
        self.prev_cy = smoothed_cy
        self.prev_side = smoothed_side
        self.is_active = True

        return smoothed_cx, smoothed_cy, smoothed_side, True


class HandDetector:
    """MediaPipe Tasks API-based hand detection and cropping with temporal smoothing."""
    
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        expand_ratio: float = 1.20,
        enable_smoothing: bool = True,
        enable_osn: bool = False,
        osn_scale_factor: float = 2.0,
        osn_wrist_y_ratio: float = 0.82,
        osn_max_rotation_deg: float = 45.0
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available")
        
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.expand_ratio = expand_ratio
        self.enable_smoothing = enable_smoothing
        self.smoother = BoundingBoxSmoother()
        self.last_crop_padded = False
        self.last_crop_diagnostics: Dict = {}
        self.last_osn_diagnostics: Dict = {}
        self._lock = threading.RLock()

        # OSN Configuration (optional landmark-guided orientation and scale normalization)
        env_osn = os.environ.get("ASL_ENABLE_OSN", "").strip().lower() in ("1", "true", "yes")
        self.enable_osn = enable_osn or env_osn
        self.osn_scale_factor = osn_scale_factor
        self.osn_wrist_y_ratio = osn_wrist_y_ratio
        self.osn_max_rotation_deg = osn_max_rotation_deg
        
        # Check model file exists
        model_path = self._get_model_path()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {model_path}. "
                f"Download from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        # Create hand landmarker options
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
    
    def _get_model_path(self) -> str:
        """Get path to MediaPipe hand landmarker model."""
        # Try to find the model in mediapipe package
        import mediapipe as mp
        mp_path = Path(mp.__file__).parent
        model_path = mp_path / "models" / "hand_landmarker.task"
        
        if model_path.exists():
            return str(model_path)
        
        # Check common locations
        for base in [Path.cwd(), PROJECT_ROOT, Path.home()]:
            candidate = base / "hand_landmarker.task"
            if candidate.exists():
                return str(candidate)
        
        # Return default name - will fail if not found
        return "hand_landmarker.task"
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hands in frame and apply temporal smoothing to bounding boxes.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detection dicts with landmarks, bbox, raw_bbox, handedness, is_smoothed
        """
        with self._lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect
            results = self.landmarker.detect(mp_image)
            
            detections = []
            if not results.hand_landmarks:
                self.smoother.reset()
                return detections

            for idx, (landmarks, handedness) in enumerate(zip(
                results.hand_landmarks, results.handedness
            )):
                h, w = frame.shape[:2]
                
                # Get bounding box from landmarks in pixel space
                x_coords_px = [lm.x * w for lm in landmarks]
                y_coords_px = [lm.y * h for lm in landmarks]
                
                x_min_raw, x_max_raw = min(x_coords_px), max(x_coords_px)
                y_min_raw, y_max_raw = min(y_coords_px), max(y_coords_px)
                
                box_w = max(1.0, x_max_raw - x_min_raw)
                box_h = max(1.0, y_max_raw - y_min_raw)
                
                # Make square to preserve 1:1 aspect ratio expected by the CNN
                raw_side = max(box_w, box_h) * self.expand_ratio
                raw_cx = (x_min_raw + x_max_raw) / 2
                raw_cy = (y_min_raw + y_max_raw) / 2
                
                hand_label = handedness[0].category_name if handedness else "Unknown"
                hand_conf = handedness[0].score if handedness else 0.0

                # Apply adaptive temporal smoothing
                is_smoothed = False
                if self.enable_smoothing:
                    cx, cy, side, is_smoothed = self.smoother.smooth(
                        raw_cx, raw_cy, raw_side, handedness=hand_label
                    )
                else:
                    cx, cy, side = raw_cx, raw_cy, raw_side
                
                # Keep the square crop in the captured frame whenever it can fit.
                # Translation preserves its size and every landmark; only a crop
                # larger than the frame itself must use padding.
                int_side = int(round(min(side, max(w, h))))
                if int_side <= w:
                    cx = min(max(cx, int_side / 2.0), w - int_side / 2.0)
                if int_side <= h:
                    cy = min(max(cy, int_side / 2.0), h - int_side / 2.0)

                # Smoothed, frame-contained-when-possible bounding box coordinates.
                # Anchoring max to min + int_side guarantees exact square dimensions (crop_w == crop_h),
                # preventing subpixel rounding jitter from falsely triggering zero-padding.
                x_min = int(round(cx - int_side / 2.0))
                x_max = x_min + int_side
                y_min = int(round(cy - int_side / 2.0))
                y_max = y_min + int_side

                # Raw unsmoothed box coordinates for diagnostic reference
                raw_int_side = int(round(raw_side))
                raw_x_min = int(round(raw_cx - raw_int_side / 2.0))
                raw_x_max = raw_x_min + raw_int_side
                raw_y_min = int(round(raw_cy - raw_int_side / 2.0))
                raw_y_max = raw_y_min + raw_int_side
                
                bbox = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }
                raw_bbox = {
                    "x_min": raw_x_min,
                    "y_min": raw_y_min,
                    "x_max": raw_x_max,
                    "y_max": raw_y_max
                }
                
                detections.append({
                    "landmarks": landmarks,
                    "bbox": bbox,
                    "raw_bbox": raw_bbox,
                    "handedness": hand_label,
                    "confidence": hand_conf,
                    "landmarks_count": len(landmarks),
                    "is_smoothed": is_smoothed,
                    "frame_size": {"width": w, "height": h},
                    "crop_fully_within_frame": x_min >= 0 and y_min >= 0 and x_max <= w and y_max <= h,
                    "index": idx
                })
        
            return detections
    
    def crop_hand(
        self,
        frame: np.ndarray,
        bbox: Dict,
        landmarks: Optional[List] = None,
        use_osn: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Crop hand region from frame, guaranteeing a square output.
        If use_osn (or self.enable_osn) is True and landmarks are provided,
        applies landmark-guided Orientation and Scale Normalization (OSN).
        Falls back seamlessly to the standard square bounding box crop if
        OSN is disabled, unavailable, or degenerate.
        
        Args:
            frame: BGR image
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
            landmarks: Optional list of MediaPipe landmarks
            use_osn: Optional override flag to enable/disable OSN for this frame
            
        Returns:
            Square cropped BGR image or None if invalid
        """
        should_use_osn = (use_osn if use_osn is not None else self.enable_osn)
        if should_use_osn and landmarks is not None:
            osn_crop, osn_diag = self.crop_hand_osn(frame, landmarks)
            if osn_crop is not None:
                self.last_crop_padded = False
                self.last_crop_diagnostics = {
                    "bbox": dict(bbox),
                    "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
                    "fully_within_frame": True,
                    "padding_applied": False,
                    "crop_size": osn_crop.shape[0],
                    "valid": True,
                    "osn_applied": True,
                    "osn_diagnostics": osn_diag,
                }
                return osn_crop

        x_min, y_min = bbox["x_min"], bbox["y_min"]
        x_max, y_max = bbox["x_max"], bbox["y_max"]
        
        # Validate bbox
        self.last_crop_diagnostics = {"bbox": dict(bbox), "valid": False, "osn_applied": False}
        if x_max <= x_min or y_max <= y_min:
            return None
        
        fh, fw = frame.shape[:2]
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        
        # Calculate overlap with frame
        src_x_min = max(0, x_min)
        src_x_max = min(fw, x_max)
        src_y_min = max(0, y_min)
        src_y_max = min(fh, y_max)
        
        if src_x_max <= src_x_min or src_y_max <= src_y_min:
            return None
        
        src_crop = frame[src_y_min:src_y_max, src_x_min:src_x_max]
        
        # If fully inside frame and square, return directly
        if x_min >= 0 and y_min >= 0 and x_max <= fw and y_max <= fh and crop_w == crop_h:
            self.last_crop_padded = False
            self.last_crop_diagnostics = {
                "bbox": dict(bbox), "frame_size": {"width": fw, "height": fh},
                "fully_within_frame": True, "padding_applied": False,
                "crop_size": crop_w, "valid": True,
                "osn_applied": False,
            }
            return src_crop if src_crop.size > 0 else None
            
        # Target square size
        target_side = max(crop_w, crop_h)
        square_crop = np.zeros((target_side, target_side, frame.shape[2]), dtype=frame.dtype)
        
        dst_x_min = src_x_min - x_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_min = src_y_min - y_min
        dst_y_max = dst_y_min + (src_y_max - src_y_min)
        
        # Ensure within destination canvas
        dst_x_min = max(0, dst_x_min)
        dst_y_min = max(0, dst_y_min)
        dst_x_max = min(target_side, dst_x_max)
        dst_y_max = min(target_side, dst_y_max)
        
        slice_h = dst_y_max - dst_y_min
        slice_w = dst_x_max - dst_x_min
        if slice_h > 0 and slice_w > 0:
            square_crop[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = src_crop[:slice_h, :slice_w]
            self.last_crop_padded = True
            self.last_crop_diagnostics = {
                "bbox": dict(bbox), "frame_size": {"width": fw, "height": fh},
                "fully_within_frame": False, "padding_applied": True,
                "crop_size": target_side, "valid": True,
                "osn_applied": False,
            }
            return square_crop
            
        self.last_crop_padded = False
        return src_crop if src_crop.size > 0 else None

    def crop_hand_osn(
        self,
        frame: np.ndarray,
        landmarks,
        scale_factor: Optional[float] = None,
        wrist_y_ratio: Optional[float] = None,
        max_rotation_deg: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Landmark-guided Orientation and Scale Normalization (OSN).
        
        1. Computes hand tilt angle using wrist (landmark 0) and middle MCP (landmark 9).
        2. Clamps rotation angle to max_rotation_deg to avoid distorting naturally non-vertical signs.
        3. Rotates raw frame so palm axis is aligned upright.
        4. Scales and pads bounding box using detected hand landmarks (scale_factor ~ 2.0x, wrist at ~82% height).
        5. Returns normalized crop and metadata.
        Falls back to None if landmarks are degenerate or missing.
        """
        sf = scale_factor if scale_factor is not None else self.osn_scale_factor
        wyr = wrist_y_ratio if wrist_y_ratio is not None else self.osn_wrist_y_ratio
        max_rot = max_rotation_deg if max_rotation_deg is not None else self.osn_max_rotation_deg
        
        diag = {
            "osn_applied": False,
            "fallback_reason": None,
            "rotation_deg": 0.0,
            "scale_factor": sf,
            "wrist_y_ratio": wyr,
        }
        
        if landmarks is None or len(landmarks) < 21:
            diag["fallback_reason"] = "Insufficient landmarks (< 21)"
            self.last_osn_diagnostics = diag
            return None, diag
            
        fh, fw = frame.shape[:2]
        try:
            lm_pts = np.array([(lm.x * fw, lm.y * fh) for lm in landmarks], dtype=np.float32)
            wrist = lm_pts[0]
            middle_mcp = lm_pts[9]
            
            dx = float(middle_mcp[0] - wrist[0])
            dy = float(middle_mcp[1] - wrist[1])
            dist = (dx * dx + dy * dy) ** 0.5
            
            if dist < 5.0:
                diag["fallback_reason"] = f"Degenerate wrist-MCP distance ({dist:.1f}px < 5.0px)"
                self.last_osn_diagnostics = diag
                return None, diag
                
            # Angle in image coordinates (-90 is straight up)
            theta = math.degrees(math.atan2(dy, dx))
            raw_rot = theta + 90.0
            while raw_rot > 180.0:
                raw_rot -= 360.0
            while raw_rot < -180.0:
                raw_rot += 360.0
                
            # Clamp rotation to safety limit to preserve naturally tilted / horizontal signs
            if max_rot > 0.0:
                clamped_rot = max(-max_rot, min(max_rot, raw_rot))
            else:
                clamped_rot = 0.0
                
            diag["raw_rotation_deg"] = round(raw_rot, 2)
            diag["clamped_rotation_deg"] = round(clamped_rot, 2)
            diag["rotation_deg"] = round(clamped_rot, 2)
            
            # Rotate raw frame around hand center
            cx = float(lm_pts[:, 0].mean())
            cy = float(lm_pts[:, 1].mean())
            
            M = cv2.getRotationMatrix2D((cx, cy), clamped_rot, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (fw, fh), borderMode=cv2.BORDER_REFLECT)
            
            # Transform landmarks into rotated coordinate system
            rotated_pts = cv2.transform(lm_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
            rx_min, ry_min = rotated_pts.min(axis=0)
            rx_max, ry_max = rotated_pts.max(axis=0)
            
            rw = rx_max - rx_min
            rh = ry_max - ry_min
            hand_size = max(rw, rh)
            
            box_side = int(round(hand_size * sf))
            if box_side < 20:
                diag["fallback_reason"] = f"Degenerate box size ({box_side}px < 20px)"
                self.last_osn_diagnostics = diag
                return None, diag
                
            # Position box so wrist is at wrist_y_ratio height (e.g. 82% from top, near bottom margin)
            rwrist_y = float(rotated_pts[0, 1])
            rcx = float((rx_min + rx_max) / 2.0)
            
            y2 = int(round(rwrist_y + box_side * (1.0 - wyr)))
            y1 = y2 - box_side
            x1 = int(round(rcx - box_side / 2.0))
            x2 = x1 + box_side
            
            # Destination canvas
            crop = np.zeros((box_side, box_side, 3), dtype=frame.dtype)
            src_x1, src_x2 = max(0, x1), min(fw, x2)
            src_y1, src_y2 = max(0, y1), min(fh, y2)
            dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            if (src_x2 > src_x1) and (src_y2 > src_y1) and (dst_x2 > dst_x1) and (dst_y2 > dst_y1):
                crop[dst_y1:dst_y2, dst_x1:dst_x2] = rotated_frame[src_y1:src_y2, src_x1:src_x2]
                diag["osn_applied"] = True
                diag["box_side"] = box_side
                diag["hand_size"] = round(float(hand_size), 1)
                diag["hand_coverage_pct"] = round(float(hand_size / box_side * 100), 1)
                self.last_osn_diagnostics = diag
                return crop, diag
            else:
                diag["fallback_reason"] = "Crop bounding box out of bounds"
                self.last_osn_diagnostics = diag
                return None, diag
        except Exception as e:
            diag["fallback_reason"] = f"Exception: {str(e)}"
            self.last_osn_diagnostics = diag
            return None, diag

    def was_last_crop_padded(self) -> bool:
        """Check whether the most recent crop required zero-padding."""
        return self.last_crop_padded

    def get_last_crop_diagnostics(self) -> Dict:
        """Return metadata for the exact crop supplied to CNN inference."""
        return dict(self.last_crop_diagnostics)

    def get_last_osn_diagnostics(self) -> Dict:
        """Return metadata for landmark-guided orientation and scale normalization."""
        return dict(self.last_osn_diagnostics)

    def reset_tracking(self):
        """Reset temporal smoother and tracking state."""
        with self._lock:
            self.smoother.reset()
            self.last_crop_padded = False
            self.last_crop_diagnostics = {}
    
    def preprocess_for_inference(
        self,
        hand_crop: np.ndarray,
        target_size: Tuple[int, int] = (128, 128),
        normalization_mean: List[float] = None,
        normalization_std: List[float] = None,
        flip_horizontal: bool = False
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Preprocess hand crop for model inference.
        
        Args:
            hand_crop: BGR cropped hand image
            target_size: Target size (H, W)
            normalization_mean: Normalization mean
            normalization_std: Normalization std
            flip_horizontal: If True, mirror horizontally to normalize left hand to right-hand trained space
            
        Returns:
            Preprocessed tensor (1, 3, H, W) and metadata dict
        """
        if hand_crop is None or hand_crop.size == 0:
            return None, {"error": "Empty hand crop"}
        
        try:
            # If left hand, flip horizontally to map into right-hand representation trained by the CNN
            if flip_horizontal:
                hand_crop = cv2.flip(hand_crop, 1)

            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL for torchvision transforms
            from PIL import Image
            pil_image = Image.fromarray(rgb_crop)
            
            # Apply transforms matching training (no augmentation)
            # Resize -> ToTensor -> Normalize
            pil_image = pil_image.resize(target_size, Image.BILINEAR)
            
            # Convert to tensor [0, 1]
            tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Normalize
            mean = torch.tensor(normalization_mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor(normalization_std or [0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            meta = {
                "original_shape": hand_crop.shape[:2],
                "target_size": target_size,
                "normalized": True,
                "flipped_horizontal": flip_horizontal
            }
            
            return tensor, meta
            
        except Exception as e:
            return None, {"error": f"Preprocessing failed: {str(e)}"}
    
    def draw_landmarks(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Draw hand landmarks on frame for visualization."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        landmarks = detection["landmarks"]
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
        
        # Draw bbox
        bbox = detection["bbox"]
        cv2.rectangle(
            annotated,
            (bbox["x_min"], bbox["y_min"]),
            (bbox["x_max"], bbox["y_max"]),
            (0, 255, 0), 2
        )
        # Draw label
        label = f"{detection['handedness']} {detection['confidence']:.2f}"
        cv2.putText(
            annotated, label,
            (bbox["x_min"], bbox["y_min"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        return annotated
    
    def close(self):
        """Release MediaPipe resources."""
        global _hand_detector_stream
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except Exception:
                pass
        if _hand_detector_stream is self:
            _hand_detector_stream = None


def get_hand_detector() -> Optional[HandDetector]:
    """Create hand detector with default settings (static image mode)."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    try:
        return HandDetector(static_image_mode=True)
    except FileNotFoundError:
        return None


_hand_detector_stream: Optional[HandDetector] = None


def get_hand_detector_stream() -> Optional[HandDetector]:
    """Get or create singleton hand detector optimized for video stream (static image mode for API)."""
    global _hand_detector_stream
    if not MEDIAPIPE_AVAILABLE:
        return None
    if _hand_detector_stream is None:
        try:
            # API processes static images, so use IMAGE mode
            _hand_detector_stream = HandDetector(static_image_mode=True, min_detection_confidence=0.6)
        except FileNotFoundError:
            return None
    return _hand_detector_stream
