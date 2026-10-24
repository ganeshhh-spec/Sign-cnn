"""
Prediction stabilization and translation service.
Maintains prediction history, applies smoothing, handles SPACE/DELETE/CLEAR.
"""

from collections import deque
from typing import List, Dict, Optional, Deque, Any
from dataclasses import dataclass
from enum import Enum
import time


class ActionType(Enum):
    APPEND = "append"
    DELETE = "delete"
    SPACE = "space"
    CLEAR = "clear"
    NONE = "none"


@dataclass
class StabilizedPrediction:
    """Result of stabilization."""
    action: ActionType
    character: str
    confidence: float
    raw_prediction: str
    raw_confidence: float
    is_stable: bool


class PredictionStabilizer:
    """
    Stabilizes predictions using temporal smoothing.
    
    Logic:
    - Maintain sliding window of recent predictions and confidences.
    - Require min_consistent consecutive predictions of the SAME class.
    - EVERY ONE of the min_consistent predictions must have confidence >= confidence_threshold.
    - Low-confidence frames (< confidence_threshold) reset the consistent streak.
    - Handle special classes: space, del, nothing with duplicate guards.
    - Prevent duplicate characters while a sign is held steady.
    - Require release (neutral sign 'nothing' or no-hand detected for >= 2 frames) before
      the same character can be appended again.
    - Cooldown period prevents rapid accidental appends during hand transitions.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        min_consistent: int = 3,
        confidence_threshold: float = 0.7,
        special_classes: Dict[str, str] = None,
        duplicate_guard: bool = True,
        cooldown_seconds: float = 0.8
    ):
        self.window_size = window_size
        self.min_consistent = min_consistent
        self.confidence_threshold = confidence_threshold
        self.duplicate_guard = duplicate_guard
        self.cooldown_seconds = cooldown_seconds
        
        # Special class mappings
        self.special_classes = special_classes or {
            "space": " ",
            "del": "DELETE",
            "nothing": "NOTHING"
        }
        
        # State
        self.prediction_window: Deque[str] = deque(maxlen=window_size)
        self.confidence_window: Deque[float] = deque(maxlen=window_size)
        self.frame_window: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.last_appended: Optional[str] = None
        self.last_action_time: float = 0.0
        self.held_char_count: int = 0
        self.no_hand_count: int = 0
        
    def process_no_hand(self) -> StabilizedPrediction:
        """
        Process a frame where no hand was detected.
        Clears prediction window. After 2+ consecutive frames of no hand,
        resets duplicate guard and cooldown so the next gesture can be accepted.
        """
        self.no_hand_count += 1
        self.prediction_window.clear()
        self.confidence_window.clear()
        self.frame_window.clear()
        
        if self.no_hand_count >= 2:
            self.last_appended = None
            self.held_char_count = 0
            self.last_action_time = 0.0
            
        return StabilizedPrediction(
            action=ActionType.NONE,
            character="",
            confidence=0.0,
            raw_prediction="",
            raw_confidence=0.0,
            is_stable=False
        )

    def process(self, prediction: str, confidence: float, frame_metadata: Optional[Dict[str, Any]] = None) -> StabilizedPrediction:
        """
        Process a new prediction and return stabilized result.
        
        Args:
            prediction: Raw class label from model
            confidence: Confidence score (0-1)
            
        Returns:
            StabilizedPrediction with action to take
        """
        current_time = time.time()
        self.no_hand_count = 0
        
        # Low-confidence frame check:
        # If confidence is below threshold, it cannot form part of a stable sequence.
        if confidence < self.confidence_threshold:
            self.prediction_window.clear()
            self.confidence_window.clear()
            self.frame_window.clear()
            return StabilizedPrediction(
                action=ActionType.NONE,
                character="",
                confidence=confidence,
                raw_prediction=prediction,
                raw_confidence=confidence,
                is_stable=False
            )
            
        # Add to windows
        self.prediction_window.append(prediction)
        self.confidence_window.append(confidence)
        self.frame_window.append({
            "timestamp": current_time,
            "predicted_class": prediction,
            "confidence": confidence,
            **(frame_metadata or {}),
        })
        
        # Not enough history yet
        if len(self.prediction_window) < self.min_consistent:
            return StabilizedPrediction(
                action=ActionType.NONE,
                character="",
                confidence=confidence,
                raw_prediction=prediction,
                raw_confidence=confidence,
                is_stable=False
            )
            
        recent_preds = list(self.prediction_window)[-self.min_consistent:]
        recent_conf = list(self.confidence_window)[-self.min_consistent:]
        
        # All predictions must be the exact same class
        if len(set(recent_preds)) != 1:
            return StabilizedPrediction(
                action=ActionType.NONE,
                character="",
                confidence=max(recent_conf),
                raw_prediction=prediction,
                raw_confidence=confidence,
                is_stable=False
            )
            
        # ALL predictions must meet or exceed confidence threshold
        if any(c < self.confidence_threshold for c in recent_conf):
            return StabilizedPrediction(
                action=ActionType.NONE,
                character="",
                confidence=sum(recent_conf) / len(recent_conf),
                raw_prediction=prediction,
                raw_confidence=confidence,
                is_stable=False
            )
            
        avg_confidence = sum(recent_conf) / len(recent_conf)
        stable_class = recent_preds[0]
        
        # Handle special classes
        if stable_class in self.special_classes:
            action_type = self.special_classes[stable_class]
            
            if action_type == "NOTHING":
                # NOTHING is neutral: resets duplicate guard, held counter, and action cooldown
                self.last_appended = None
                self.held_char_count = 0
                self.last_action_time = 0.0
                return StabilizedPrediction(
                    action=ActionType.NONE,
                    character="",
                    confidence=avg_confidence,
                    raw_prediction=stable_class,
                    raw_confidence=confidence,
                    is_stable=True
                )
            elif action_type == " ":
                # Space gesture with duplicate guard
                if self.duplicate_guard and self.last_appended == "space":
                    self.held_char_count += 1
                    return StabilizedPrediction(
                        action=ActionType.NONE,
                        character="",
                        confidence=avg_confidence,
                        raw_prediction=stable_class,
                        raw_confidence=confidence,
                        is_stable=False
                    )
                self.last_appended = "space"
                self.held_char_count = 1
                self.last_action_time = current_time
                return StabilizedPrediction(
                    action=ActionType.SPACE,
                    character=" ",
                    confidence=avg_confidence,
                    raw_prediction=stable_class,
                    raw_confidence=confidence,
                    is_stable=True
                )
            elif action_type == "DELETE":
                # Delete gesture with duplicate guard
                if self.duplicate_guard and self.last_appended == "del":
                    self.held_char_count += 1
                    return StabilizedPrediction(
                        action=ActionType.NONE,
                        character="",
                        confidence=avg_confidence,
                        raw_prediction=stable_class,
                        raw_confidence=confidence,
                        is_stable=False
                    )
                self.last_appended = "del"
                self.held_char_count = 1
                self.last_action_time = current_time
                return StabilizedPrediction(
                    action=ActionType.DELETE,
                    character="",
                    confidence=avg_confidence,
                    raw_prediction=stable_class,
                    raw_confidence=confidence,
                    is_stable=True
                )
        
        # Regular letter:
        # 1. Duplicate guard check for same letter
        if self.duplicate_guard and stable_class == self.last_appended:
            self.held_char_count += 1
            return StabilizedPrediction(
                action=ActionType.NONE,
                character="",
                confidence=avg_confidence,
                raw_prediction=stable_class,
                raw_confidence=confidence,
                is_stable=False
            )
            
        # 2. Cooldown check between different letters (to avoid spurious rapid transitions)
        if self.cooldown_seconds > 0 and self.last_action_time > 0:
            if (current_time - self.last_action_time) < self.cooldown_seconds:
                return StabilizedPrediction(
                    action=ActionType.NONE,
                    character="",
                    confidence=avg_confidence,
                    raw_prediction=stable_class,
                    raw_confidence=confidence,
                    is_stable=False
                )
                
        # Accept stable letter
        self.last_appended = stable_class
        self.held_char_count = 1
        self.last_action_time = current_time
        
        return StabilizedPrediction(
            action=ActionType.APPEND,
            character=stable_class,
            confidence=avg_confidence,
            raw_prediction=stable_class,
            raw_confidence=confidence,
            is_stable=True
        )
    
    def reset(self):
        """Reset stabilizer state."""
        self.prediction_window.clear()
        self.confidence_window.clear()
        self.frame_window.clear()
        self.last_appended = None
        self.held_char_count = 0
        self.last_action_time = 0.0
        self.no_hand_count = 0
    
    def force_clear(self):
        """Force clear (for explicit CLEAR action)."""
        self.reset()
    
    def get_state(self) -> Dict:
        """Get current stabilizer state for debugging."""
        return {
            "window_size": len(self.prediction_window),
            "last_appended": self.last_appended,
            "held_char_count": self.held_char_count,
            "recent_predictions": list(self.prediction_window),
            "recent_frames": list(self.frame_window),
            "confidence_threshold": self.confidence_threshold,
            "min_consistent": self.min_consistent,
            "no_hand_count": self.no_hand_count
            ,"last_action_time": self.last_action_time
        }


class TextBuffer:
    """Manages the accumulated text with SPACE, DELETE, CLEAR operations."""
    
    def __init__(self, max_length: int = 1000):
        self.buffer: List[str] = []
        self.max_length = max_length
    
    def append(self, char: str) -> str:
        """Append character to buffer."""
        if len(self.buffer) < self.max_length:
            self.buffer.append(char)
        return self.get_text()
    
    def delete(self) -> str:
        """Delete last character."""
        if self.buffer:
            self.buffer.pop()
        return self.get_text()
    
    def space(self) -> str:
        """Add space."""
        return self.append(" ")
    
    def clear(self) -> str:
        """Clear entire buffer."""
        self.buffer.clear()
        return ""
    
    def set_text(self, text: str):
        """Set buffer from string."""
        self.buffer = list(text[:self.max_length])
    
    def get_text(self) -> str:
        """Get current text."""
        return "".join(self.buffer)
    
    def get_state(self) -> Dict:
        """Get buffer state."""
        return {
            "text": self.get_text(),
            "length": len(self.buffer),
            "max_length": self.max_length
        }


# Global instances
_stabilizer: Optional[PredictionStabilizer] = None
_text_buffer: Optional[TextBuffer] = None


def get_stabilizer(
    window_size: int = 5,
    min_consistent: int = 3,
    confidence_threshold: float = 0.7,
    duplicate_guard: bool = True,
    cooldown_seconds: float = 0.8
) -> PredictionStabilizer:
    """Get or create global stabilizer instance."""
    global _stabilizer
    if _stabilizer is None:
        _stabilizer = PredictionStabilizer(
            window_size=window_size,
            min_consistent=min_consistent,
            confidence_threshold=confidence_threshold,
            duplicate_guard=duplicate_guard,
            cooldown_seconds=cooldown_seconds
        )
    else:
        _stabilizer.window_size = window_size
        _stabilizer.min_consistent = min_consistent
        _stabilizer.confidence_threshold = confidence_threshold
        _stabilizer.duplicate_guard = duplicate_guard
        _stabilizer.cooldown_seconds = cooldown_seconds
    return _stabilizer


def get_text_buffer() -> TextBuffer:
    """Get or create global text buffer instance."""
    global _text_buffer
    if _text_buffer is None:
        _text_buffer = TextBuffer()
    return _text_buffer


def reset_translation_state():
    """Reset both stabilizer and text buffer."""
    global _stabilizer, _text_buffer
    if _stabilizer:
        _stabilizer.reset()
    if _text_buffer:
        _text_buffer.clear()
