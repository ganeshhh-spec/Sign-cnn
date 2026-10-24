"""
Model loading and inference service for ASL CNN.
Loads checkpoint with weights_only=True for safety.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.model import ASLCNN


class ModelLoader:
    """Handles model loading, checkpoint verification, and inference."""
    
    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.checkpoint_path = checkpoint_path or str(PROJECT_ROOT / "results" / "models" / "checkpoint_best.pth")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[ASLCNN] = None
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.input_size = (3, 128, 128)
        self._loaded = False
        self._load_time = 0.0
    
    def load(self) -> bool:
        """Load model checkpoint and metadata."""
        if self._loaded and self.model is not None:
            return True
        import time
        start = time.time()
        
        try:
            # Load deployment info for metadata
            deploy_path = PROJECT_ROOT / "results" / "models" / "deployment_info.json"
            if deploy_path.exists():
                with open(deploy_path, 'r') as f:
                    deploy_info = json.load(f)
                self.classes = deploy_info.get("classes", [])
                self.class_to_idx = deploy_info.get("class_to_idx", {})
                self.idx_to_class = {int(k): v for k, v in deploy_info.get("idx_to_class", {}).items()}
                self.normalization_mean = deploy_info.get("normalization", {}).get("mean", self.normalization_mean)
                self.normalization_std = deploy_info.get("normalization", {}).get("std", self.normalization_std)
                self.input_size = tuple(deploy_info.get("input_size", self.input_size))
            
            # Load checkpoint with weights_only=True for safety
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            
            # Create model and load weights
            num_classes = len(self.classes) if self.classes else 29
            self.model = ASLCNN(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Verify model output
            with torch.no_grad():
                test_input = torch.randn(1, *self.input_size).to(self.device)
                output = self.model(test_input)
                assert output.shape == (1, num_classes), f"Expected (1, {num_classes}), got {output.shape}"
            
            self._loaded = True
            self._load_time = time.time() - start
            print(f"Model loaded successfully on {self.device} in {self._load_time:.3f}s")
            print(f"Classes: {len(self.classes)}, Output shape: {output.shape}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._loaded = False
            return False
    
    def predict(self, tensor: torch.Tensor) -> Tuple[List[Dict], torch.Tensor]:
        """
        Run inference on preprocessed tensor.
        
        Args:
            tensor: Input tensor of shape (B, 3, 128, 128)
            
        Returns:
            List of top-3 predictions per sample, raw logits
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        tensor = tensor.to(self.device)
        
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        results = []
        for i in range(logits.shape[0]):
            probs_i = probs[i]
            top3_probs, top3_idx = torch.topk(probs_i, 3)
            
            predictions = []
            for rank, (prob, idx) in enumerate(zip(top3_probs.tolist(), top3_idx.tolist())):
                predictions.append({
                    "rank": rank + 1,
                    "class_index": idx,
                    "class_label": self.idx_to_class.get(idx, "unknown"),
                    "confidence": round(prob, 4)
                })
            results.append(predictions)
        
        return results, logits
    
    def get_info(self) -> Dict:
        """Get model and service information."""
        return {
            "loaded": self._loaded,
            "device": self.device,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "input_size": self.input_size,
            "load_time_seconds": round(self._load_time, 3),
            "checkpoint_path": self.checkpoint_path,
            "normalization": {
                "mean": self.normalization_mean,
                "std": self.normalization_std
            }
        }
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_preprocessing_config(self) -> Dict:
        """Get preprocessing configuration for inference."""
        return {
            "input_size": [self.input_size[1], self.input_size[2]],
            "normalization": {
                "mean": self.normalization_mean,
                "std": self.normalization_std
            },
            "format": "RGB"
        }


# Global instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def initialize_model() -> bool:
    """Initialize model on startup."""
    loader = get_model_loader()
    return loader.load()


def cleanup_model():
    """Cleanup model on shutdown."""
    global _model_loader
    if _model_loader is not None:
        _model_loader = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()