"""License plate OCR using LPRNet or PaddleOCR."""

import torch
import numpy as np
import cv2
from typing import Optional
from pathlib import Path
from src.utils.logger import logger
from src.utils.plate_text import normalize_plate_text


CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "_"
]


class PlateRecognizer:
    """
    Base interface for plate recognition.
    Subclass this to support different OCR engines.
    """
    
    def recognize(self, plate_crop: np.ndarray) -> str:
        """
        Recognize text from a license plate crop.
        
        Args:
            plate_crop: Cropped plate image (H, W, 3) BGR format
            
        Returns:
            Recognized plate text as string
        """
        raise NotImplementedError


class LPRNetRecognizer(PlateRecognizer):
    """License plate OCR with LPRNet."""
    
    def __init__(
        self,
        model_path: str,
        max_plate_len: int = 9,
        device: str = "auto",
        use_augmentation: bool = True
    ):
        """
        Initialize LPRNet recognizer.
        
        Args:
            model_path: Path to LPRNet weights (.pth file)
            max_plate_len: Maximum license plate length
            device: "cuda", "cpu", or "auto"
            use_augmentation: Apply augmentation during inference for robustness
        """
        self.model_path = Path(model_path)
        self.max_plate_len = max_plate_len
        self.device = self._resolve_device(device)
        self.use_augmentation = use_augmentation
        
        self.model = self._load_model()
        logger.info(f"LPRNetRecognizer initialized on {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load LPRNet model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"LPRNet weights not found: {self.model_path}")
        
        from src.ocr.lpr_net import build_lprnet
        
        model = build_lprnet(
            lpr_max_len=self.max_plate_len,
            phase='test',
            class_num=len(CHARS),
            dropout_rate=0
        )
        
        # Load weights
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded LPRNet from {self.model_path}")
        return model
    
    def recognize(self, plate_crop: np.ndarray) -> str:
        """
        Recognize plate text using LPRNet.
        
        Args:
            plate_crop: Cropped plate image
            
        Returns:
            Recognized text
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""
        
        # Preprocess
        image = self._preprocess(plate_crop)
        
        # Inference
        with torch.no_grad():
            preds = self.model(image)
        
        # Decode predictions
        text = self._decode(preds)
        return normalize_plate_text(text)
    
    def _preprocess(self, plate_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess plate image for LPRNet.
        
        LPRNet expects: (H=24, W=94, 3 channels)
        """
        image = plate_crop.copy()
        
        # Apply augmentation for robustness
        if self.use_augmentation and np.random.rand() > 0.5:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.95, 1.05)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Resize to model input size
        image = cv2.resize(image, (94, 24))
        
        # Normalize
        image = image.astype("float32")
        image -= 127.5
        image *= 0.0078125  # 1/128
        
        # Convert to tensor and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)
        
        return image
    
    def _decode(self, preds: torch.Tensor) -> str:
        """
        Decode LPRNet predictions to text.
        
        LPRNet outputs: (batch_size, num_chars, seq_len)
        """
        preds = preds.cpu().detach().numpy()
        text = ""
        
        for pred in preds:  # batch
            pred_labels = []
            for j in range(pred.shape[1]):
                pred_labels.append(np.argmax(pred[:, j], axis=0))
            
            # Remove duplicates and blanks
            pre_c = pred_labels[0]
            blank_id = len(CHARS) - 1
            
            if pre_c != blank_id:
                text += CHARS[pre_c]
            
            for c in pred_labels[1:]:
                if (pre_c == c) or (c == blank_id):
                    if c == blank_id:
                        pre_c = c
                    continue
                text += CHARS[c]
                pre_c = c
        
        return text


class PaddleOCRRecognizer(PlateRecognizer):
    """Alternative OCR engine using PaddleOCR."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize PaddleOCR recognizer.
        
        Args:
            device: "cuda" or "cpu"
        """
        self.device = self._resolve_device(device)
        self.model = self._load_model()
        logger.info(f"PaddleOCRRecognizer initialized on {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "gpu" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load PaddleOCR model"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr not installed. Install with: pip install paddleocr"
            )
        
        try:
            model = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device=self.device
            )
        except TypeError:
            model = PaddleOCR(use_angle_cls=True, lang='en')
        return model
    
    def recognize(self, plate_crop: np.ndarray) -> str:
        """
        Recognize plate text using PaddleOCR.
        
        Args:
            plate_crop: Cropped plate image
            
        Returns:
            Recognized text
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""
        
        try:
            result = self.model.predict(
                plate_crop,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True
            )
        except TypeError:
            result = self.model.ocr(plate_crop, cls=True)

        return normalize_plate_text(self._extract_text(result))

    def _extract_text(self, result) -> str:
        """Extract text from PaddleOCR 2.x or 3.x result objects."""
        if not result:
            return ""

        texts = []
        for item in result:
            data = None
            if isinstance(item, dict):
                data = item
            elif hasattr(item, "json"):
                data = item.json() if callable(item.json) else item.json
            elif hasattr(item, "dict"):
                data = item.dict() if callable(item.dict) else item.dict

            if isinstance(data, dict):
                payload = data.get("res", data)
                rec_texts = payload.get("rec_texts") or payload.get("texts")
                if rec_texts:
                    texts.extend(str(text) for text in rec_texts)
                continue

            if isinstance(item, (list, tuple)):
                for line in item:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        value = line[1]
                        if isinstance(value, (list, tuple)) and value:
                            texts.append(str(value[0]))
                        else:
                            texts.append(str(value))

        return "".join(texts)
