"""YOLO detector wrappers for vehicle and license plate localization."""

import contextlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.utils.logger import logger


@dataclass
class Detection:
    """Result of a single detection"""

    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # Normalized [x1, y1, x2, y2]

    def denormalize(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        x1, y1, x2, y2 = self.bbox
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )


class YOLODetector:
    """Backend-neutral YOLO detector."""

    def __init__(
        self,
        weights_path: str,
        model_format: str = "auto",
        model_type: str = "yolov8",
        model_variant: str = "s",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "auto",
        allow_pretrained_fallback: bool = False,
    ):
        self.weights_path = Path(weights_path)
        self.model_format = (model_format or "auto").lower()
        self.model_type = model_type
        self.model_variant = model_variant
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)
        self.allow_pretrained_fallback = allow_pretrained_fallback

        self.backend = None
        self.model = self._load_model()
        logger.info(
            f"YOLODetector initialized: backend={self.backend}, "
            f"weights={self.weights_path}, device={self.device}"
        )

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _torchhub_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device.startswith("cuda:"):
            return self.device.split(":", 1)[1]
        if self.device == "cuda":
            return "0"
        return self.device

    def _torch_device(self) -> str:
        if self.device == "cuda":
            return "cuda:0"
        return self.device

    def _load_model(self):
        requested = self.model_format
        if requested in {"auto", "yolov8", "yolov11", "ultralytics"}:
            try:
                return self._load_ultralytics_model()
            except Exception as exc:
                if requested != "auto":
                    raise
                logger.warning(
                    "Ultralytics loader failed, trying YOLOv5 loader: "
                    f"{type(exc).__name__}: {exc}"
                )

        if requested in {"auto", "yolov5", "v5"}:
            return self._load_yolov5_model()

        raise ValueError(
            f"Unknown YOLO model_format '{self.model_format}'. "
            "Use 'auto', 'yolov8', 'yolov11', or 'yolov5'."
        )

    def _load_ultralytics_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        if self.weights_path.exists():
            model = YOLO(str(self.weights_path))
            logger.info(f"Loaded Ultralytics YOLO weights from {self.weights_path}")
        elif self.allow_pretrained_fallback:
            model_name = f"{self.model_type}{self.model_variant}.pt"
            model = YOLO(model_name)
            logger.info(f"Loaded pretrained {model_name}")
        else:
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        model.to(self.device)
        self.backend = "ultralytics"
        return model

    def _load_yolov5_model(self):
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLOv5 weights not found: {self.weights_path}")

        capture = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                try:
                    model = torch.hub.load(
                        "ultralytics/yolov5",
                        "custom",
                        path=str(self.weights_path),
                        source="github",
                        trust_repo=True,
                        verbose=False,
                        device=self._torchhub_device(),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to load YOLOv5 weights. The first YOLOv5 run "
                        "requires access to the ultralytics/yolov5 torch.hub "
                        "repo, unless it is already cached locally."
                    ) from exc

        model.conf = self.conf_threshold
        model.iou = self.iou_threshold
        model.max_det = 20
        model.to(self._torch_device())
        self.backend = "yolov5"
        logger.info(f"Loaded YOLOv5 weights from {self.weights_path}")
        return model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.backend == "ultralytics":
            try:
                return self._detect_ultralytics(frame)
            except Exception as exc:
                if self.model_format == "auto" and self.weights_path.exists():
                    logger.warning(
                        "Ultralytics prediction failed, switching to YOLOv5 "
                        f"loader: {type(exc).__name__}: {exc}"
                    )
                    self.model = self._load_yolov5_model()
                    return self._detect_yolov5(frame)
                raise
        if self.backend == "yolov5":
            return self._detect_yolov5(frame)
        raise RuntimeError("Detector backend is not initialized")

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxyn.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes))

        for bbox, conf, class_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = bbox
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections

    def _detect_yolov5(self, frame: np.ndarray) -> List[Detection]:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.model(frame)

        detections = []
        if not hasattr(results, "xyxyn") or not results.xyxyn:
            return detections

        rows = results.xyxyn[0].detach().cpu().numpy()
        for row in rows:
            x1, y1, x2, y2, conf, class_id = row[:6]
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        return detections


class VehicleDetector(YOLODetector):
    """Optional vehicle detector retained for compatibility."""

    def __init__(
        self,
        weights_path: str,
        model_type: str = "yolov8",
        model_variant: str = "m",
        model_format: str = "auto",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        super().__init__(
            weights_path=weights_path,
            model_format=model_format or model_type,
            model_type=model_type,
            model_variant=model_variant,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            allow_pretrained_fallback=True,
        )
        logger.info(f"VehicleDetector initialized on {self.device}")


class PlateDetector(YOLODetector):
    """License plate detector supporting YOLOv5 and Ultralytics YOLO weights."""

    def __init__(
        self,
        weights_path: str,
        model_type: str = "yolov8",
        model_format: str = "auto",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        super().__init__(
            weights_path=weights_path,
            model_format=model_format or model_type,
            model_type=model_type,
            model_variant="s",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            allow_pretrained_fallback=False,
        )
        logger.info(f"PlateDetector initialized on {self.device}")
