"""
Configuration management for the LPR system.
Loads and validates config from YAML file.
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Vehicle detector configuration"""
    model_type: str = "yolov8"
    model_variant: str = "s"
    weights_path: str = "data/weights/yolo8s.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    device: str = "auto"
    model_format: str = "auto"


@dataclass
class PlateDetectorConfig:
    """License plate detector configuration"""
    model_type: str
    weights_path: str
    confidence_threshold: float
    iou_threshold: float
    device: str
    model_format: str = "auto"


@dataclass
class OCRConfig:
    """OCR engine configuration"""
    engine: str
    model_path: str
    max_plate_length: int
    device: str
    use_augmentation: bool


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str
    path: str
    json_path: str


@dataclass
class ControllerConfig:
    """Barrier controller configuration"""
    type: str
    serial_port: str
    baud_rate: int
    simulation_mode: bool


@dataclass
class PipelineConfig:
    """Pipeline settings"""
    frame_skip: int
    max_frame_rate: int
    plate_matching_iou_threshold: float
    duplicate_detection_timeout: float


@dataclass
class InputConfig:
    """Input source configuration"""
    video_file: str
    rtsp_stream: str
    camera_index: int


@dataclass
class OutputConfig:
    """Output configuration"""
    display_detections: bool
    save_detections: bool
    detection_output_dir: str
    fps: int


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Resolve config paths relative to the repository root, not src/
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / config_path

        self.config_path = config_path
        self.raw_config = self._load_config()
        self._parse_config()
    
    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            project_root = self.config_path.parent.parent
            path = project_root / path
        return str(path)

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _parse_config(self):
        """Parse config sections into dataclasses"""
        cfg = self.raw_config
        
        self.device = cfg.get('device', {}).get('use_cuda', True)
        
        self.detector = DetectorConfig(**cfg.get('detector', {}))
        self.plate_detector = PlateDetectorConfig(**cfg.get('plate_detector', {}))
        self.ocr = OCRConfig(**cfg.get('ocr', {}))
        self.database = DatabaseConfig(**cfg.get('database', {}))
        self.controller = ControllerConfig(**cfg.get('controller', {}))
        self.pipeline = PipelineConfig(**cfg.get('pipeline', {}))
        self.input = InputConfig(**cfg.get('input', {}))
        self.output = OutputConfig(**cfg.get('output', {}))

        self.detector.weights_path = self._resolve_path(self.detector.weights_path)
        self.plate_detector.weights_path = self._resolve_path(self.plate_detector.weights_path)
        self.ocr.model_path = self._resolve_path(self.ocr.model_path)
        self.database.path = self._resolve_path(self.database.path)
        self.database.json_path = self._resolve_path(self.database.json_path)
        self.output.detection_output_dir = self._resolve_path(self.output.detection_output_dir)
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  detector={self.detector}\n"
            f"  ocr={self.ocr}\n"
            f"  database={self.database}\n"
            f"  controller={self.controller}\n"
            f")"
        )
