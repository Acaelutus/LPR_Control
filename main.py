"""Main entry point for the LPR access-control demo."""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logger import logger
from src.detector.detector import PlateDetector
from src.ocr.recognizer import LPRNetRecognizer, PaddleOCRRecognizer
from src.database.access_db import create_database
from src.controller.barrier_controller import create_controller
from src.pipeline.lpr_pipeline import LPRPipeline
from src.pipeline.frame_source import create_frame_source, FrameBuffer


class LPRSystem:
    """
    Main LPR System controller.
    
    Initializes all components and orchestrates pipeline execution.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize LPR system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 80)
        logger.info("License Plate Recognition System")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = Config(config_path)
        logger.info(f"Loaded config from {config_path}")
        
        # Initialize components
        self._init_detector()
        self._init_ocr()
        self._init_database()
        self._init_controller()
        self._init_pipeline()
        
        logger.info("LPR System initialized successfully")
    
    def _init_detector(self):
        """Initialize plate detector"""
        logger.info("Initializing plate detector...")
        self.vehicle_detector = None

        self.plate_detector = PlateDetector(
            weights_path=self.config.plate_detector.weights_path,
            model_type=self.config.plate_detector.model_type,
            model_format=self.config.plate_detector.model_format,
            conf_threshold=self.config.plate_detector.confidence_threshold,
            iou_threshold=self.config.plate_detector.iou_threshold,
            device=self.config.plate_detector.device
        )
        
        logger.info("Plate detector initialized")
    
    def _init_ocr(self):
        """Initialize OCR engine"""
        logger.info("Initializing OCR...")
        
        if self.config.ocr.engine == "lprnet":
            self.ocr = LPRNetRecognizer(
                model_path=self.config.ocr.model_path,
                max_plate_len=self.config.ocr.max_plate_length,
                device=self.config.ocr.device,
                use_augmentation=self.config.ocr.use_augmentation
            )
        elif self.config.ocr.engine == "paddleocr":
            self.ocr = PaddleOCRRecognizer(
                device=self.config.ocr.device
            )
        else:
            raise ValueError(f"Unknown OCR engine: {self.config.ocr.engine}")
        
        logger.info(f"OCR engine initialized: {self.config.ocr.engine}")
    
    def _init_database(self):
        """Initialize access control database"""
        logger.info("Initializing database...")
        
        self.database = create_database(
            self.config.database.type,
            db_path=self.config.database.path,
            json_path=self.config.database.json_path
        )
        
        logger.info(f"Database initialized: {self.config.database.type}")
    
    def _init_controller(self):
        """Initialize barrier controller"""
        logger.info("Initializing barrier controller...")
        
        self.controller = create_controller(
            self.config.controller.type,
            port=self.config.controller.serial_port,
            baudrate=self.config.controller.baud_rate
        )
        
        logger.info(f"Barrier controller initialized: {self.config.controller.type}")
    
    def _init_pipeline(self):
        """Initialize LPR pipeline"""
        logger.info("Initializing pipeline...")
        
        self.pipeline = LPRPipeline(
            vehicle_detector=None,
            plate_detector=self.plate_detector,
            ocr=self.ocr,
            database=self.database,
            controller=self.controller,
            plate_matching_iou_threshold=self.config.pipeline.plate_matching_iou_threshold,
            duplicate_suppression_timeout=self.config.pipeline.duplicate_detection_timeout
        )
        
        logger.info("Pipeline initialized")
    
    def run_video_demo(self, video_path: str, display: bool = True, output_path: Optional[str] = None):
        """
        Run demo on video file.
        
        Args:
            video_path: Path to input video
            display: Display output video
            output_path: Optional path to save output video
        """
        logger.info(f"Starting video demo: {video_path}")
        
        # Create frame source
        source = create_frame_source("video", path=video_path)
        frame_buffer = FrameBuffer(
            source,
            frame_skip=self.config.pipeline.frame_skip,
            max_fps=self.config.pipeline.max_frame_rate
        )
        
        # Video writer for output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = source.get_fps()
            width = int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output video will be saved to {output_path}")
        
        frame_count = 0
        processed_count = 0
        
        try:
            while frame_buffer.is_open():
                should_process, frame = frame_buffer.read()
                
                if frame is None:
                    break
                
                frame_count += 1
                
                if should_process:
                    processed_count += 1
                    result = self.pipeline.process_frame(frame)
                    
                    # Visualize results
                    vis_frame = self._visualize_results(frame, result)
                    
                    if display:
                        cv2.imshow("LPR System", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if writer:
                        writer.write(vis_frame)
            
            logger.info(
                f"Demo completed: {frame_count} frames total, "
                f"{processed_count} frames processed"
            )
        
        finally:
            frame_buffer.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
    
    def run_rtsp_stream(self, rtsp_url: str, display: bool = True):
        """
        Run on RTSP stream.
        
        Args:
            rtsp_url: RTSP stream URL
            display: Display output
        """
        logger.info(f"Starting RTSP stream: {rtsp_url}")
        
        # Create frame source
        source = create_frame_source("rtsp", url=rtsp_url)
        frame_buffer = FrameBuffer(
            source,
            frame_skip=self.config.pipeline.frame_skip,
            max_fps=self.config.pipeline.max_frame_rate
        )
        
        frame_count = 0
        processed_count = 0
        
        try:
            while frame_buffer.is_open():
                should_process, frame = frame_buffer.read()
                
                if frame is None:
                    logger.warning("Lost connection to RTSP stream")
                    break
                
                frame_count += 1
                
                if should_process:
                    processed_count += 1
                    result = self.pipeline.process_frame(frame)
                    
                    vis_frame = self._visualize_results(frame, result)
                    
                    if display:
                        cv2.imshow("LPR RTSP Stream", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        
        except KeyboardInterrupt:
            logger.info("RTSP stream interrupted by user")
        
        finally:
            frame_buffer.release()
            if display:
                cv2.destroyAllWindows()
            
            logger.info(
                f"Stream ended: {frame_count} frames total, "
                f"{processed_count} frames processed"
            )
    
    def _visualize_results(self, frame, result) -> np.ndarray:
        """
        Draw detection and recognition results on frame.
        
        Args:
            frame: Input frame
            result: Pipeline result
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # Draw vehicles
        for vehicle in result['vehicles']:
            x1, y1, x2, y2 = vehicle.denormalize(width, height)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw plates and text
        for plate_info in result['recognized_plates']:
            x1, y1, x2, y2 = (
                int(plate_info['bbox'][0] * width),
                int(plate_info['bbox'][1] * height),
                int(plate_info['bbox'][2] * width),
                int(plate_info['bbox'][3] * height)
            )
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw plate text
            text = plate_info['text']
            cv2.putText(
                vis_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        # Draw access control results
        y_offset = 30
        for result_item in result['results']:
            if result_item['authorized']:
                color = (0, 255, 0)
                text = f"OK {result_item['plate']} - GRANTED"
            else:
                color = (0, 0, 255)
                text = f"NO {result_item['plate']} - DENIED"
            
            cv2.putText(
                vis_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            y_offset += 25
        
        return vis_frame


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="License Plate Recognition System"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")
    
    # Video demo mode
    video_parser = subparsers.add_parser("video", help="Process video file")
    video_parser.add_argument(
        "video_path",
        help="Path to video file"
    )
    video_parser.add_argument(
        "--output",
        default=None,
        help="Output video path"
    )
    video_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video"
    )
    
    # RTSP mode
    rtsp_parser = subparsers.add_parser("rtsp", help="Process RTSP stream")
    rtsp_parser.add_argument(
        "rtsp_url",
        help="RTSP stream URL"
    )
    
    args = parser.parse_args()
    
    if args.mode is None:
        default_video = Path("data/videos/demo.mp4")
        print("No mode specified. Starting default demo video.")
        print(f"Video: {default_video}")
        print("Press 'q' in the video window to stop.")
        args.mode = "video"
        args.video_path = str(default_video)
        args.output = None
        args.no_display = False

    # Initialize system
    system = LPRSystem(args.config)

    # Run appropriate mode
    if args.mode == "video":
        system.run_video_demo(
            args.video_path,
            display=not args.no_display,
            output_path=args.output
        )
    elif args.mode == "rtsp":
        system.run_rtsp_stream(args.rtsp_url)


if __name__ == "__main__":
    main()
