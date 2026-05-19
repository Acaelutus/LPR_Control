"""Frame sources for video files, RTSP streams, and webcams."""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import time

from src.utils.logger import logger


class FrameSource(ABC):
    """Abstract base class for frame sources"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if source is open"""
        pass
    
    @abstractmethod
    def release(self):
        """Release resources"""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second"""
        pass
    
    @abstractmethod
    def get_frame_count(self) -> int:
        """Get total frame count (if available)"""
        pass


class VideoFileSource(FrameSource):
    """
    Read frames from a video file.
    
    Supports MP4, AVI, MOV, etc.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video file source.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(
            f"VideoFileSource initialized: {self.video_path.name} "
            f"({self._frame_count} frames, {self._fps:.2f} FPS)"
        )
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video"""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def is_open(self) -> bool:
        """Check if video is open"""
        return self.cap.isOpened()
    
    def release(self):
        """Release video resource"""
        self.cap.release()
        logger.info("VideoFileSource released")
    
    def get_fps(self) -> float:
        """Get video FPS"""
        return self._fps
    
    def get_frame_count(self) -> int:
        """Get total frames"""
        return self._frame_count


class RTSPSource(FrameSource):
    """
    Stream frames from RTSP source.
    
    Real-time streaming protocol for IP cameras and streaming servers.
    """
    
    def __init__(self, rtsp_url: str, buffer_size: int = 1):
        """
        Initialize RTSP source.
        
        Args:
            rtsp_url: RTSP stream URL
            buffer_size: OpenCV buffer size (lower = lower latency)
        """
        self.rtsp_url = rtsp_url
        
        self.cap = cv2.VideoCapture(rtsp_url)
        
        # Set buffer size for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        if not self.cap.isOpened():
            logger.warning(f"Failed to open RTSP stream: {rtsp_url}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30  # Default assumption
        
        logger.info(f"RTSPSource initialized: {rtsp_url} ({self._fps:.2f} FPS)")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from stream"""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def is_open(self) -> bool:
        """Check if stream is connected"""
        return self.cap.isOpened()
    
    def release(self):
        """Release stream"""
        self.cap.release()
        logger.info("RTSPSource released")
    
    def get_fps(self) -> float:
        """Get stream FPS"""
        return self._fps
    
    def get_frame_count(self) -> int:
        """Not applicable for RTSP streams"""
        return -1


class WebcamSource(FrameSource):
    """
    Capture frames from webcam.
    
    Useful for testing on laptops and development machines.
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize webcam source.
        
        Args:
            camera_index: Camera device index (0 for default)
        """
        self.camera_index = camera_index
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30
        
        logger.info(f"WebcamSource initialized: Camera {camera_index} ({self._fps:.2f} FPS)")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from webcam"""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def is_open(self) -> bool:
        """Check if camera is open"""
        return self.cap.isOpened()
    
    def release(self):
        """Release camera"""
        self.cap.release()
        logger.info("WebcamSource released")
    
    def get_fps(self) -> float:
        """Get camera FPS"""
        return self._fps
    
    def get_frame_count(self) -> int:
        """Not applicable for webcams"""
        return -1


class FrameBuffer:
    """
    Wrapper around frame source with frame skipping and FPS control.
    
    Useful for:
    - Skipping frames for faster processing
    - Controlling processing FPS independent of source FPS
    """
    
    def __init__(
        self,
        source: FrameSource,
        frame_skip: int = 1,
        max_fps: int = 30
    ):
        """
        Initialize frame buffer.
        
        Args:
            source: Underlying frame source
            frame_skip: Skip every N frames (1 = process all)
            max_fps: Maximum processing FPS
        """
        self.source = source
        self.frame_skip = frame_skip
        self.max_fps = max_fps
        self.frame_count = 0
        self.last_process_time = 0
        
        logger.info(
            f"FrameBuffer initialized: "
            f"skip={frame_skip}, max_fps={max_fps}"
        )
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame, applying skip and FPS control.
        
        Returns:
            (should_process, frame)
        """
        frame_delay = 1.0 / self.max_fps
        
        while True:
            ret, frame = self.source.read()
            self.frame_count += 1
            
            if not ret:
                return False, None
            
            # Check if we should skip this frame
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # FPS control
            current_time = time.time()
            elapsed = current_time - self.last_process_time
            
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            
            self.last_process_time = time.time()
            return True, frame
    
    def release(self):
        """Release underlying source"""
        self.source.release()
    
    def is_open(self) -> bool:
        """Check if source is open"""
        return self.source.is_open()


def create_frame_source(
    source_type: str,
    **kwargs
) -> FrameSource:
    """
    Factory function to create appropriate frame source.
    
    Args:
        source_type: "video", "rtsp", or "webcam"
        **kwargs: Additional arguments
        
    Returns:
        FrameSource instance
    """
    if source_type == "video":
        return VideoFileSource(kwargs.get("path"))
    elif source_type == "rtsp":
        return RTSPSource(kwargs.get("url"))
    elif source_type == "webcam":
        return WebcamSource(kwargs.get("camera_index", 0))
    else:
        raise ValueError(f"Unknown source type: {source_type}")
